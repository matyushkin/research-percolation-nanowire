#!/usr/bin/env python3
"""Finite-size scaling for bridge percolation threshold.

Measures η_c(L) for both parallel and Voronoi cracks at different domain sizes.
Extrapolates to η_c(∞) and estimates critical exponent ν.

Theory: η_c(L) = η_c(∞) + a·L^{-1/ν}

Usage:
    uv run python scripts/run_finite_size_scaling.py [--trials 200]
    uv run python scripts/run_finite_size_scaling.py --resume

Results saved to: data/processed/finite_size_scaling.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from percolation.simulation import percolation_probability
from percolation.bridge import bridge_percolation_probability
from percolation.cracks import (
    crack_segments_from_parallel,
    generate_voronoi_cracks,
)


def scan_threshold(
    eta_range: tuple[float, float],
    n_points: int,
    domain_size: float,
    n_trials: int,
    rng: np.random.Generator,
    crack_segments=None,
    direction: str = "both",
) -> tuple[list[dict], float]:
    """Scan η and estimate threshold at P=0.5."""
    etas = np.linspace(eta_range[0], eta_range[1], n_points)
    results = []

    for eta in etas:
        if crack_segments is not None:
            p = bridge_percolation_probability(
                eta, crack_segments, domain_size, 1.0, n_trials, "x", rng
            )
        else:
            p = percolation_probability(
                eta, domain_size, 1.0, n_trials, direction, rng
            )
        results.append({"eta": round(float(eta), 4), "p": round(float(p), 4)})

    etas_arr = np.array([r["eta"] for r in results])
    probs_arr = np.array([r["p"] for r in results])

    if probs_arr.min() < 0.5 < probs_arr.max():
        eta_c = float(np.interp(0.5, probs_arr, etas_arr))
    else:
        eta_c = float("nan")

    return results, round(eta_c, 4)


def fss_model(L, eta_c_inf, a, nu_inv):
    """Finite-size scaling: η_c(L) = η_c(∞) + a·L^{-1/ν}."""
    return eta_c_inf + a * L ** (-nu_inv)


def main():
    parser = argparse.ArgumentParser(description="Finite-size scaling")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--n-points", type=int, default=20,
                        help="η points per scan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/finite_size_scaling.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    domain_sizes = [8, 10, 15, 20, 30]
    n_parallel_cracks = 3  # fixed crack count, vary domain
    n_voronoi_seeds = 20

    # Load existing
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    all_results = existing.get("results", {})
    t_start = time.time()

    # === Standard percolation (no cracks) — baseline ===
    print("=== Standard percolation (no cracks) ===")
    if "standard" not in all_results:
        all_results["standard"] = {}

    for L in domain_sizes:
        key = str(L)
        if key in all_results["standard"]:
            eta_c = all_results["standard"][key]["eta_c"]
            print(f"  L={L:3d}: η_c = {eta_c:.4f}  [cached]")
            continue

        scan, eta_c = scan_threshold(
            (4.0, 8.0), args.n_points, float(L), args.trials, rng
        )
        all_results["standard"][key] = {
            "domain_size": L,
            "eta_c": eta_c,
            "scan": scan,
        }
        elapsed = time.time() - t_start
        print(f"  L={L:3d}: η_c = {eta_c:.4f}  ({elapsed:.0f}s)")
        _save(output_path, all_results, args)

    # === Parallel cracks ===
    print("\n=== Bridge percolation: parallel cracks ===")
    if "parallel" not in all_results:
        all_results["parallel"] = {}

    for L in domain_sizes:
        key = str(L)
        if key in all_results["parallel"]:
            eta_c = all_results["parallel"][key]["eta_c"]
            print(f"  L={L:3d}: η_c = {eta_c:.4f}  [cached]")
            continue

        cracks = crack_segments_from_parallel(
            n_parallel_cracks, domain_size=float(L), orientation="vertical"
        )
        scan, eta_c = scan_threshold(
            (1.0, 8.0), args.n_points, float(L), args.trials, rng,
            crack_segments=cracks
        )
        all_results["parallel"][key] = {
            "domain_size": L,
            "n_cracks": n_parallel_cracks,
            "eta_c": eta_c,
            "scan": scan,
        }
        elapsed = time.time() - t_start
        print(f"  L={L:3d}: η_c = {eta_c:.4f}  ({elapsed:.0f}s)")
        _save(output_path, all_results, args)

    # === Voronoi cracks ===
    print("\n=== Bridge percolation: Voronoi cracks ===")
    if "voronoi" not in all_results:
        all_results["voronoi"] = {}

    for L in domain_sizes:
        key = str(L)
        if key in all_results["voronoi"]:
            eta_c = all_results["voronoi"][key]["eta_c"]
            print(f"  L={L:3d}: η_c = {eta_c:.4f}  [cached]")
            continue

        # Scale Voronoi seeds with domain area to keep crack density constant
        n_seeds_scaled = int(n_voronoi_seeds * (L / 15.0) ** 2)
        n_seeds_scaled = max(n_seeds_scaled, 5)

        crack_rng = np.random.default_rng(args.seed + L)
        cracks = generate_voronoi_cracks(n_seeds_scaled, float(L), rng=crack_rng)

        scan, eta_c = scan_threshold(
            (3.0, 10.0), args.n_points, float(L), args.trials, rng,
            crack_segments=cracks
        )
        all_results["voronoi"][key] = {
            "domain_size": L,
            "n_voronoi_seeds": n_seeds_scaled,
            "n_segments": len(cracks),
            "eta_c": eta_c,
            "scan": scan,
        }
        elapsed = time.time() - t_start
        print(f"  L={L:3d} (seeds={n_seeds_scaled}): η_c = {eta_c:.4f}  ({elapsed:.0f}s)")
        _save(output_path, all_results, args)

    # === Fit FSS model ===
    print("\n=== Finite-size scaling fits ===")
    fits = {}
    for label in ["standard", "parallel", "voronoi"]:
        data = all_results[label]
        Ls = []
        eta_cs = []
        for key in sorted(data.keys(), key=int):
            L = data[key]["domain_size"]
            ec = data[key]["eta_c"]
            if not np.isnan(ec):
                Ls.append(L)
                eta_cs.append(ec)

        if len(Ls) >= 3:
            Ls = np.array(Ls, dtype=float)
            eta_cs = np.array(eta_cs)
            try:
                popt, pcov = curve_fit(
                    fss_model, Ls, eta_cs,
                    p0=[5.5, 1.0, 0.75],
                    bounds=([0, -np.inf, 0.1], [20, np.inf, 3.0]),
                    maxfev=5000,
                )
                perr = np.sqrt(np.diag(pcov))
                fits[label] = {
                    "eta_c_inf": round(float(popt[0]), 4),
                    "a": round(float(popt[1]), 4),
                    "nu_inv": round(float(popt[2]), 4),
                    "nu": round(1.0 / popt[2], 3) if popt[2] > 0 else None,
                    "eta_c_inf_err": round(float(perr[0]), 4),
                }
                print(f"  {label}: η_c(∞) = {popt[0]:.3f} ± {perr[0]:.3f}, "
                      f"ν = {1/popt[2]:.2f}")
            except RuntimeError:
                print(f"  {label}: fit failed")
        else:
            print(f"  {label}: not enough data points")

    all_results["fss_fits"] = fits
    _save(output_path, all_results, args)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Saved to {output_path}")


def _save(path, results, args):
    data = {
        "description": "Finite-size scaling for percolation thresholds",
        "parameters": {
            "n_trials": args.trials,
            "n_points_per_scan": args.n_points,
            "seed": args.seed,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
