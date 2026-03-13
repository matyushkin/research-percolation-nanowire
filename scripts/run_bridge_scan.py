#!/usr/bin/env python3
"""Parametric scan: bridge percolation threshold vs crack density.

Varies the number of cracks (parallel) or Voronoi seeds and measures
η_c(n_cracks) for both topologies. This is the main result for Paper #1.

Usage:
    uv run python scripts/run_bridge_scan.py [--trials 100] [--domain 15]
    uv run python scripts/run_bridge_scan.py --resume

Results saved to: data/processed/bridge_scan.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.bridge import bridge_percolation_probability
from percolation.cracks import (
    crack_segments_from_parallel,
    generate_voronoi_cracks,
)


def find_threshold_bisection(
    crack_segments,
    domain_size: float,
    stick_length: float,
    n_trials: int,
    direction: str,
    rng: np.random.Generator,
    eta_low: float = 1.0,
    eta_high: float = 15.0,
    tol: float = 0.3,
    max_iter: int = 12,
) -> tuple[float, list[dict]]:
    """Find η_c by bisection on P(η) = 0.5.

    Returns estimated η_c and scan data points.
    """
    scan_data = []

    for _ in range(max_iter):
        eta_mid = (eta_low + eta_high) / 2
        p = bridge_percolation_probability(
            eta_mid, crack_segments, domain_size, stick_length,
            n_trials, direction, rng
        )
        scan_data.append({"eta": round(eta_mid, 4), "p": round(p, 4)})

        if p < 0.5:
            eta_low = eta_mid
        else:
            eta_high = eta_mid

        if eta_high - eta_low < tol:
            break

    return round((eta_low + eta_high) / 2, 4), scan_data


def main():
    parser = argparse.ArgumentParser(description="Bridge percolation parameter scan")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/bridge_scan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Parameter ranges
    n_cracks_list = [1, 2, 3, 5, 7, 10, 15]
    n_voronoi_list = [5, 10, 20, 30, 50, 80]

    # Load existing
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Resumed from {output_path}")

    results = existing.get("results", {})
    t_start = time.time()

    # --- Parallel cracks ---
    print("=== Parallel cracks ===")
    if "parallel" not in results:
        results["parallel"] = {}

    for nc in n_cracks_list:
        key = str(nc)
        if key in results["parallel"]:
            eta_c = results["parallel"][key]["eta_c"]
            print(f"  n_cracks={nc:3d}: η_c = {eta_c:.3f}  [cached]")
            continue

        cracks = crack_segments_from_parallel(nc, args.domain, orientation="vertical")
        eta_c, scan = find_threshold_bisection(
            cracks, args.domain, 1.0, args.trials, "x", rng
        )
        results["parallel"][key] = {
            "n_cracks": nc,
            "n_segments": len(cracks),
            "eta_c": eta_c,
            "scan": scan,
        }
        elapsed = time.time() - t_start
        print(f"  n_cracks={nc:3d}: η_c = {eta_c:.3f}  ({elapsed:.0f}s)")
        _save(output_path, results, args)

    # --- Voronoi cracks ---
    print("\n=== Voronoi cracks ===")
    if "voronoi" not in results:
        results["voronoi"] = {}

    for nv in n_voronoi_list:
        key = str(nv)
        if key in results["voronoi"]:
            eta_c = results["voronoi"][key]["eta_c"]
            print(f"  n_seeds={nv:3d}: η_c = {eta_c:.3f}  [cached]")
            continue

        # Average over multiple Voronoi realizations
        eta_c_samples = []
        n_realizations = 5
        for r in range(n_realizations):
            crack_rng = np.random.default_rng(args.seed + 1000 * nv + r)
            cracks = generate_voronoi_cracks(nv, args.domain, rng=crack_rng)
            eta_c_r, _ = find_threshold_bisection(
                cracks, args.domain, 1.0, args.trials, "x", rng
            )
            eta_c_samples.append(eta_c_r)

        eta_c_mean = round(float(np.mean(eta_c_samples)), 4)
        eta_c_std = round(float(np.std(eta_c_samples)), 4)

        results["voronoi"][key] = {
            "n_voronoi_seeds": nv,
            "n_realizations": n_realizations,
            "eta_c_mean": eta_c_mean,
            "eta_c_std": eta_c_std,
            "eta_c_samples": eta_c_samples,
        }
        elapsed = time.time() - t_start
        print(f"  n_seeds={nv:3d}: η_c = {eta_c_mean:.3f} ± {eta_c_std:.3f}  ({elapsed:.0f}s)")
        _save(output_path, results, args)

    _save(output_path, results, args)
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Saved to {output_path}")


def _save(path, results, args):
    data = {
        "description": "Bridge percolation threshold vs crack density",
        "parameters": {
            "n_trials": args.trials,
            "domain_size": args.domain,
            "seed": args.seed,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
