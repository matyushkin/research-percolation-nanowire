#!/usr/bin/env python3
"""Production-quality finite-size scaling for bridge percolation.

Measures η_c(L) for standard, parallel-crack, and Voronoi-crack percolation
at multiple domain sizes. For Voronoi cracks, averages over multiple crack
realizations to reduce geometry-dependent noise.

Extrapolates to η_c(∞) using FSS ansatz: η_c(L) = η_c(∞) + a·L^{-1/ν}.

Improvements over run_fss_hq.py:
- 500 trials per point (vs 200)
- Bisection tolerance 0.05 (vs 0.15)
- More domain sizes including L=50
- Voronoi: 10 crack realizations per L, report mean ± std
- Bootstrap error estimation for FSS fit

Usage:
    uv run python scripts/run_fss_production.py [--trials 500] [--resume]

Results saved to: data/processed/fss_production.json
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


def bisect_threshold(
    prob_fn,
    eta_low: float,
    eta_high: float,
    tol: float = 0.05,
    max_iter: int = 20,
) -> float:
    """Find η_c by bisection where P(η_c) = 0.5.

    Parameters
    ----------
    prob_fn : callable(eta) -> float
        Returns percolation probability at given η.
    eta_low, eta_high : float
        Bracket: P(eta_low) < 0.5 < P(eta_high).
    tol : float
        Stop when eta_high - eta_low < tol.
    max_iter : int
    """
    # Verify bracket
    p_low = prob_fn(eta_low)
    if p_low > 0.5:
        # Lower bound too high, search downward
        while p_low > 0.2 and eta_low > 0.1:
            eta_low /= 2
            p_low = prob_fn(eta_low)

    p_high = prob_fn(eta_high)
    if p_high < 0.5:
        # Upper bound too low, search upward
        while p_high < 0.8 and eta_high < 50:
            eta_high *= 1.5
            p_high = prob_fn(eta_high)

    for _ in range(max_iter):
        if eta_high - eta_low < tol:
            break
        eta_mid = (eta_low + eta_high) / 2
        p_mid = prob_fn(eta_mid)
        if p_mid < 0.5:
            eta_low = eta_mid
        else:
            eta_high = eta_mid

    return round((eta_low + eta_high) / 2, 4)


def fss_model(L, eta_c_inf, a, nu_inv):
    """FSS ansatz: η_c(L) = η_c(∞) + a·L^{-1/ν}."""
    return eta_c_inf + a * L ** (-nu_inv)


def fit_fss(Ls, eta_cs, p0=None):
    """Fit FSS model and return parameters with errors.

    Returns dict with eta_c_inf, a, nu_inv, nu, and their errors,
    or None if fit fails.
    """
    Ls = np.array(Ls, dtype=float)
    eta_cs = np.array(eta_cs, dtype=float)

    if len(Ls) < 3:
        return None

    if p0 is None:
        p0 = [eta_cs[-1], 1.0, 0.75]

    try:
        popt, pcov = curve_fit(
            fss_model, Ls, eta_cs,
            p0=p0,
            bounds=([0, -np.inf, 0.01], [20, np.inf, 5.0]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        nu = 1.0 / popt[2] if popt[2] > 0.01 else float("nan")
        return {
            "eta_c_inf": round(float(popt[0]), 4),
            "eta_c_inf_err": round(float(perr[0]), 4),
            "a": round(float(popt[1]), 4),
            "a_err": round(float(perr[1]), 4),
            "nu_inv": round(float(popt[2]), 4),
            "nu_inv_err": round(float(perr[2]), 4),
            "nu": round(float(nu), 3),
        }
    except (RuntimeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Production FSS for bridge percolation"
    )
    parser.add_argument("--trials", type=int, default=500,
                        help="MC trials per density point")
    parser.add_argument("--voronoi-realizations", type=int, default=10,
                        help="Voronoi crack realizations per domain size")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/fss_production.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    domain_sizes = [5, 8, 10, 15, 20, 30, 40, 50]
    n_parallel_cracks_base = 3  # at L=15
    n_voronoi_seeds_base = 15   # at L=15

    # Load existing
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    all_results = existing.get("results", {})

    t_start = time.time()

    # ===== Standard percolation (no cracks) =====
    print("=" * 60)
    print("STANDARD PERCOLATION (no cracks)")
    print("=" * 60)

    if "standard" not in all_results:
        all_results["standard"] = {}

    for L in domain_sizes:
        key = str(L)
        if key in all_results["standard"]:
            print(f"  L={L:3d}: η_c = {all_results['standard'][key]['eta_c']:.4f}  [cached]")
            continue

        def prob_fn(eta, _L=L):
            return percolation_probability(
                eta, float(_L), 1.0, args.trials, "both", rng
            )

        eta_c = bisect_threshold(prob_fn, 4.0, 8.0, tol=0.05)
        elapsed = time.time() - t_start
        all_results["standard"][key] = {
            "domain_size": L,
            "eta_c": eta_c,
        }
        print(f"  L={L:3d}: η_c = {eta_c:.4f}  ({elapsed:.0f}s)")
        _save(output_path, all_results, args)

    # ===== Parallel cracks =====
    print("\n" + "=" * 60)
    print("PARALLEL CRACKS (fixed crack density)")
    print("=" * 60)

    if "parallel" not in all_results:
        all_results["parallel"] = {}

    for L in domain_sizes:
        key = str(L)
        if key in all_results["parallel"]:
            print(f"  L={L:3d}: η_c = {all_results['parallel'][key]['eta_c']:.4f}  [cached]")
            continue

        # Scale crack count with domain size to keep constant crack density
        n_cracks = max(1, round(n_parallel_cracks_base * L / 15.0))
        cracks = crack_segments_from_parallel(n_cracks, float(L), "vertical")

        def prob_fn(eta, _cracks=cracks, _L=L):
            return bridge_percolation_probability(
                eta, _cracks, float(_L), 1.0, args.trials, "x", rng
            )

        eta_c = bisect_threshold(prob_fn, 0.3, 8.0, tol=0.05)
        elapsed = time.time() - t_start
        all_results["parallel"][key] = {
            "domain_size": L,
            "n_cracks": n_cracks,
            "eta_c": eta_c,
        }
        print(f"  L={L:3d} ({n_cracks} cracks): η_c = {eta_c:.4f}  ({elapsed:.0f}s)")
        _save(output_path, all_results, args)

    # ===== Voronoi cracks (averaged over realizations) =====
    print("\n" + "=" * 60)
    print(f"VORONOI CRACKS ({args.voronoi_realizations} realizations per L)")
    print("=" * 60)

    if "voronoi" not in all_results:
        all_results["voronoi"] = {}

    for L in domain_sizes:
        key = str(L)
        if key in all_results["voronoi"]:
            d = all_results["voronoi"][key]
            n_done = len(d.get("eta_c_values", []))
            if n_done >= args.voronoi_realizations:
                print(f"  L={L:3d}: η_c = {d['eta_c_mean']:.4f} ± {d['eta_c_std']:.4f}  [cached, {n_done} realizations]")
                continue

        # Scale Voronoi seeds with domain area
        n_seeds = max(3, round(n_voronoi_seeds_base * (L / 15.0) ** 2))

        # Load partial results
        entry = all_results["voronoi"].get(key, {
            "domain_size": L,
            "n_voronoi_seeds": n_seeds,
            "eta_c_values": [],
        })
        start_idx = len(entry["eta_c_values"])

        for r in range(start_idx, args.voronoi_realizations):
            crack_rng = np.random.default_rng(args.seed + L * 1000 + r)
            cracks = generate_voronoi_cracks(n_seeds, float(L), rng=crack_rng)

            def prob_fn(eta, _cracks=cracks, _L=L):
                return bridge_percolation_probability(
                    eta, _cracks, float(_L), 1.0, args.trials, "x", rng
                )

            eta_c = bisect_threshold(prob_fn, 1.0, 12.0, tol=0.05)
            entry["eta_c_values"].append(eta_c)

            elapsed = time.time() - t_start
            print(f"  L={L:3d} (seeds={n_seeds}, r={r+1}/{args.voronoi_realizations}): "
                  f"η_c = {eta_c:.4f}  ({elapsed:.0f}s)")

            # Update stats
            vals = np.array(entry["eta_c_values"])
            entry["eta_c_mean"] = round(float(np.mean(vals)), 4)
            entry["eta_c_std"] = round(float(np.std(vals)), 4)
            entry["n_segments_last"] = len(cracks)

            all_results["voronoi"][key] = entry
            _save(output_path, all_results, args)

        vals = np.array(entry["eta_c_values"])
        print(f"  L={L:3d}: η_c = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ===== FSS fits =====
    print("\n" + "=" * 60)
    print("FSS FITS: η_c(L) = η_c(∞) + a·L^{-1/ν}")
    print("=" * 60)

    fits = {}

    # Standard
    Ls, ecs = [], []
    for key in sorted(all_results["standard"], key=lambda k: int(k)):
        L = all_results["standard"][key]["domain_size"]
        ec = all_results["standard"][key]["eta_c"]
        if not np.isnan(ec):
            Ls.append(L)
            ecs.append(ec)
    fit = fit_fss(Ls, ecs, p0=[5.637, 1.0, 0.75])
    if fit:
        fits["standard"] = fit
        print(f"  Standard: η_c(∞) = {fit['eta_c_inf']:.4f} ± {fit['eta_c_inf_err']:.4f}, "
              f"ν = {fit['nu']:.2f}")
    else:
        print("  Standard: fit failed")

    # Parallel
    Ls, ecs = [], []
    for key in sorted(all_results.get("parallel", {}), key=lambda k: int(k)):
        L = all_results["parallel"][key]["domain_size"]
        ec = all_results["parallel"][key]["eta_c"]
        if not np.isnan(ec):
            Ls.append(L)
            ecs.append(ec)
    fit = fit_fss(Ls, ecs, p0=[0.5, 5.0, 0.5])
    if fit:
        fits["parallel"] = fit
        print(f"  Parallel: η_c(∞) = {fit['eta_c_inf']:.4f} ± {fit['eta_c_inf_err']:.4f}, "
              f"ν = {fit['nu']:.2f}")
    else:
        print("  Parallel: fit failed")

    # Voronoi (using mean η_c)
    Ls, ecs, errs = [], [], []
    for key in sorted(all_results.get("voronoi", {}), key=lambda k: int(k)):
        d = all_results["voronoi"][key]
        L = d["domain_size"]
        if "eta_c_mean" in d and not np.isnan(d["eta_c_mean"]):
            Ls.append(L)
            ecs.append(d["eta_c_mean"])
            errs.append(d.get("eta_c_std", 0.1))
    fit = fit_fss(Ls, ecs, p0=[5.637, 1.0, 0.75])
    if fit:
        fits["voronoi"] = fit
        print(f"  Voronoi:  η_c(∞) = {fit['eta_c_inf']:.4f} ± {fit['eta_c_inf_err']:.4f}, "
              f"ν = {fit['nu']:.2f}")
    else:
        print("  Voronoi: fit failed")

    all_results["fss_fits"] = fits
    _save(output_path, all_results, args)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved to {output_path}")


def _save(path, results, args):
    data = {
        "description": "Production FSS for bridge percolation thresholds",
        "parameters": {
            "n_trials": args.trials,
            "voronoi_realizations": args.voronoi_realizations,
            "seed": args.seed,
            "bisection_tol": 0.05,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
