#!/usr/bin/env python3
"""Voronoi ensemble: η_c averaged over many random crack realizations.

For each n_seeds value, generates N_real independent Voronoi crack networks
and measures η_c for each. Reports mean, std, and full distribution.

This is needed because a single Voronoi realization can be atypical.

Usage:
    uv run python scripts/run_voronoi_ensemble.py [--realizations 50]
    uv run python scripts/run_voronoi_ensemble.py --resume

Results saved to: data/processed/voronoi_ensemble.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.bridge import bridge_percolation_probability
from percolation.cracks import generate_voronoi_cracks


def bisect_threshold(
    crack_segments,
    domain_size: float,
    n_trials: int,
    rng: np.random.Generator,
    eta_low: float = 2.0,
    eta_high: float = 12.0,
    tol: float = 0.3,
    max_iter: int = 12,
) -> float:
    """Find η_c by bisection."""
    for _ in range(max_iter):
        eta_mid = (eta_low + eta_high) / 2
        p = bridge_percolation_probability(
            eta_mid, crack_segments, domain_size, 1.0, n_trials, "x", rng
        )
        if p < 0.5:
            eta_low = eta_mid
        else:
            eta_high = eta_mid
        if eta_high - eta_low < tol:
            break
    return round((eta_low + eta_high) / 2, 4)


def main():
    parser = argparse.ArgumentParser(description="Voronoi ensemble statistics")
    parser.add_argument("--realizations", type=int, default=50,
                        help="Voronoi realizations per n_seeds")
    parser.add_argument("--trials", type=int, default=80,
                        help="MC trials per η value in bisection")
    parser.add_argument("--domain", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/voronoi_ensemble.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_seeds_list = [5, 10, 15, 20, 30, 50]

    # Load existing
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    results = existing.get("results", {})
    t_start = time.time()

    for ns in n_seeds_list:
        key = str(ns)
        if key in results and len(results[key].get("eta_c_samples", [])) >= args.realizations:
            m = results[key]["eta_c_mean"]
            s = results[key]["eta_c_std"]
            print(f"n_seeds={ns:3d}: η_c = {m:.3f} ± {s:.3f}  [cached, {len(results[key]['eta_c_samples'])} realizations]")
            continue

        # Resume partial
        existing_samples = results.get(key, {}).get("eta_c_samples", [])
        start_idx = len(existing_samples)
        samples = list(existing_samples)

        print(f"\nn_seeds={ns}: computing {args.realizations - start_idx} realizations "
              f"(have {start_idx})...")

        for r in range(start_idx, args.realizations):
            crack_rng = np.random.default_rng(args.seed + 10000 * ns + r)
            mc_rng = np.random.default_rng(args.seed + 20000 * ns + r)

            cracks = generate_voronoi_cracks(ns, args.domain, rng=crack_rng)
            eta_c = bisect_threshold(cracks, args.domain, args.trials, mc_rng)
            samples.append(eta_c)

            if (r + 1) % 5 == 0 or r == args.realizations - 1:
                elapsed = time.time() - t_start
                m = float(np.mean(samples))
                s = float(np.std(samples))
                print(f"  [{r+1}/{args.realizations}] η_c = {m:.3f} ± {s:.3f}  ({elapsed:.0f}s)")

                results[key] = {
                    "n_voronoi_seeds": ns,
                    "n_realizations": len(samples),
                    "eta_c_mean": round(m, 4),
                    "eta_c_std": round(s, 4),
                    "eta_c_samples": samples,
                    "eta_c_min": round(float(np.min(samples)), 4),
                    "eta_c_max": round(float(np.max(samples)), 4),
                }
                _save(output_path, results, args)

    # Summary
    print("\n" + "=" * 60)
    print("Summary: η_c vs n_voronoi_seeds")
    print(f"{'n_seeds':>8}  {'η_c':>8}  {'±σ':>6}  {'min':>6}  {'max':>6}  {'N':>4}")
    print("-" * 50)
    for ns in n_seeds_list:
        key = str(ns)
        if key in results:
            r = results[key]
            print(f"{r['n_voronoi_seeds']:8d}  {r['eta_c_mean']:8.3f}  "
                  f"{r['eta_c_std']:6.3f}  {r['eta_c_min']:6.2f}  "
                  f"{r['eta_c_max']:6.2f}  {r['n_realizations']:4d}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Saved to {output_path}")


def _save(path, results, args):
    data = {
        "description": "Voronoi ensemble: η_c statistics over many crack realizations",
        "parameters": {
            "n_realizations_target": args.realizations,
            "n_trials_per_bisection": args.trials,
            "domain_size": args.domain,
            "seed": args.seed,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
