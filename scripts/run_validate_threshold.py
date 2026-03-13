#!/usr/bin/env python3
"""Validate stick percolation threshold η_c ≈ 5.63.

Scans dimensionless density η and estimates P(percolation) via Monte Carlo.
The known value for thin monodisperse sticks is η_c ≈ 5.63726
(Li & Zhang, PRE 80, 040104, 2009).

Usage:
    uv run python scripts/run_validate_threshold.py [--trials 200] [--domain 20]

Results saved to: data/processed/threshold_validation.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.simulation import percolation_probability


def main():
    parser = argparse.ArgumentParser(description="Validate percolation threshold")
    parser.add_argument("--eta-min", type=float, default=4.0)
    parser.add_argument("--eta-max", type=float, default=8.0)
    parser.add_argument("--n-points", type=int, default=25)
    parser.add_argument("--trials", type=int, default=200,
                        help="MC trials per η value")
    parser.add_argument("--domain", type=float, default=15.0,
                        help="Domain size in units of stick length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file")
    args = parser.parse_args()

    output_path = Path("data/processed/threshold_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    etas = np.linspace(args.eta_min, args.eta_max, args.n_points)

    # Load existing results if resuming
    results = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        for entry in data["results"]:
            results[f"{entry['eta']:.6f}"] = entry
        print(f"Loaded {len(results)} existing data points")

    print(f"Validating η_c: η ∈ [{args.eta_min}, {args.eta_max}], "
          f"{args.n_points} points, {args.trials} trials each")
    print(f"Domain: {args.domain}×{args.domain}, stick length: 1.0")
    print(f"Expected η_c ≈ 5.637")
    print("-" * 60)

    t_start = time.time()

    for i, eta in enumerate(etas):
        key = f"{eta:.6f}"
        if key in results:
            p = results[key]["p_percolation"]
            print(f"  η = {eta:.3f}  P = {p:.3f}  [cached]")
            continue

        p = percolation_probability(
            eta,
            domain_size=args.domain,
            stick_length=1.0,
            n_trials=args.trials,
            direction="both",
            rng=rng,
        )

        results[key] = {
            "eta": round(float(eta), 6),
            "p_percolation": float(p),
            "n_trials": args.trials,
            "domain_size": args.domain,
        }

        elapsed = time.time() - t_start
        print(f"  η = {eta:.3f}  P = {p:.3f}  "
              f"({i + 1}/{args.n_points}, {elapsed:.1f}s)")

        # Save incrementally
        _save(output_path, results, args)

    # Estimate threshold via interpolation
    sorted_results = sorted(results.values(), key=lambda r: r["eta"])
    etas_arr = np.array([r["eta"] for r in sorted_results])
    probs_arr = np.array([r["p_percolation"] for r in sorted_results])
    eta_c = float(np.interp(0.5, probs_arr, etas_arr))

    _save(output_path, results, args, eta_c=eta_c)

    elapsed = time.time() - t_start
    print("-" * 60)
    print(f"Estimated η_c = {eta_c:.3f}  (reference: 5.637)")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Saved to {output_path}")


def _save(path, results, args, eta_c=None):
    sorted_results = sorted(results.values(), key=lambda r: r["eta"])
    data = {
        "description": "Percolation threshold validation for monodisperse sticks",
        "parameters": {
            "eta_range": [args.eta_min, args.eta_max],
            "n_points": args.n_points,
            "n_trials": args.trials,
            "domain_size": args.domain,
            "stick_length": 1.0,
            "seed": args.seed,
        },
        "results": sorted_results,
    }
    if eta_c is not None:
        data["eta_c_estimated"] = eta_c
        data["eta_c_reference"] = 5.63726
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
