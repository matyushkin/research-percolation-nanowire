#!/usr/bin/env python3
"""Fill in the n=20..30 transition region for parallel cracks (Fig. 3).

The HQ scan shows a jump from η_c≈2.0 (n=20) to η_c≈5.6 (n=30).
This script adds points at n=21,22,23,24,25,26,27,28,29 to resolve the transition.

Usage:
    uv run python scripts/run_bridge_scan_transition.py [--resume]

Results merged into: data/processed/bridge_scan_hq.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.bridge import bridge_percolation_probability
from percolation.cracks import crack_segments_from_parallel


def bisect_threshold(
    crack_segments, domain_size, n_trials, rng,
    eta_low=0.5, eta_high=12.0, tol=0.15, max_iter=16,
):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/bridge_scan_hq.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed + 5000)

    # Transition region for parallel cracks
    n_cracks_transition = [21, 22, 23, 24, 25, 26, 27, 28, 29]

    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    results = existing.get("results", {})
    if "parallel" not in results:
        results["parallel"] = {}

    t_start = time.time()

    print("=== Parallel cracks: transition region n=21..29 ===")
    for nc in n_cracks_transition:
        key = str(nc)
        if key in results["parallel"]:
            print(f"  n_cracks={nc}: η_c = {results['parallel'][key]['eta_c']:.3f}  [cached]")
            continue

        cracks = crack_segments_from_parallel(nc, args.domain, orientation="vertical")
        eta_c = bisect_threshold(cracks, args.domain, args.trials, rng)
        results["parallel"][key] = {"n_cracks": nc, "eta_c": eta_c}

        elapsed = time.time() - t_start
        print(f"  n_cracks={nc}: η_c = {eta_c:.3f}  ({elapsed:.0f}s)")
        _save(output_path, results, existing.get("parameters", {
            "n_trials": args.trials, "domain_size": args.domain
        }))

    _save(output_path, results, {"n_trials": args.trials, "domain_size": args.domain})
    print(f"\nTotal: {time.time() - t_start:.0f}s → {output_path}")


def _save(path, results, params):
    data = {
        "description": "HQ bridge scan for Fig. 3",
        "parameters": params,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
