#!/usr/bin/env python3
"""HQ bridge scan: η_c vs crack density for Fig. 3.

More crack counts, more Voronoi realizations, higher trials.
Parallel cracks extended to n_c=80 to match Voronoi x-axis.

Usage:
    uv run python scripts/run_bridge_scan_hq.py [--trials 150]
    uv run python scripts/run_bridge_scan_hq.py --resume

Results saved to: data/processed/bridge_scan_hq.json
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


def bisect_threshold(
    crack_segments, domain_size, n_trials, rng,
    eta_low=0.5, eta_high=12.0, tol=0.2, max_iter=14,
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
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--n-voronoi-real", type=int, default=10,
                        help="Voronoi realizations per seed count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/bridge_scan_hq.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Wider range, finer step
    n_cracks_list = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80]
    n_voronoi_list = [3, 5, 7, 10, 13, 15, 20, 30, 50, 80]

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    results = existing.get("results", {})
    t_start = time.time()

    # --- Parallel ---
    print("=== Parallel cracks ===")
    if "parallel" not in results:
        results["parallel"] = {}

    for nc in n_cracks_list:
        key = str(nc)
        if key in results["parallel"]:
            print(f"  n_cracks={nc:3d}: η_c = {results['parallel'][key]['eta_c']:.3f}  [cached]")
            continue

        cracks = crack_segments_from_parallel(nc, args.domain, orientation="vertical")
        eta_c = bisect_threshold(cracks, args.domain, args.trials, rng)
        results["parallel"][key] = {"n_cracks": nc, "eta_c": eta_c}

        elapsed = time.time() - t_start
        print(f"  n_cracks={nc:3d}: η_c = {eta_c:.3f}  ({elapsed:.0f}s)")
        _save(output_path, results, args)

    # --- Voronoi ---
    print("\n=== Voronoi cracks ===")
    if "voronoi" not in results:
        results["voronoi"] = {}

    for nv in n_voronoi_list:
        key = str(nv)
        existing_samples = results.get("voronoi", {}).get(key, {}).get("eta_c_samples", [])
        if len(existing_samples) >= args.n_voronoi_real:
            m = results["voronoi"][key]["eta_c_mean"]
            s = results["voronoi"][key]["eta_c_std"]
            print(f"  n_seeds={nv:3d}: η_c = {m:.3f} ± {s:.3f}  [cached]")
            continue

        samples = list(existing_samples)
        for r in range(len(samples), args.n_voronoi_real):
            crack_rng = np.random.default_rng(args.seed + 10000 * nv + r)
            mc_rng = np.random.default_rng(args.seed + 20000 * nv + r)
            cracks = generate_voronoi_cracks(nv, args.domain, rng=crack_rng)
            eta_c = bisect_threshold(cracks, args.domain, args.trials, mc_rng)
            samples.append(eta_c)

        m = round(float(np.mean(samples)), 4)
        s = round(float(np.std(samples)), 4)
        results["voronoi"][key] = {
            "n_voronoi_seeds": nv,
            "n_realizations": len(samples),
            "eta_c_mean": m,
            "eta_c_std": s,
            "eta_c_samples": samples,
        }
        elapsed = time.time() - t_start
        print(f"  n_seeds={nv:3d}: η_c = {m:.3f} ± {s:.3f}  ({elapsed:.0f}s)")
        _save(output_path, results, args)

    _save(output_path, results, args)
    print(f"\nTotal: {time.time() - t_start:.0f}s → {output_path}")


def _save(path, results, args):
    data = {
        "description": "HQ bridge scan for Fig. 3",
        "parameters": {"n_trials": args.trials, "domain_size": args.domain},
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
