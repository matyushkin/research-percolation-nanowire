#!/usr/bin/env python3
"""HQ finite-size scaling for Fig. 4.

More domain sizes, more trials, includes Voronoi.
Domain sizes: L/l = 5, 8, 10, 12, 15, 20, 25, 30, 40.

Usage:
    uv run python scripts/run_fss_hq.py [--trials 200] [--resume]

Results saved to: data/processed/finite_size_scaling_hq.json
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
    crack_fn, domain_size, n_trials, rng,
    eta_low=0.5, eta_high=12.0, tol=0.15, max_iter=16,
):
    """Find eta_c by bisection. crack_fn(domain_size) -> crack_segments or []."""
    cracks = crack_fn(domain_size)
    for _ in range(max_iter):
        eta_mid = (eta_low + eta_high) / 2
        p = bridge_percolation_probability(
            eta_mid, cracks, domain_size, 1.0, n_trials, "x", rng
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/finite_size_scaling_hq.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    domain_sizes = [5, 8, 10, 12, 15, 20, 25, 30, 40]

    # Crack density scales with domain: keep ~3 cracks per domain for parallel,
    # ~15 seeds per domain for Voronoi (relative density constant)
    def make_parallel(ds):
        n_c = max(1, int(3 * ds / 15))
        return crack_segments_from_parallel(n_c, ds, orientation="vertical")

    def make_voronoi(ds):
        n_s = max(3, int(15 * (ds / 15)**2))
        vrng = np.random.default_rng(args.seed + 3000 + int(ds * 100))
        return generate_voronoi_cracks(n_s, ds, rng=vrng)

    configs = {
        "standard": lambda ds: [],
        "parallel": make_parallel,
        "voronoi": make_voronoi,
    }

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    results = existing.get("results", {})

    t_start = time.time()

    for label, crack_fn in configs.items():
        print(f"\n=== {label} ===")
        if label not in results:
            results[label] = {}

        for ds in domain_sizes:
            key = str(ds)
            if key in results[label]:
                print(f"  L={ds:3d}: η_c = {results[label][key]['eta_c']:.3f}  [cached]")
                continue

            eta_c = bisect_threshold(crack_fn, float(ds), args.trials, rng)
            results[label][key] = {"domain_size": ds, "eta_c": eta_c}

            elapsed = time.time() - t_start
            print(f"  L={ds:3d}: η_c = {eta_c:.3f}  ({elapsed:.0f}s)")
            _save(output_path, results, args)

    _save(output_path, results, args)
    print(f"\nTotal: {time.time() - t_start:.0f}s → {output_path}")


def _save(path, results, args):
    data = {
        "description": "HQ finite-size scaling for Fig. 4",
        "parameters": {"n_trials": args.trials},
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
