#!/usr/bin/env python3
"""Improved polydispersity scan for Fig. 5.

Finer PD grid, higher trials, tighter bisection tolerance.
Standard + Voronoi sparse + Voronoi dense.

Usage:
    uv run python scripts/run_polydisperse_hq2.py [--trials 250] [--resume]

Results saved to: data/processed/polydisperse_hq2.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.bridge import bridge_percolation
from percolation.cracks import generate_voronoi_cracks


def generate_polydisperse_sticks(n, mean_L, pd, domain, rng):
    if pd < 1e-6:
        lengths = np.full(n, mean_L)
    else:
        sigma2 = np.log(1 + pd**2)
        mu = np.log(mean_L) - sigma2 / 2
        lengths = rng.lognormal(mu, np.sqrt(sigma2), n)
    cx = rng.uniform(0, domain, n)
    cy = rng.uniform(0, domain, n)
    theta = rng.uniform(0, np.pi, n)
    half_l = lengths / 2
    sticks = np.empty((n, 2, 2))
    sticks[:, 0, 0] = cx - half_l * np.cos(theta)
    sticks[:, 0, 1] = cy - half_l * np.sin(theta)
    sticks[:, 1, 0] = cx + half_l * np.cos(theta)
    sticks[:, 1, 1] = cy + half_l * np.sin(theta)
    return sticks


def bridge_prob_polydisperse(eta, cracks, domain, mean_L, pd, n_trials, rng):
    n_sticks = int(eta * domain**2 / mean_L**2)
    count = 0
    for _ in range(n_trials):
        sticks = generate_polydisperse_sticks(n_sticks, mean_L, pd, domain, rng)
        if bridge_percolation(sticks, cracks, domain, "x"):
            count += 1
    return count / n_trials


def bisect(cracks, domain, mean_L, pd, n_trials, rng,
           eta_lo=1.0, eta_hi=10.0, tol=0.15, max_iter=16):
    for _ in range(max_iter):
        mid = (eta_lo + eta_hi) / 2
        p = bridge_prob_polydisperse(mid, cracks, domain, mean_L, pd, n_trials, rng)
        if p < 0.5:
            eta_lo = mid
        else:
            eta_hi = mid
        if eta_hi - eta_lo < tol:
            break
    return round((eta_lo + eta_hi) / 2, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/polydisperse_hq2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Fine PD grid — 15 points for smooth curves
    pds = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3,
           0.4, 0.5, 0.6, 0.7, 0.85, 1.0]

    configs = {
        "standard": [],
        "voronoi_sparse": generate_voronoi_cracks(
            8, args.domain, rng=np.random.default_rng(args.seed + 100)),
        "voronoi_dense": generate_voronoi_cracks(
            25, args.domain, rng=np.random.default_rng(args.seed + 200)),
    }

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    results = existing.get("results", {})
    t_start = time.time()

    for label, cracks in configs.items():
        print(f"\n=== {label} ({len(cracks)} crack segments) ===")
        if label not in results:
            results[label] = {}

        for pd in pds:
            key = f"{pd:.2f}"
            if key in results[label]:
                print(f"  PD={pd:.2f}: η_c={results[label][key]['eta_c']:.3f}  [cached]")
                continue

            eta_c = bisect(cracks, args.domain, 1.0, pd, args.trials, rng)
            results[label][key] = {"polydispersity": pd, "eta_c": eta_c}

            elapsed = time.time() - t_start
            print(f"  PD={pd:.2f}: η_c={eta_c:.3f}  ({elapsed:.0f}s)")
            _save(output_path, results, args)

    _save(output_path, results, args)
    print(f"\nTotal: {time.time() - t_start:.0f}s → {output_path}")


def _save(path, results, args):
    data = {
        "description": "HQ polydispersity for Fig. 5 (v2, fine grid)",
        "parameters": {"n_trials": args.trials, "domain_size": args.domain},
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
