#!/usr/bin/env python3
"""Bridge percolation: parallel cracks vs Voronoi crack networks.

Compares bridge percolation thresholds for:
1. Parallel straight cracks (Baret et al. 2024 model)
2. Voronoi random crack networks (NEW — this work)

For each crack topology, scans nanowire density η and estimates P(percolation).

Usage:
    uv run python scripts/run_bridge_percolation.py [--trials 100] [--domain 15]
    uv run python scripts/run_bridge_percolation.py --resume

Results saved to: data/processed/bridge_percolation.json
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


def main():
    parser = argparse.ArgumentParser(description="Bridge percolation comparison")
    parser.add_argument("--eta-min", type=float, default=3.0)
    parser.add_argument("--eta-max", type=float, default=12.0)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100,
                        help="MC trials per η value")
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--n-cracks", type=int, default=5,
                        help="Number of parallel cracks / Voronoi seeds")
    parser.add_argument("--n-voronoi-seeds", type=int, default=30,
                        help="Voronoi seed count (more = denser crack network)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/bridge_percolation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    etas = np.linspace(args.eta_min, args.eta_max, args.n_points)

    # Generate crack geometries
    parallel_cracks = crack_segments_from_parallel(
        args.n_cracks, domain_size=args.domain, orientation="vertical"
    )
    voronoi_cracks = generate_voronoi_cracks(
        args.n_voronoi_seeds, domain_size=args.domain, rng=rng
    )

    print(f"Parallel cracks: {len(parallel_cracks)} segments")
    print(f"Voronoi cracks: {len(voronoi_cracks)} segments")
    print(f"η range: [{args.eta_min}, {args.eta_max}], {args.n_points} points")
    print(f"Domain: {args.domain}×{args.domain}, {args.trials} trials each")
    print("=" * 60)

    # Load existing results
    results = {"parallel": {}, "voronoi": {}}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        for entry in data.get("parallel", []):
            results["parallel"][f"{entry['eta']:.6f}"] = entry
        for entry in data.get("voronoi", []):
            results["voronoi"][f"{entry['eta']:.6f}"] = entry
        print(f"Resumed: {len(results['parallel'])} parallel, "
              f"{len(results['voronoi'])} voronoi points")

    t_start = time.time()

    # --- Parallel cracks ---
    print("\n--- Parallel cracks (Baret model) ---")
    for i, eta in enumerate(etas):
        key = f"{eta:.6f}"
        if key in results["parallel"]:
            p = results["parallel"][key]["p_percolation"]
            print(f"  η = {eta:.2f}  P = {p:.3f}  [cached]")
            continue

        p = bridge_percolation_probability(
            eta, parallel_cracks, args.domain, 1.0, args.trials, "x", rng
        )
        results["parallel"][key] = {
            "eta": round(float(eta), 6),
            "p_percolation": float(p),
        }
        elapsed = time.time() - t_start
        print(f"  η = {eta:.2f}  P = {p:.3f}  ({i+1}/{args.n_points}, {elapsed:.0f}s)")
        _save(output_path, results, args, parallel_cracks, voronoi_cracks)

    # --- Voronoi cracks ---
    print("\n--- Voronoi cracks (this work) ---")
    for i, eta in enumerate(etas):
        key = f"{eta:.6f}"
        if key in results["voronoi"]:
            p = results["voronoi"][key]["p_percolation"]
            print(f"  η = {eta:.2f}  P = {p:.3f}  [cached]")
            continue

        p = bridge_percolation_probability(
            eta, voronoi_cracks, args.domain, 1.0, args.trials, "x", rng
        )
        results["voronoi"][key] = {
            "eta": round(float(eta), 6),
            "p_percolation": float(p),
        }
        elapsed = time.time() - t_start
        print(f"  η = {eta:.2f}  P = {p:.3f}  ({i+1}/{args.n_points}, {elapsed:.0f}s)")
        _save(output_path, results, args, parallel_cracks, voronoi_cracks)

    # Estimate thresholds
    for label in ["parallel", "voronoi"]:
        sorted_r = sorted(results[label].values(), key=lambda r: r["eta"])
        etas_arr = np.array([r["eta"] for r in sorted_r])
        probs_arr = np.array([r["p_percolation"] for r in sorted_r])
        eta_c = float(np.interp(0.5, probs_arr, etas_arr))
        print(f"\n{label}: η_c ≈ {eta_c:.3f}")

    _save(output_path, results, args, parallel_cracks, voronoi_cracks)
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Saved to {output_path}")


def _save(path, results, args, parallel_cracks, voronoi_cracks):
    def _sorted_list(d):
        return sorted(d.values(), key=lambda r: r["eta"])

    data = {
        "description": "Bridge percolation: parallel vs Voronoi cracks",
        "parameters": {
            "eta_range": [args.eta_min, args.eta_max],
            "n_points": args.n_points,
            "n_trials": args.trials,
            "domain_size": args.domain,
            "n_parallel_cracks": args.n_cracks,
            "n_voronoi_seeds": args.n_voronoi_seeds,
            "n_voronoi_segments": len(voronoi_cracks),
            "seed": args.seed,
        },
        "parallel": _sorted_list(results["parallel"]),
        "voronoi": _sorted_list(results["voronoi"]),
    }

    # Estimate thresholds
    for label in ["parallel", "voronoi"]:
        vals = data[label]
        if len(vals) >= 3:
            etas_arr = np.array([r["eta"] for r in vals])
            probs_arr = np.array([r["p_percolation"] for r in vals])
            if probs_arr.min() < 0.5 < probs_arr.max():
                data[f"eta_c_{label}"] = round(
                    float(np.interp(0.5, probs_arr, etas_arr)), 4
                )

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
