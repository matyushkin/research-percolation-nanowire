#!/usr/bin/env python3
"""High-quality P(η) curves for parallel vs Voronoi (Fig. 2).

Dense sampling near the transition, 300 trials per point.

Usage:
    uv run python scripts/run_bridge_percolation_hq.py
    uv run python scripts/run_bridge_percolation_hq.py --resume

Results saved to: data/processed/bridge_percolation_hq.json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/bridge_percolation_hq.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Dense sampling: coarse over full range + fine near transitions
    # Parallel: transition near η ≈ 1–3
    etas_parallel = np.sort(np.unique(np.concatenate([
        np.linspace(0.5, 1.0, 5),
        np.linspace(1.0, 3.5, 20),
        np.linspace(3.5, 6.0, 8),
    ])))
    # Voronoi: transition near η ≈ 4–7
    etas_voronoi = np.sort(np.unique(np.concatenate([
        np.linspace(2.0, 4.0, 6),
        np.linspace(4.0, 7.0, 25),
        np.linspace(7.0, 9.0, 6),
    ])))

    n_parallel_cracks = 3
    n_voronoi_seeds = 20

    parallel_cracks = crack_segments_from_parallel(
        n_parallel_cracks, args.domain, orientation="vertical"
    )
    voronoi_rng = np.random.default_rng(args.seed + 999)
    voronoi_cracks = generate_voronoi_cracks(
        n_voronoi_seeds, args.domain, rng=voronoi_rng
    )

    # Load existing
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    # _save writes lists; convert back to dicts keyed by eta for resume
    def _to_dict(data):
        if isinstance(data, list):
            return {f"{r['eta']:.4f}": r for r in data}
        return data

    results = {
        "parallel": _to_dict(existing.get("parallel", {})),
        "voronoi": _to_dict(existing.get("voronoi", {})),
    }

    t_start = time.time()

    for label, cracks, etas in [
        ("parallel", parallel_cracks, etas_parallel),
        ("voronoi", voronoi_cracks, etas_voronoi),
    ]:
        print(f"\n=== {label} ({len(etas)} points, {args.trials} trials) ===")
        for i, eta in enumerate(etas):
            key = f"{eta:.4f}"
            if key in results[label]:
                p = results[label][key]["p"]
                print(f"  η={eta:.3f}  P={p:.3f}  [cached]")
                continue

            p = bridge_percolation_probability(
                eta, cracks, args.domain, 1.0, args.trials, "x", rng
            )
            results[label][key] = {"eta": round(float(eta), 4), "p": round(float(p), 4)}

            elapsed = time.time() - t_start
            print(f"  η={eta:.3f}  P={p:.3f}  ({i+1}/{len(etas)}, {elapsed:.0f}s)")
            _save(output_path, results, args)

    _save(output_path, results, args)
    print(f"\nTotal: {time.time() - t_start:.0f}s → {output_path}")


def _save(path, results, args):
    data = {
        "description": "HQ P(η) curves for Fig. 2",
        "parameters": {
            "n_trials": args.trials,
            "domain_size": args.domain,
            "n_parallel_cracks": 3,
            "n_voronoi_seeds": 20,
        },
    }
    for label in ["parallel", "voronoi"]:
        data[label] = sorted(results[label].values(), key=lambda r: r["eta"])
        pts = data[label]
        if len(pts) >= 3:
            etas = np.array([r["eta"] for r in pts])
            probs = np.array([r["p"] for r in pts])
            if probs.min() < 0.5 < probs.max():
                data[f"eta_c_{label}"] = round(float(np.interp(0.5, probs, etas)), 4)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
