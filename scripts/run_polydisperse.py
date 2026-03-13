#!/usr/bin/env python3
"""Bridge percolation with polydisperse nanowire lengths.

Real AgNW batches have a distribution of lengths, typically log-normal.
This script measures how polydispersity affects bridge percolation threshold
for both parallel and Voronoi cracks.

Polydispersity index: σ_L / <L> (0 = monodisperse, 0.3-0.5 = typical synthesis)

Usage:
    uv run python scripts/run_polydisperse.py [--trials 100]
    uv run python scripts/run_polydisperse.py --resume

Results saved to: data/processed/polydisperse.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.sticks import find_intersections
from percolation.clusters import UnionFind
from percolation.bridge import bridge_percolation
from percolation.cracks import (
    crack_segments_from_parallel,
    generate_voronoi_cracks,
)


def generate_polydisperse_sticks(
    n_sticks: int,
    mean_length: float,
    polydispersity: float,
    domain_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate sticks with log-normal length distribution.

    Parameters
    ----------
    polydispersity : float
        σ_L / <L>. 0 = monodisperse.
    """
    if polydispersity < 1e-6:
        lengths = np.full(n_sticks, mean_length)
    else:
        # Log-normal: if X ~ LogNormal(μ, σ²), then <X> = exp(μ + σ²/2)
        # and Var(X) = (exp(σ²) - 1) * exp(2μ + σ²)
        # So CV = σ_L/<L> = sqrt(exp(σ²) - 1)
        # → σ² = log(1 + CV²)
        sigma2 = np.log(1 + polydispersity ** 2)
        mu = np.log(mean_length) - sigma2 / 2
        lengths = rng.lognormal(mu, np.sqrt(sigma2), n_sticks)

    cx = rng.uniform(0, domain_size, n_sticks)
    cy = rng.uniform(0, domain_size, n_sticks)
    theta = rng.uniform(0, np.pi, n_sticks)

    half_l = lengths / 2
    dx = half_l * np.cos(theta)
    dy = half_l * np.sin(theta)

    sticks = np.empty((n_sticks, 2, 2))
    sticks[:, 0, 0] = cx - dx
    sticks[:, 0, 1] = cy - dy
    sticks[:, 1, 0] = cx + dx
    sticks[:, 1, 1] = cy + dy

    return sticks


def bridge_percolation_polydisperse(
    eta: float,
    crack_segments,
    domain_size: float,
    mean_length: float,
    polydispersity: float,
    n_trials: int,
    direction: str,
    rng: np.random.Generator,
) -> float:
    """P(bridge percolation) for polydisperse sticks."""
    n_sticks = int(eta * domain_size ** 2 / mean_length ** 2)
    n_perc = 0

    for _ in range(n_trials):
        sticks = generate_polydisperse_sticks(
            n_sticks, mean_length, polydispersity, domain_size, rng
        )
        if bridge_percolation(sticks, crack_segments, domain_size, direction):
            n_perc += 1

    return n_perc / n_trials


def bisect_threshold(
    crack_segments,
    domain_size: float,
    mean_length: float,
    polydispersity: float,
    n_trials: int,
    rng: np.random.Generator,
    eta_low: float = 1.0,
    eta_high: float = 12.0,
    tol: float = 0.3,
    max_iter: int = 12,
) -> float:
    for _ in range(max_iter):
        eta_mid = (eta_low + eta_high) / 2
        p = bridge_percolation_polydisperse(
            eta_mid, crack_segments, domain_size, mean_length,
            polydispersity, n_trials, "x", rng
        )
        if p < 0.5:
            eta_low = eta_mid
        else:
            eta_high = eta_mid
        if eta_high - eta_low < tol:
            break
    return round((eta_low + eta_high) / 2, 4)


def main():
    parser = argparse.ArgumentParser(description="Polydisperse bridge percolation")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--domain", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/polydisperse.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    polydispersities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_parallel = 3
    n_voronoi_seeds = 20

    # Generate fixed crack geometries
    parallel_cracks = crack_segments_from_parallel(
        n_parallel, args.domain, orientation="vertical"
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

    results = existing.get("results", {})
    t_start = time.time()

    for crack_label, cracks in [("parallel", parallel_cracks), ("voronoi", voronoi_cracks)]:
        print(f"\n=== {crack_label} cracks ===")
        if crack_label not in results:
            results[crack_label] = {}

        for pd in polydispersities:
            key = f"{pd:.2f}"
            if key in results[crack_label]:
                eta_c = results[crack_label][key]["eta_c"]
                print(f"  PD={pd:.2f}: η_c = {eta_c:.3f}  [cached]")
                continue

            eta_c = bisect_threshold(
                cracks, args.domain, 1.0, pd, args.trials, rng
            )
            results[crack_label][key] = {
                "polydispersity": pd,
                "eta_c": eta_c,
            }
            elapsed = time.time() - t_start
            print(f"  PD={pd:.2f}: η_c = {eta_c:.3f}  ({elapsed:.0f}s)")
            _save(output_path, results, args)

    # Also measure standard percolation (no cracks) for reference
    print("\n=== Standard percolation (no cracks) ===")
    if "standard" not in results:
        results["standard"] = {}

    for pd in polydispersities:
        key = f"{pd:.2f}"
        if key in results["standard"]:
            eta_c = results["standard"][key]["eta_c"]
            print(f"  PD={pd:.2f}: η_c = {eta_c:.3f}  [cached]")
            continue

        eta_c = bisect_threshold(
            [], args.domain, 1.0, pd, args.trials, rng
        )
        results["standard"][key] = {
            "polydispersity": pd,
            "eta_c": eta_c,
        }
        elapsed = time.time() - t_start
        print(f"  PD={pd:.2f}: η_c = {eta_c:.3f}  ({elapsed:.0f}s)")
        _save(output_path, results, args)

    _save(output_path, results, args)
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Saved to {output_path}")


def _save(path, results, args):
    data = {
        "description": "Bridge percolation with polydisperse stick lengths",
        "parameters": {
            "n_trials": args.trials,
            "domain_size": args.domain,
            "mean_length": 1.0,
            "seed": args.seed,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
