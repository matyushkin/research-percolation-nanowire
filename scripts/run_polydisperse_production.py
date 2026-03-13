#!/usr/bin/env python3
"""Production polydispersity scan: η_c(σ/⟨L⟩) for standard, Voronoi cracks.

Higher trials (300), tighter bisection (tol=0.1), more PD values.
Averages over 5 Voronoi realizations per PD to reduce geometry noise.

Usage:
    uv run python scripts/run_polydisperse_production.py [--trials 300] [--resume]

Results saved to: data/processed/polydisperse_production.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from percolation.bridge import bridge_percolation
from percolation.cracks import generate_voronoi_cracks


def generate_polydisperse_sticks(n, mean_L, pd, domain, rng):
    """Generate n sticks with log-normal length distribution."""
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
           eta_lo=0.5, eta_hi=12.0, tol=0.1, max_iter=18):
    # Auto-adjust bracket
    p_lo = bridge_prob_polydisperse(eta_lo, cracks, domain, mean_L, pd, n_trials, rng)
    if p_lo > 0.5:
        while p_lo > 0.2 and eta_lo > 0.1:
            eta_lo /= 2
            p_lo = bridge_prob_polydisperse(eta_lo, cracks, domain, mean_L, pd, n_trials, rng)

    p_hi = bridge_prob_polydisperse(eta_hi, cracks, domain, mean_L, pd, n_trials, rng)
    if p_hi < 0.5:
        while p_hi < 0.8 and eta_hi < 50:
            eta_hi *= 1.5
            p_hi = bridge_prob_polydisperse(eta_hi, cracks, domain, mean_L, pd, n_trials, rng)

    for _ in range(max_iter):
        if eta_hi - eta_lo < tol:
            break
        mid = (eta_lo + eta_hi) / 2
        p = bridge_prob_polydisperse(mid, cracks, domain, mean_L, pd, n_trials, rng)
        if p < 0.5:
            eta_lo = mid
        else:
            eta_hi = mid
    return round((eta_lo + eta_hi) / 2, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--domain", type=float, default=15.0)
    parser.add_argument("--voronoi-realizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path("data/processed/polydisperse_production.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    results = existing.get("results", {})
    t_start = time.time()

    # === Standard (no cracks) ===
    print("=" * 60)
    print("STANDARD (no cracks)")
    print("=" * 60)
    if "standard" not in results:
        results["standard"] = {}

    for pd in pds:
        key = f"{pd:.2f}"
        if key in results["standard"]:
            print(f"  PD={pd:.2f}: η_c={results['standard'][key]['eta_c']:.4f}  [cached]")
            continue
        eta_c = bisect([], args.domain, 1.0, pd, args.trials, rng)
        results["standard"][key] = {"polydispersity": pd, "eta_c": eta_c}
        elapsed = time.time() - t_start
        print(f"  PD={pd:.2f}: η_c={eta_c:.4f}  ({elapsed:.0f}s)")
        _save(output_path, results, args)

    # === Voronoi sparse (8 seeds) — averaged ===
    print("\n" + "=" * 60)
    print(f"VORONOI SPARSE (8 seeds, {args.voronoi_realizations} realizations)")
    print("=" * 60)
    if "voronoi_sparse" not in results:
        results["voronoi_sparse"] = {}

    for pd in pds:
        key = f"{pd:.2f}"
        entry = results["voronoi_sparse"].get(key, {
            "polydispersity": pd,
            "eta_c_values": [],
        })
        n_done = len(entry.get("eta_c_values", []))
        if n_done >= args.voronoi_realizations:
            print(f"  PD={pd:.2f}: η_c={entry['eta_c_mean']:.4f} ± {entry['eta_c_std']:.4f}  [cached]")
            continue

        for r in range(n_done, args.voronoi_realizations):
            crack_rng = np.random.default_rng(args.seed + 100 + r * 1000)
            cracks = generate_voronoi_cracks(8, args.domain, rng=crack_rng)
            eta_c = bisect(cracks, args.domain, 1.0, pd, args.trials, rng)
            entry.setdefault("eta_c_values", []).append(eta_c)
            elapsed = time.time() - t_start
            print(f"  PD={pd:.2f} r={r+1}: η_c={eta_c:.4f}  ({elapsed:.0f}s)")

        vals = np.array(entry["eta_c_values"])
        entry["eta_c_mean"] = round(float(np.mean(vals)), 4)
        entry["eta_c_std"] = round(float(np.std(vals)), 4)
        results["voronoi_sparse"][key] = entry
        _save(output_path, results, args)

    # === Voronoi dense (25 seeds) — averaged ===
    print("\n" + "=" * 60)
    print(f"VORONOI DENSE (25 seeds, {args.voronoi_realizations} realizations)")
    print("=" * 60)
    if "voronoi_dense" not in results:
        results["voronoi_dense"] = {}

    for pd in pds:
        key = f"{pd:.2f}"
        entry = results["voronoi_dense"].get(key, {
            "polydispersity": pd,
            "eta_c_values": [],
        })
        n_done = len(entry.get("eta_c_values", []))
        if n_done >= args.voronoi_realizations:
            print(f"  PD={pd:.2f}: η_c={entry['eta_c_mean']:.4f} ± {entry['eta_c_std']:.4f}  [cached]")
            continue

        for r in range(n_done, args.voronoi_realizations):
            crack_rng = np.random.default_rng(args.seed + 200 + r * 1000)
            cracks = generate_voronoi_cracks(25, args.domain, rng=crack_rng)
            eta_c = bisect(cracks, args.domain, 1.0, pd, args.trials, rng)
            entry.setdefault("eta_c_values", []).append(eta_c)
            elapsed = time.time() - t_start
            print(f"  PD={pd:.2f} r={r+1}: η_c={eta_c:.4f}  ({elapsed:.0f}s)")

        vals = np.array(entry["eta_c_values"])
        entry["eta_c_mean"] = round(float(np.mean(vals)), 4)
        entry["eta_c_std"] = round(float(np.std(vals)), 4)
        results["voronoi_dense"][key] = entry
        _save(output_path, results, args)

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min) → {output_path}")


def _save(path, results, args):
    data = {
        "description": "Production polydispersity: η_c vs σ/⟨L⟩",
        "parameters": {
            "n_trials": args.trials,
            "domain_size": args.domain,
            "voronoi_realizations": args.voronoi_realizations,
            "bisection_tol": 0.1,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
