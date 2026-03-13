#!/usr/bin/env python3
"""Plot all figures for bridge percolation paper.

Figures:
1. geometry_example.pdf     — crack + nanowire visualization (Fig. 1)
2. bridge_percolation_curves.pdf — P(η) parallel vs Voronoi (Fig. 2)
3. bridge_threshold_vs_cracks.pdf — η_c vs n_cracks (Fig. 3)
4. finite_size_scaling.pdf  — η_c(L) + FSS (Fig. 4)
5. polydispersity.pdf       — η_c(PD) (Fig. 5)

Usage:
    uv run python scripts/plot_bridge_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
})

FIGDIR = Path("figures")
DATADIR = Path("data/processed")


def _load(name):
    path = DATADIR / f"{name}.json"
    if not path.exists():
        print(f"  {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


# ── Fig. 1: Geometry example ──────────────────────────────────

def plot_geometry_example():
    from percolation.sticks import generate_sticks
    from percolation.cracks import (
        crack_segments_from_parallel,
        generate_voronoi_cracks,
    )

    rng = np.random.default_rng(42)
    domain = 10.0
    n_sticks = 200

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))

    for ax, title, cracks in [
        (axes[0], "(a) Parallel cracks",
         crack_segments_from_parallel(3, domain, orientation="vertical")),
        (axes[1], "(b) Voronoi cracks",
         generate_voronoi_cracks(15, domain, rng=rng)),
    ]:
        sticks = generate_sticks(n_sticks, 1.0, domain, rng)

        for p1, p2 in cracks:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "r-", linewidth=1.5, alpha=0.8)

        for s in sticks:
            ax.plot([s[0, 0], s[1, 0]], [s[0, 1], s[1, 1]],
                    "b-", linewidth=0.4, alpha=0.4)

        ax.set_xlim(0, domain)
        ax.set_ylim(0, domain)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(r"$x / L$")
        ax.set_ylabel(r"$y / L$")

    fig.tight_layout()
    out = FIGDIR / "geometry_example.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Fig. 2: P(η) curves ──────────────────────────────────────

def plot_percolation_curves():
    # Prefer HQ data
    data = _load("bridge_percolation_hq") or _load("bridge_percolation")
    if data is None:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for label, color, marker, nice in [
        ("parallel", "C0", "o", "Parallel cracks"),
        ("voronoi", "C1", "s", "Voronoi cracks"),
    ]:
        pts = data[label]
        etas = [p["eta"] for p in pts]
        # Support both key names
        probs = [p.get("p_percolation", p.get("p", 0)) for p in pts]
        ax.plot(etas, probs, f"-{marker}", color=color,
                label=nice, markersize=3, linewidth=1.0)

        if f"eta_c_{label}" in data:
            eta_c = data[f"eta_c_{label}"]
            ax.axvline(eta_c, color=color, linestyle="--",
                       alpha=0.4, linewidth=0.7)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.6)
    ax.set_xlabel(r"Dimensionless density $\eta$")
    ax.set_ylabel(r"$P(\mathrm{percolation})$")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)

    out = FIGDIR / "bridge_percolation_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Fig. 3: η_c vs crack density ─────────────────────────────

def plot_threshold_vs_cracks():
    data = _load("bridge_scan_hq") or _load("bridge_scan")
    if data is None:
        return

    results = data["results"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    if "parallel" in results:
        par = results["parallel"]
        ns = sorted(par.keys(), key=int)
        x = [par[n]["n_cracks"] for n in ns]
        y = [par[n]["eta_c"] for n in ns]
        ax.plot(x, y, "-o", color="C0", label="Parallel cracks",
                markersize=3, linewidth=1.0)

    if "voronoi" in results:
        vor = results["voronoi"]
        ns = sorted(vor.keys(), key=int)
        x = [vor[n]["n_voronoi_seeds"] for n in ns]
        y = [vor[n]["eta_c_mean"] for n in ns]
        yerr = [vor[n]["eta_c_std"] for n in ns]
        ax.errorbar(x, y, yerr=yerr, fmt="-s", color="C1",
                    label="Voronoi cracks", markersize=3,
                    capsize=2, linewidth=1.0)

    ax.axhline(5.637, color="gray", linestyle="--", alpha=0.5,
               linewidth=0.7, label=r"$\eta_c$ (no cracks)")

    ax.set_xlabel(r"Number of cracks / Voronoi seeds $n$")
    ax.set_ylabel(r"Bridge threshold $\eta_c$")
    ax.legend()
    ax.grid(alpha=0.2)

    out = FIGDIR / "bridge_threshold_vs_cracks.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Fig. 4: Finite-size scaling ───────────────────────────────

def plot_finite_size_scaling():
    # Prefer production data, fall back to HQ, then original
    data = _load("fss_production") or _load("finite_size_scaling_hq") or _load("finite_size_scaling")
    if data is None:
        return

    results = data["results"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for label, color, marker, nice in [
        ("standard", "C2", "^", "No cracks"),
        ("parallel", "C0", "o", "Parallel cracks"),
        ("voronoi", "C1", "s", "Voronoi cracks"),
    ]:
        if label not in results:
            continue
        sec = results[label]
        Ls = sorted(sec.keys(), key=lambda k: int(k))
        x, y, yerr = [], [], []
        for k in Ls:
            entry = sec[k]
            x.append(entry["domain_size"])
            if "eta_c_mean" in entry:
                y.append(entry["eta_c_mean"])
                yerr.append(entry.get("eta_c_std", 0))
            else:
                y.append(entry["eta_c"])
                yerr.append(0)

        if any(e > 0 for e in yerr):
            ax.errorbar(x, y, yerr=yerr, fmt=f"-{marker}", color=color,
                        label=nice, markersize=4, linewidth=1.0,
                        capsize=2)
        else:
            ax.plot(x, y, f"-{marker}", color=color, label=nice,
                    markersize=4, linewidth=1.0)

    ax.axhline(5.637, color="k", linestyle="--", alpha=0.7,
               linewidth=0.9, label=r"$\eta_c = 5.637$")
    ax.set_xlabel(r"Domain size $\mathcal{L} / L$")
    ax.set_ylabel(r"$\eta_c(\mathcal{L})$")
    ax.legend()
    ax.grid(alpha=0.2)

    out = FIGDIR / "finite_size_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Fig. 5: Polydispersity ────────────────────────────────────

def plot_polydispersity():
    # Prefer production data
    data = _load("polydisperse_production") or _load("polydisperse_hq2") or _load("polydisperse_hq") or _load("polydisperse")
    if data is None:
        return

    results = data["results"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Plot informative curves; skip flat parallel cracks
    curve_defs = [
        ("standard", "C2", "^", "No cracks"),
        ("voronoi_dense", "C1", "s", r"Voronoi ($n_s=25$)"),
        ("voronoi_sparse", "C3", "D", r"Voronoi ($n_s=8$)"),
        ("voronoi", "C1", "s", "Voronoi cracks"),  # fallback
    ]

    plotted = set()
    for label, color, marker, nice in curve_defs:
        if label not in results or label in plotted:
            continue
        # Skip fallback 'voronoi' if we have sparse/dense
        if label == "voronoi" and ("voronoi_dense" in results or "voronoi_sparse" in results):
            continue
        sec = results[label]
        pds = sorted(sec.keys(), key=float)
        x = [sec[k]["polydispersity"] for k in pds]
        # Support both eta_c and eta_c_mean keys
        y = [sec[k].get("eta_c", sec[k].get("eta_c_mean")) for k in pds]
        yerr = [sec[k].get("eta_c_std", 0) for k in pds]
        if any(e > 0 for e in yerr):
            ax.errorbar(x, y, yerr=yerr, fmt=f"-{marker}", color=color,
                        label=nice, markersize=3, linewidth=1.0, capsize=2)
        else:
            ax.plot(x, y, f"-{marker}", color=color, label=nice,
                    markersize=3, linewidth=1.0)
        plotted.add(label)

    ax.set_xlabel(r"Polydispersity $\sigma_L / \langle L \rangle$")
    ax.set_ylabel(r"Threshold $\eta_c$")
    ax.legend()
    ax.grid(alpha=0.2)

    out = FIGDIR / "polydispersity.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────

def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    plot_geometry_example()
    plot_percolation_curves()
    plot_threshold_vs_cracks()
    plot_finite_size_scaling()
    plot_polydispersity()
    print("Done.")


if __name__ == "__main__":
    main()
