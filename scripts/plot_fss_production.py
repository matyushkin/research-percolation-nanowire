#!/usr/bin/env python3
"""Plot finite-size scaling results from fss_production.json.

Generates Figure 4: η_c(L) for standard, parallel, and Voronoi percolation
with FSS fits overlaid.

Usage:
    uv run python scripts/plot_fss_production.py

Output: figures/fss_production.pdf
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (7, 5),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fss_model(L, eta_c_inf, a, nu_inv):
    return eta_c_inf + a * L ** (-nu_inv)


def load_data():
    path = Path("data/processed/fss_production.json")
    if not path.exists():
        raise FileNotFoundError(f"Run run_fss_production.py first: {path}")
    with open(path) as f:
        return json.load(f)


def main():
    data = load_data()
    results = data["results"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): Standard + Voronoi ---
    ax = axes[0]
    colors = {"standard": "#2196F3", "voronoi": "#E91E63"}
    markers = {"standard": "o", "voronoi": "s"}
    labels = {"standard": "Standard (no cracks)", "voronoi": "Voronoi cracks"}

    for label in ["standard", "voronoi"]:
        if label not in results:
            continue

        Ls, ecs, errs = [], [], []
        for key in sorted(results[label], key=lambda k: int(k)):
            d = results[label][key]
            L = d["domain_size"]
            if label == "voronoi" and "eta_c_mean" in d:
                ec = d["eta_c_mean"]
                err = d.get("eta_c_std", 0)
            elif "eta_c" in d:
                ec = d["eta_c"]
                err = 0
            else:
                continue
            if not np.isnan(ec):
                Ls.append(L)
                ecs.append(ec)
                errs.append(err)

        Ls = np.array(Ls, dtype=float)
        ecs = np.array(ecs)
        errs = np.array(errs)

        if len(errs) > 0 and np.any(errs > 0):
            ax.errorbar(1 / Ls, ecs, yerr=errs, fmt=markers[label],
                        color=colors[label], label=labels[label],
                        capsize=3, markersize=6)
        else:
            ax.plot(1 / Ls, ecs, markers[label], color=colors[label],
                    label=labels[label], markersize=6)

        # FSS fit
        if len(Ls) >= 3:
            try:
                p0_ec = ecs[-1]
                popt, _ = curve_fit(
                    fss_model, Ls, ecs, p0=[p0_ec, 1.0, 0.75],
                    bounds=([0, -np.inf, 0.01], [20, np.inf, 5.0]),
                    maxfev=10000,
                )
                L_fit = np.linspace(Ls.min(), 200, 100)
                ax.plot(1 / L_fit, fss_model(L_fit, *popt), "--",
                        color=colors[label], alpha=0.6,
                        label=f"fit: $\\eta_c(\\infty)={popt[0]:.3f}$, "
                              f"$\\nu={1/popt[2]:.2f}$")
            except RuntimeError:
                pass

    # Reference value
    ax.axhline(5.6372, color="gray", ls=":", alpha=0.5, label="$\\eta_c = 5.637$ (theory)")

    ax.set_xlabel("$1/L$")
    ax.set_ylabel("$\\eta_c(L)$")
    ax.set_title("(a) Standard & Voronoi")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 0.25)

    # --- Panel (b): Parallel cracks ---
    ax = axes[1]

    if "parallel" in results:
        Ls, ecs = [], []
        for key in sorted(results["parallel"], key=lambda k: int(k)):
            d = results["parallel"][key]
            L = d["domain_size"]
            ec = d["eta_c"]
            if not np.isnan(ec):
                Ls.append(L)
                ecs.append(ec)

        Ls = np.array(Ls, dtype=float)
        ecs = np.array(ecs)

        ax.plot(1 / Ls, ecs, "D", color="#4CAF50", label="Parallel cracks",
                markersize=6)

        if len(Ls) >= 3:
            try:
                popt, _ = curve_fit(
                    fss_model, Ls, ecs, p0=[0.5, 5.0, 0.5],
                    bounds=([0, -np.inf, 0.01], [20, np.inf, 5.0]),
                    maxfev=10000,
                )
                L_fit = np.linspace(Ls.min(), 200, 100)
                ax.plot(1 / L_fit, fss_model(L_fit, *popt), "--",
                        color="#4CAF50", alpha=0.6,
                        label=f"fit: $\\eta_c(\\infty)={popt[0]:.3f}$, "
                              f"$\\nu={1/popt[2]:.2f}$")
            except RuntimeError:
                pass

    ax.set_xlabel("$1/L$")
    ax.set_ylabel("$\\eta_c(L)$")
    ax.set_title("(b) Parallel cracks")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 0.25)

    plt.tight_layout()

    out = Path("figures/fss_production.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"Saved: {out}")

    # Also save PNG for quick preview
    plt.savefig(out.with_suffix(".png"))
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
