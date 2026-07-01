"""
Plot the AUROC degradation curve: neural risk-ranking quality vs training distance
from the breakdown boundary, compared against a trivial single-variable baseline.

Reads:  data/degradation_curve/auroc_results.npz
Writes: figures/degradation_curve.png  (300 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np

DATA_PATH = Path("data/degradation_curve/auroc_results.npz")
FIG_PATH  = Path("figures/degradation_curve.png")

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
d = np.load(DATA_PATH)
M             = d["M"]               # [1.05, 2, 5, 10, 20]
neural_auroc  = d["neural_auroc"]
trivial_auroc = d["trivial_auroc"]

# Find M where neural AUROC first drops below 0.80 (may not exist in range)
m_below_08: float | None = None
for m_val, na in zip(M, neural_auroc):
    if na < 0.80:
        m_below_08 = float(m_val)
        break

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ---- Shaded gap between neural and trivial --------------------------------
ax.fill_between(
    M, trivial_auroc, neural_auroc,
    alpha=0.18, color="steelblue", zorder=2,
    label="_nolegend_",   # legend entry handled by fill label below
)
# Invisible proxy for the shaded region legend entry
ax.fill_between([], [], [], alpha=0.35, color="steelblue",
                label="Shaded gap = value of multivariate structure")

# ---- Main lines -----------------------------------------------------------
ax.plot(
    M, neural_auroc,
    "o-", color="steelblue", linewidth=2.5, markersize=7, zorder=4,
    label="Neural predictor (skip-MLP)",
)
ax.plot(
    M, trivial_auroc,
    "s--", color="dimgray", linewidth=1.8, markersize=5, zorder=3,
    label="Trivial baseline (threshold on x alone)",
)

# ---- Horizontal reference lines ------------------------------------------
ax.axhline(0.80, linestyle="--", color="darkorange", linewidth=1.4,
           alpha=0.90, zorder=2, label="Practically useful threshold (0.80)")
ax.axhline(0.50, linestyle="--", color="crimson",    linewidth=1.2,
           alpha=0.70, zorder=2, label="Random chance (0.50)")

# In-plot labels for reference lines using blended transform
# (axes x-coordinate, data y-coordinate)
blend = blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.988, 0.802, "practically useful\nthreshold",
        va="bottom", ha="right", fontsize=7.5, color="darkorange",
        transform=blend, linespacing=1.25, zorder=6)
ax.text(0.988, 0.502, "random chance",
        va="bottom", ha="right", fontsize=7.5, color="crimson",
        transform=blend, zorder=6)

# ---- Vertical "useful limit" annotation ----------------------------------
if m_below_08 is not None:
    ax.axvline(m_below_08, linestyle=":", color="darkorange",
               linewidth=1.8, zorder=3, alpha=0.9)
    ax.text(m_below_08 * 1.07, 0.43, "useful\nlimit",
            va="bottom", ha="left", fontsize=8, color="darkorange", zorder=6)
else:
    # Never reached in tested range: curved arrow from label to M=20 at y=0.80
    ax.annotate(
        "useful limit\n(>20×, not reached\nin tested range)",
        xy=(20.0, 0.800),
        xytext=(5.2, 0.645),
        fontsize=7.5,
        color="darkorange",
        ha="center",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color="darkorange",
            lw=1.2,
            connectionstyle="arc3,rad=0.20",
        ),
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="darkorange", alpha=0.88),
    )

# ---- Axes ----------------------------------------------------------------
ax.set_xscale("log")
ax.set_xticks(M)
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:g}×")
)
ax.xaxis.set_minor_locator(mticker.NullLocator())

ax.set_xlim(0.88, M[-1] * 1.12)
ax.set_ylim(0.40, 1.05)

ax.set_xlabel(
    "Training data distance from breakdown boundary (× L_entry)",
    fontsize=11,
)
ax.set_ylabel("AUROC on near-boundary test set", fontsize=11)
ax.set_title(
    "Risk ranking quality vs training distance from breakdown boundary",
    fontsize=12, pad=10,
)

# ---- Legend --------------------------------------------------------------
ax.legend(
    loc="lower left",
    fontsize=8.2,
    framealpha=0.93,
    handlelength=2.2,
    borderpad=0.7,
)

# ---- Explanatory text box  (upper right, below the neural AUROC line) ---
# Neural AUROC sits at y≈1.0; place box top well below that.
textbox = (
    "Model trained only on safe-regime data.\n"
    "Test set straddles the true breakdown boundary.\n"
    "Shaded region = value of multivariate structure\n"
    "over trivial baseline."
)
ax.text(
    0.988, 0.96, textbox,
    transform=ax.transAxes,
    fontsize=8,
    va="top", ha="right",
    linespacing=1.45,
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="white",
        edgecolor="#aaaaaa",
        alpha=0.93,
    ),
    zorder=7,
)

# ---- Grid ----------------------------------------------------------------
ax.yaxis.grid(True, which="major", linestyle=":", alpha=0.35, color="gray")
ax.set_axisbelow(True)

# ---- Save ----------------------------------------------------------------
FIG_PATH.parent.mkdir(exist_ok=True)
plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {FIG_PATH}")

# Summary printout
print()
print(f"Neural AUROC range : [{neural_auroc.min():.4f}, {neural_auroc.max():.4f}]")
print(f"Trivial AUROC      : {trivial_auroc[0]:.4f} (constant across all M)")
print(f"Gap range          : [{(neural_auroc - trivial_auroc).min():.4f}, "
      f"{(neural_auroc - trivial_auroc).max():.4f}]")
if m_below_08 is not None:
    print(f"Useful limit (AUROC < 0.80) reached at M = {m_below_08:g}")
else:
    print("Useful limit (AUROC < 0.80) : not reached in tested range [1.05, 20]")
