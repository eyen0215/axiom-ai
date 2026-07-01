"""
Schematic comparing one-class anomaly detection vs validity-predicate approach.

Left panel  — A blob of training data (P, V) with a one-class SVM / convex-hull
              style boundary.  A test point inside the hull but in the actual
              violation zone is NOT flagged: the anomaly detector cannot see the
              physical boundary.

Right panel — Same data but the X axis is the physically meaningful criterion
              (free-volume per molecule V/nb).  The analytical validity boundary
              is a vertical line.  The same test point now lies clearly to the
              left of the boundary and is flagged.

Saves figures/oneclass_comparison.png at 300 dpi.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "oneclass_comparison.png",
)

# ---------------------------------------------------------
# Colour palette
# ---------------------------------------------------------
C_TRAIN  = "#2980B9"   # training blob colour
C_VALID  = "#27AE60"   # valid held-out
C_FLAG   = "#C0392B"   # flagged / violation
C_BOUND  = "#2C3E50"   # analytical boundary
C_HULL   = "#8E44AD"   # anomaly-detector hull
C_BG     = "#F8F9F9"
C_ANN    = "#7F8C8D"   # annotation text


def _ellipse_boundary(cx, cy, rx, ry, n=200):
    """Return (x, y) points of an ellipse."""
    theta = np.linspace(0, 2 * np.pi, n)
    return cx + rx * np.cos(theta), cy + ry * np.sin(theta)


def main() -> None:
    rng = np.random.default_rng(7)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor(C_BG)
    for ax in (ax_l, ax_r):
        ax.set_facecolor(C_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ================================================================
    # LEFT PANEL — (P, V) feature space, anomaly detector fails
    # ================================================================

    # Training data cluster (low P, high V — ideal-gas regime)
    n_tr = 200
    P_tr = rng.uniform(1, 10, n_tr)
    V_tr = rng.uniform(5, 50, n_tr) + rng.normal(0, 1, n_tr)
    V_tr = np.clip(V_tr, 3, 55)

    # Held-out VALID states (high P, large V — analytic still OK, just extrapolated)
    n_ho_ok = 40
    P_ok = rng.uniform(50, 200, n_ho_ok)
    V_ok = rng.uniform(18, 50, n_ho_ok)

    # Held-out VIOLATED states (high P, small V — free volume collapses)
    # CRITICALLY: these still have (P, V) overlapping the anomaly hull on the left!
    n_ho_bad = 30
    P_bad = rng.uniform(50, 200, n_ho_bad)
    V_bad = rng.uniform(1.5, 6, n_ho_bad)   # small V — violation zone

    ax_l.scatter(P_tr,  V_tr,  s=22, color=C_TRAIN, alpha=0.60, zorder=3,
                 label="Training data  (low P)")
    ax_l.scatter(P_ok,  V_ok,  s=22, color=C_VALID,  alpha=0.70, zorder=4,
                 label="Held-out: actually valid")
    ax_l.scatter(P_bad, V_bad, s=50, color=C_FLAG,   alpha=0.85, zorder=5,
                 marker="X", label="Held-out: VIOLATED  (undetected)")

    # Anomaly-detector envelope — an ellipse that covers the training cluster
    # and ALSO covers some P_bad states projected on V, making them look "normal"
    hull_x, hull_y = _ellipse_boundary(cx=12, cy=25, rx=14, ry=22)
    ax_l.plot(hull_x, hull_y, color=C_HULL, lw=2.2, ls="--", zorder=2,
              label="One-class boundary  (anomaly detector)")

    # Annotation: a specific "trap" point inside the hull but violated
    trap_P, trap_V = 8.0, 3.8
    ax_l.annotate(
        "Inside hull\nbut violated!",
        xy=(trap_P, trap_V),
        xytext=(20, 10),
        fontsize=8.2, color=C_FLAG, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_FLAG, lw=1.4),
        zorder=6,
    )
    ax_l.scatter([trap_P], [trap_V], s=110, color=C_FLAG, zorder=7, marker="X")

    ax_l.set_xlim(-3, 230)
    ax_l.set_ylim(-3, 58)
    ax_l.set_xlabel("Pressure  P  (atm)", fontsize=11)
    ax_l.set_ylabel("Volume  V  (L)", fontsize=11)
    ax_l.set_title(
        "One-class anomaly detector  (P–V feature space)\n"
        "Test point inside training hull  →  NOT flagged",
        fontsize=10.5, color="#2C3E50", pad=8,
    )
    ax_l.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax_l.text(
        0.02, 0.97,
        "Training\nregime",
        transform=ax_l.transAxes,
        fontsize=9, color=C_TRAIN, va="top", style="italic",
    )

    # ================================================================
    # RIGHT PANEL — free-volume criterion space, predicate succeeds
    # ================================================================

    # VDW_B for N₂ or similar: b ≈ 0.039 L/mol; with n=1 mol threshold at V/nb=10
    VDW_B_N   = 0.039   # L/mol
    N_MOL     = 1.0
    THRESHOLD  = 10.0   # (V/n) / b

    # Map the same data to free-volume ratio
    FV_tr    = V_tr    / (N_MOL * VDW_B_N)   # training
    FV_ok    = V_ok    / (N_MOL * VDW_B_N)   # valid held-out
    FV_bad   = V_bad   / (N_MOL * VDW_B_N)   # violated

    trap_FV  = trap_V  / (N_MOL * VDW_B_N)

    # Jitter y positions for display (simulated validity score 0-1)
    y_tr   = np.clip(FV_tr  / (FV_tr  + THRESHOLD), 0.50, 0.99)
    y_ok   = np.clip(FV_ok  / (FV_ok  + THRESHOLD), 0.50, 0.99)
    y_bad  = np.clip(FV_bad / (FV_bad + THRESHOLD), 0.01, 0.49)
    y_trap = trap_FV / (trap_FV + THRESHOLD)

    # Plot as 1D strip to emphasise the single-axis separation
    ax_r.scatter(FV_tr,  y_tr  + rng.normal(0, 0.015, len(FV_tr)),
                 s=22, color=C_TRAIN, alpha=0.60, zorder=3,
                 label="Training data")
    ax_r.scatter(FV_ok,  y_ok  + rng.normal(0, 0.015, len(FV_ok)),
                 s=22, color=C_VALID,  alpha=0.70, zorder=4,
                 label="Held-out: actually valid")
    ax_r.scatter(FV_bad, y_bad + rng.normal(0, 0.015, len(FV_bad)),
                 s=50, color=C_FLAG,   alpha=0.85, zorder=5, marker="X",
                 label="Held-out: VIOLATED  (correctly flagged)")

    ax_r.scatter([trap_FV], [y_trap], s=130, color=C_FLAG, zorder=7, marker="X")

    # Analytical boundary — vertical dashed line at free-volume = threshold
    ax_r.axvline(THRESHOLD, color=C_BOUND, lw=2.5, ls="--", zorder=6,
                 label=f"Validity boundary  (V/nb = {THRESHOLD:.0f})")

    # Decision threshold — horizontal at 0.5
    ax_r.axhline(0.5, color="#555", lw=1.2, ls=":", zorder=5,
                 label="Score threshold  0.5")

    # Shade regions
    ax_r.axvspan(0, THRESHOLD, alpha=0.07, color=C_FLAG, zorder=0)
    ax_r.axvspan(THRESHOLD, 1500, alpha=0.07, color=C_VALID, zorder=0)

    # Annotation for the same trap point
    ax_r.annotate(
        "Now correctly\nflagged!",
        xy=(trap_FV, y_trap),
        xytext=(trap_FV + 40, y_trap + 0.15),
        fontsize=8.2, color=C_FLAG, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_FLAG, lw=1.4),
        zorder=8,
    )

    # Region labels
    ax_r.text(THRESHOLD * 0.55, 0.92, "Violation\nzone",
              ha="center", fontsize=9, color=C_FLAG, style="italic")
    ax_r.text(THRESHOLD * 4.5, 0.92, "Valid\nzone",
              ha="center", fontsize=9, color=C_VALID, style="italic")

    ax_r.set_xlim(-20, 1450)
    ax_r.set_ylim(-0.04, 1.04)
    ax_r.set_xlabel(r"Free-volume ratio  $(V/n)\,/\,b$", fontsize=11)
    ax_r.set_ylabel("Validity score", fontsize=11)
    ax_r.set_title(
        "Validity predicate  (physics-informed criterion)\n"
        "Same test point left of boundary  →  FLAGGED",
        fontsize=10.5, color="#2C3E50", pad=8,
    )
    ax_r.legend(fontsize=8, loc="lower right", framealpha=0.95)

    # ================================================================
    # Main title
    # ================================================================
    fig.suptitle(
        "Why anomaly detection fails in the ideal gas regime:\n"
        "the violation zone overlaps the training hull in raw feature space",
        fontsize=11.5, color="#2C3E50", y=1.01, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
