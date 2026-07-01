"""
Pilot 2 decision boundary plot — Hooke's Law.

X axis: normalised strain  ε / ε_y  (log scale)
Y axis: validity score  0–1

Three predicate score curves (A1 linearity, A2 elasticity, A3 small strain)
plotted against a 1-D sweep of strain.  Training and held-out regions are
shaded; the elastic limit (ε = ε_y) is marked with a vertical dashed line.

Each predicate sees only its own criterion feature (per Decision 6):
    A1 → stress_ratio        = ε / ε_y
    A2 → strain_energy_ratio = (ε / ε_y)²
    A3 → epsilon             = ε  (raw)

Saves figures/pilot2_boundary.png at 200 dpi.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data.generate import (
    generate_hooke_dataset,
    EPSILON_Y,
    EPSILON_TRAIN_LOW,  EPSILON_TRAIN_HIGH,
    EPSILON_TEST_LOW,   EPSILON_TEST_HIGH,
    STRESS_RATIO_THRESHOLD,
    STRAIN_ENERGY_THRESHOLD,
    EPSILON_THRESHOLD,
)
from axiom_graph.graph import build_hooke_law_graph
from validity_predicates.train import train_all_hooke_predicates

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "pilot2_boundary.png",
)

# Analytical validity boundaries in normalised strain (ε / ε_y)
_A1_BOUNDARY = STRESS_RATIO_THRESHOLD                            # 0.90
_A2_BOUNDARY = STRAIN_ENERGY_THRESHOLD ** 0.5                   # sqrt(0.80) ≈ 0.894
_A3_BOUNDARY = EPSILON_THRESHOLD / EPSILON_Y                     # 0.85

# Colour scheme
C_A1 = "#2980B9"   # blue
C_A2 = "#C0392B"   # red
C_A3 = "#27AE60"   # green


def main() -> None:
    print("Generating data ...")
    train_df, held_out_df = generate_hooke_dataset(n_train=5000, n_held_out=2000, seed=42)

    print("Training Hooke's Law predicates ...")
    graph = build_hooke_law_graph()
    predicates = train_all_hooke_predicates(graph, train_df, verbose=False)
    pred_a1 = predicates["A1_linearity"]
    pred_a2 = predicates["A2_elasticity"]
    pred_a3 = predicates["A3_small_strain"]

    # ------------------------------------------------------------------
    # 1-D sweep in normalised strain  ε / ε_y  (log-spaced)
    # ------------------------------------------------------------------
    eps_norm = np.logspace(
        np.log10(EPSILON_TRAIN_LOW  / EPSILON_Y * 0.85),
        np.log10(EPSILON_TEST_HIGH  / EPSILON_Y * 1.15),
        1000,
    )
    eps_abs = eps_norm * EPSILON_Y

    # Features: each predicate sees exactly its criterion column
    feat_a1 = eps_norm.reshape(-1, 1).astype(np.float32)          # stress_ratio
    feat_a2 = (eps_norm ** 2).reshape(-1, 1).astype(np.float32)   # strain_energy_ratio
    feat_a3 = eps_abs.reshape(-1, 1).astype(np.float32)           # epsilon (raw)

    scores_a1 = pred_a1.predict(feat_a1)
    scores_a2 = pred_a2.predict(feat_a2)
    scores_a3 = pred_a3.predict(feat_a3)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- Shaded regime bands -----------------------------------------
    train_lo = EPSILON_TRAIN_LOW  / EPSILON_Y
    train_hi = EPSILON_TRAIN_HIGH / EPSILON_Y
    test_lo  = EPSILON_TEST_LOW   / EPSILON_Y
    test_hi  = EPSILON_TEST_HIGH  / EPSILON_Y

    ax.axvspan(train_lo, train_hi,
               alpha=0.13, color="steelblue",
               label=f"Training  ε ∈ [{train_lo:.2f}, {train_hi:.1f}]ε_y")
    ax.axvspan(test_lo, test_hi,
               alpha=0.10, color="tomato",
               label=f"Held-out  ε ∈ [{test_lo:.1f}, {test_hi:.0f}]ε_y  (post-yield)")

    # Unshaded gap — light annotation
    ax.annotate(
        "gap\n(no data)",
        xy=((train_hi * test_lo) ** 0.5, 0.50),   # geometric midpoint of gap
        ha="center", va="center",
        fontsize=8, color="#888",
        style="italic",
    )

    # ---- Elastic limit -----------------------------------------------
    ax.axvline(1.0, color="black", linewidth=2.0, linestyle="--",
               label="Elastic limit  ε = ε_y", zorder=4)
    ax.text(1.0, 1.025, "ε = ε_y",
            ha="center", va="bottom", fontsize=8.5, color="black",
            transform=ax.get_xaxis_transform())

    # ---- Analytical boundaries (subtle vertical lines) ---------------
    for x_bound, color, label in [
        (_A3_BOUNDARY, C_A3, "A3 criterion"),
        (_A2_BOUNDARY, C_A2, "A2 criterion"),
        (_A1_BOUNDARY, C_A1, "A1 criterion"),
    ]:
        ax.axvline(x_bound, color=color, linewidth=1.0, linestyle=":",
                   alpha=0.55, zorder=3)
        ax.text(x_bound, 0.03, f"{x_bound:.2f}",
                ha="center", va="bottom", fontsize=7, color=color, alpha=0.8)

    # ---- Decision threshold ------------------------------------------
    ax.axhline(0.5, color="#555", linewidth=1.2, linestyle=":",
               label="Decision threshold  0.5", zorder=3)

    # ---- Predicate score curves --------------------------------------
    ax.plot(eps_norm, scores_a1, color=C_A1, linewidth=2.5, zorder=5,
            label=(f"A1 — Linearity       "
                   f"[criterion: stress ratio < {STRESS_RATIO_THRESHOLD:.2f}]"))
    ax.plot(eps_norm, scores_a2, color=C_A2, linewidth=2.5, zorder=5,
            label=(f"A2 — Elasticity      "
                   f"[criterion: strain-energy ratio < {STRAIN_ENERGY_THRESHOLD:.2f}]"))
    ax.plot(eps_norm, scores_a3, color=C_A3, linewidth=2.5, zorder=5,
            label=(f"A3 — Small strain    "
                   f"[criterion: ε < {_A3_BOUNDARY:.2f}ε_y]"))

    # ---- Axes and formatting -----------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(train_lo * 0.85, test_hi * 1.15)
    ax.set_ylim(-0.05, 1.08)

    # Custom x-tick labels in units of ε_y
    x_ticks = [0.04, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:g}ε_y")
    )
    ax.tick_params(axis="x", labelsize=9)

    ax.set_xlabel("Normalised strain  ε / ε_y  (log scale)", fontsize=11)
    ax.set_ylabel("Validity score", fontsize=11)
    ax.set_title(
        "Hooke's Law (Steel Rod) — Validity Predicate Scores vs. Strain\n"
        "Trained on ε < 0.5ε_y (elastic regime) only  |  "
        "Held-out: ε > 1.5ε_y (post-yield)",
        fontsize=11,
    )

    ax.legend(fontsize=8.5, loc="center left",
              bbox_to_anchor=(0.01, 0.30),
              framealpha=0.92, edgecolor="#CCC")
    ax.grid(True, which="both", alpha=0.25)

    # ---- Region labels -----------------------------------------------
    ax.text((train_lo * train_hi) ** 0.5, 0.93, "Training\nregime",
            ha="center", va="top", fontsize=8.5,
            color="steelblue", style="italic",
            transform=ax.get_xaxis_transform())
    ax.text((test_lo * test_hi) ** 0.5, 0.93, "Held-out\nregime",
            ha="center", va="top", fontsize=8.5,
            color="tomato", style="italic",
            transform=ax.get_xaxis_transform())

    # ---- Save --------------------------------------------------------
    plt.tight_layout()
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
