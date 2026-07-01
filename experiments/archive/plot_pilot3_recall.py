"""
Grouped bar chart: Pilot 3 (Fourier heat conduction) recall across three
feature-set strategies and four assumptions.

Three bars per assumption group:
  Attempt 1 — per-criterion scalar (Kn, Fo, A3_ratio, t)
  Attempt 2 — all observables (T, L, t, dT/dx, dT/dt)
  Attempt 3 — criterion scalar + all observables

A4 bars are highlighted red to draw attention to the structural failure.
A horizontal dashed line at recall = 0.5 marks the "acceptable" threshold.
Bar labels show exact recall values.

Saves figures/pilot3_recall_comparison.png at 300 dpi.
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from data.generate import generate_fourier_dataset
from axiom_graph.graph import build_fourier_law_graph
from validity_predicates.train import train_all_fourier_predicates

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "pilot3_recall_comparison.png",
)

# Assumption IDs in display order
AIDS = ["A1_continuum", "A2_steady_state", "A3_linear_response", "A4_local_equilibrium"]
SHORT_NAMES = ["A1\nContinuum\n(Kn<0.1)", "A2\nSteady-state\n(Fo>1)",
               "A3\nLinear resp.\n(ratio<0.1)", "A4\nLocal equil.\n(t>1 ps)"]

# Label columns for violation
LABEL_COLS = {
    "A1_continuum":         "valid_continuum",
    "A2_steady_state":      "valid_steady_state",
    "A3_linear_response":   "valid_linear_response",
    "A4_local_equilibrium": "valid_local_equilibrium",
}

ATTEMPT_LABELS = [
    "Attempt 1: criterion scalar\n(Kn, Fo, A3_ratio, t)",
    "Attempt 2: all observables\n(T, L, t, dT/dx, dT/dt)",
    "Attempt 3: criterion + observables",
]

ATTEMPT_COLORS = ["#3498DB", "#27AE60", "#F39C12"]
A4_COLORS      = ["#C0392B", "#C0392B", "#C0392B"]   # red for A4 bars in all attempts


def _recall(y_violated: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """Breakdown-detection recall = TP / (TP + FN)."""
    flagged = scores < threshold
    tp = int((flagged & y_violated).sum())
    fn = int((~flagged & y_violated).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")


def run_attempt(attempt: int, train_df, held_out_df) -> dict[str, float]:
    """Train Fourier predicates for the given attempt and return recall per assumption."""
    graph = build_fourier_law_graph()
    preds = train_all_fourier_predicates(graph, train_df, attempt=attempt, verbose=False)
    recalls: dict[str, float] = {}
    for aid in AIDS:
        pred = preds.get(aid)
        if pred is None:
            recalls[aid] = 0.0
            continue
        feat_cols = pred.feature_cols
        # Check all columns are available in held_out_df
        missing = [c for c in feat_cols if c not in held_out_df.columns]
        if missing:
            recalls[aid] = float("nan")
            continue
        X_ho = held_out_df[feat_cols].values.astype("float32")
        scores = pred.predict(X_ho)
        violated = ~held_out_df[LABEL_COLS[aid]].values.astype(bool)
        recalls[aid] = _recall(violated, scores)
    return recalls


def main() -> None:
    print("Generating Fourier training / held-out data ...")
    train_df, held_out_df = generate_fourier_dataset(n_train=5000, n_held_out=3000, seed=42)

    all_recalls: list[dict[str, float]] = []
    for attempt in (1, 2, 3):
        print(f"Training Attempt {attempt} ...")
        recalls = run_attempt(attempt, train_df, held_out_df)
        all_recalls.append(recalls)
        for aid in AIDS:
            print(f"  {aid:30s}  recall={recalls[aid]:.3f}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(AIDS))
    n_attempts = 3
    bar_w = 0.24
    offsets = np.array([-1, 0, 1]) * bar_w

    for i, (label, color, recalls) in enumerate(
        zip(ATTEMPT_LABELS, ATTEMPT_COLORS, all_recalls)
    ):
        heights = [recalls[aid] for aid in AIDS]
        colors  = [A4_COLORS[i] if aid == "A4_local_equilibrium" else color
                   for aid in AIDS]
        bars = ax.bar(
            x + offsets[i], heights, bar_w,
            color=colors, alpha=0.88,
            edgecolor="white", linewidth=1.2,
            label=label, zorder=3,
        )
        for bar, val in zip(bars, heights):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.022,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if val >= 0.90 else "normal",
                    color="#7B241C" if val < 0.10 else "#1E5799",
                )

    # Decision line
    ax.axhline(0.5, color="#555", lw=1.8, ls="--", zorder=4,
               label="Acceptable recall threshold (0.5)")

    # Mark A4 failure region
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, alpha=0.06, color="#C0392B", zorder=1)
    ax.text(x[-1], 1.08,
            "A4 structural failure\n(calibration bias)",
            ha="center", va="bottom", fontsize=8.5, color="#C0392B",
            fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10.5)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Breakdown detection recall", fontsize=11)
    ax.set_title(
        "Fourier Heat Conduction — Breakdown Detection Recall\n"
        "Three feature strategies × Four assumptions  |  Silicon, 3000 held-out states",
        fontsize=11, pad=6,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Attempt-number legend ticks
    for i, att_label in enumerate(["Att.1", "Att.2", "Att.3"]):
        ax.text(0.08 + i * 0.30, -0.14, att_label, ha="center",
                transform=ax.transAxes, fontsize=8, color=ATTEMPT_COLORS[i])

    plt.tight_layout()
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
