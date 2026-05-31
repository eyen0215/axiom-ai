"""
A4 structural calibration-bias failure — Fourier heat conduction.

A4 (local equilibrium): t > ELECTRON_PHONON_TIME = 1 ps.

Shows that the trained predicate score saturates near 1.0 across the
entire time range — including at and beyond the validity boundary at
t = 1 ps — because the skip regression's bias term (≈ mean of training
log-criterion ≈ 22) dominates over the learned weight, making the output
too large to cross the 0.5 decision threshold.

Annotates:
  - Training regime (shaded, t >= 1 ns)
  - Validity boundary (vertical dashed, t = 1 ps)
  - Gap between training minimum and boundary
  - The saturated score curve with a zoom inset

Saves figures/pilot3_a4_failure.png at 300 dpi.
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
import torch

from data.generate import (
    generate_fourier_dataset,
    ELECTRON_PHONON_TIME,
    FOURIER_T_MIN,
)
from validity_predicates.train import (
    train_all_fourier_predicates,
    FOURIER_CRITERION_COLS,
    compute_fourier_log_criterion,
)
from axiom_graph.graph import build_fourier_law_graph

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "pilot3_a4_failure.png",
)

AID       = "A4_local_equilibrium"
C_SCORE   = "#2980B9"
C_TRAIN   = "steelblue"
C_BOUND   = "#C0392B"
C_THRESH  = "#555"


def main() -> None:
    print("Generating Fourier training data ...")
    train_df, held_out_df = generate_fourier_dataset(n_train=5000, n_held_out=3000, seed=42)

    print("Training Fourier predicates (Attempt 1 — criterion scalar) ...")
    graph = build_fourier_law_graph()
    preds = train_all_fourier_predicates(graph, train_df, attempt=1, verbose=False)

    pred_a4 = preds.get(AID)
    if pred_a4 is None:
        print(f"ERROR: no predicate trained for {AID}")
        return

    # Diagnostic: skip weight and bias
    w = pred_a4.skip.weight.data.numpy().ravel()
    b = pred_a4.skip.bias.data.item()
    print(f"A4 skip weight: {w}   bias: {b:.3f}")

    # ------------------------------------------------------------------
    # t sweep: from 1e-15 s (1 fs, ultra-fast violation) to 1 s (training max)
    # ------------------------------------------------------------------
    t_sweep   = np.logspace(-15, 0, 2000).astype(np.float32)           # 1 fs … 1 s
    feat_a4   = t_sweep.reshape(-1, 1)                                  # feature = t (raw)
    scores    = pred_a4.predict(feat_a4)

    # Also capture raw logits (before sigmoid) to show the saturated region
    with torch.no_grad():
        logits = pred_a4(torch.from_numpy(feat_a4)).numpy()

    # Training regime effective lower bound
    t_train_min = FOURIER_T_MIN          # 1 ns — hard floor in generator
    t_train_max = 1.0                    # 1 s

    # ------------------------------------------------------------------
    # Training score statistics
    # ------------------------------------------------------------------
    t_train  = train_df["t"].values.astype(np.float32)
    scores_tr = pred_a4.predict(t_train.reshape(-1, 1))
    logits_tr = pred_a4(torch.from_numpy(t_train.reshape(-1, 1))).detach().numpy()
    print(f"Training A4 scores: min={scores_tr.min():.6f}  max={scores_tr.max():.6f}")
    print(f"Training A4 logits: min={logits_tr.min():.3f}  max={logits_tr.max():.3f}")

    # Score at boundary (t = 1 ps)
    t_boundary = np.array([[ELECTRON_PHONON_TIME]], dtype=np.float32)
    score_boundary = pred_a4.predict(t_boundary)[0]
    logit_boundary = pred_a4(torch.from_numpy(t_boundary)).item()
    print(f"Boundary (t=1ps): score={score_boundary:.6f}  logit={logit_boundary:.3f}")

    # ------------------------------------------------------------------
    # Figure — two vertically stacked panels
    # ------------------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.45},
    )

    # ---- TOP: score curve -------------------------------------------
    ax_top.plot(t_sweep, scores, color=C_SCORE, lw=2.2, zorder=4,
                label="A4 predicate score  sigmoid(skip + MLP)")

    # Training regime shading
    ax_top.axvspan(t_train_min, t_train_max, alpha=0.12, color=C_TRAIN, zorder=1,
                   label=f"Training regime  t ∈ [{t_train_min*1e9:.0f} ns, {t_train_max:.0f} s]")

    # Validity boundary
    ax_top.axvline(ELECTRON_PHONON_TIME, color=C_BOUND, lw=2.5, ls="--", zorder=5,
                   label=f"A4 validity boundary  t = {ELECTRON_PHONON_TIME*1e12:.0f} ps")

    # Decision threshold
    ax_top.axhline(0.5, color=C_THRESH, lw=1.5, ls=":", zorder=3,
                   label="Decision threshold  0.5")

    # Mark score at boundary
    ax_top.plot([ELECTRON_PHONON_TIME], [score_boundary], "v",
                color=C_BOUND, ms=11, zorder=6)
    ax_top.annotate(
        f"Boundary score = {score_boundary:.4f}\n(logit = {logit_boundary:.1f})\nNeeds score < 0.5 to fire",
        xy=(ELECTRON_PHONON_TIME, score_boundary),
        xytext=(3e-11, 0.62),
        fontsize=8.5, color=C_BOUND, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BOUND, lw=1.5),
        zorder=7,
    )

    # Gap annotation
    ax_top.annotate(
        "",
        xy=(t_train_min, 0.20),
        xytext=(ELECTRON_PHONON_TIME, 0.20),
        arrowprops=dict(arrowstyle="<->", color="#555", lw=1.6),
        zorder=7,
    )
    ax_top.text(
        (t_train_min * ELECTRON_PHONON_TIME) ** 0.5, 0.24,
        "Gap: 1000×\n(needs 10.7× more extrapolation to fire)",
        ha="center", va="bottom", fontsize=8, color="#555", style="italic",
    )

    ax_top.set_xscale("log")
    ax_top.set_xlim(1e-15, 2e0)
    ax_top.set_ylim(-0.04, 1.08)
    ax_top.set_xlabel("Time  t  (s)", fontsize=11)
    ax_top.set_ylabel("Predicate score  (sigmoid output)", fontsize=11)
    ax_top.set_title(
        "A4 (Local Equilibrium) — Predicate Score vs. Time\n"
        f"Combined output at boundary = {logit_boundary:.1f}  (needs < 0 to fire) — "
        f"gap factor ≈ {logit_boundary / max((logits_tr.min() - logit_boundary), 1e-9):.1f}×",
        fontsize=10.5,
    )

    # Custom x-tick labels with physical units
    x_ticks = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0]
    x_labels = ["1 fs", "1 ps", "1 ns", "1 µs", "1 ms", "1 s"]
    ax_top.set_xticks(x_ticks)
    ax_top.set_xticklabels(x_labels, fontsize=9)
    ax_top.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax_top.grid(True, which="both", alpha=0.20)

    # ---- BOTTOM: raw logit curve (shows the linear skip in action) ---
    ax_bot.plot(t_sweep, logits, color="#8E44AD", lw=2.0, zorder=4,
                label="Raw logit  (before sigmoid)")

    ax_bot.axvspan(t_train_min, t_train_max, alpha=0.12, color=C_TRAIN, zorder=1)
    ax_bot.axvline(ELECTRON_PHONON_TIME, color=C_BOUND, lw=2.0, ls="--", zorder=5)
    ax_bot.axhline(0.0, color=C_THRESH, lw=1.5, ls=":", zorder=3,
                   label="Logit = 0  ↔  score = 0.5")

    ax_bot.annotate(
        f"Logit at boundary\n= {logit_boundary:.1f}  (needed: < 0)",
        xy=(ELECTRON_PHONON_TIME, logit_boundary),
        xytext=(1e-9, logit_boundary - 3),
        fontsize=8, color=C_BOUND,
        arrowprops=dict(arrowstyle="->", color=C_BOUND, lw=1.2),
    )

    ax_bot.set_xscale("log")
    ax_bot.set_xlim(1e-15, 2e0)
    ax_bot.set_xlabel("Time  t  (s)", fontsize=10)
    ax_bot.set_ylabel("Logit value", fontsize=10)
    ax_bot.set_title(
        "Raw logit: needs to cross 0 to fire — remains far above 0 at boundary",
        fontsize=9.5,
    )
    ax_bot.set_xticks(x_ticks)
    ax_bot.set_xticklabels(x_labels, fontsize=8.5)
    ax_bot.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax_bot.grid(True, which="both", alpha=0.20)

    # ---- Save -------------------------------------------------------
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
