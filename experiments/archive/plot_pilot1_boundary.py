"""
Pilot 1 decision boundary figure for the ideal gas A1 predicate.

Saves figures/pilot1_decision_boundary.png at 300 dpi.

Two subplots:
  Left  -- log-log P-V scatter with learned and analytical A1 decision
           boundaries overlaid.  Training points: blue.  Held-out: green
           (not flagged by A1) or red (flagged).
  Right -- 1D validity score along the T=400 K, n=1 mol ideal-gas
           isotherm (V = nRT/P).  Training region shaded blue, held-out
           region shaded red, threshold at 0.5.
"""

from __future__ import annotations

import os
import sys

# Allow running from project root or from experiments/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")          # non-interactive: save without display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data.generate import (
    generate_dataset,
    R,
    VDW_B,
    FREE_VOL_THRESHOLD,
    P_TRAIN_LOW, P_TRAIN_HIGH,
    P_TEST_LOW,  P_TEST_HIGH,
)
from axiom_graph.graph import build_ideal_gas_graph
from validity_predicates.train import train_all_predicates, compute_soft_labels
from reasoner.forward_chain import run_forward_chain

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "pilot1_decision_boundary.png",
)
T_REF = 400.0   # K  — isotherm reference temperature
N_REF = 1.0     # mol


def main() -> None:
    """Train predicates, evaluate held-out set, generate and save figure."""
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Data, training, inference
    # ------------------------------------------------------------------
    print("Generating data ...")
    train_df, held_out_df = generate_dataset(n_train=5000, n_held_out=2000, seed=42)

    print("Training predicates ...")
    graph = build_ideal_gas_graph()
    predicates = train_all_predicates(graph, train_df, verbose=False)
    pred_a1 = predicates["A1_point_particles"]

    print("Running forward chain ...")
    result = run_forward_chain(graph, held_out_df)
    ho_flagged_a1 = result.assumption_flagged["A1_point_particles"]  # bool (N,)

    # ------------------------------------------------------------------
    # Scatter subsamples
    # ------------------------------------------------------------------
    tr_idx = rng.choice(len(train_df),    size=min(600, len(train_df)),    replace=False)
    ho_idx = rng.choice(len(held_out_df), size=min(600, len(held_out_df)), replace=False)

    tr_P  = train_df.iloc[tr_idx]["P"].values
    tr_V  = train_df.iloc[tr_idx]["V"].values
    ho_P  = held_out_df.iloc[ho_idx]["P"].values
    ho_V  = held_out_df.iloc[ho_idx]["V"].values
    ho_fl = ho_flagged_a1[ho_idx]          # bool subset

    # ------------------------------------------------------------------
    # Grid for learned boundary contour at (T_REF, N_REF)
    # ------------------------------------------------------------------
    P_grid = np.logspace(np.log10(0.5), np.log10(260),  400)
    V_grid = np.logspace(np.log10(0.05), np.log10(50),  320)
    PP, VV = np.meshgrid(P_grid, V_grid)

    X_grid = np.column_stack([
        PP.ravel(),
        VV.ravel(),
        np.full(PP.size, T_REF),
        np.full(PP.size, N_REF),
    ]).astype(np.float32)

    scores_2d = pred_a1.predict(X_grid).reshape(PP.shape)

    # ------------------------------------------------------------------
    # 1D isotherm slice (V = N_REF * R * T_REF / P)
    # ------------------------------------------------------------------
    P_line = np.logspace(np.log10(0.5), np.log10(260), 800)
    V_line = (N_REF * R * T_REF) / P_line
    line_df = pd.DataFrame({"P": P_line, "V": V_line, "T": T_REF, "n": N_REF})

    X_line = np.column_stack([
        P_line, V_line,
        np.full(len(P_line), T_REF),
        np.full(len(P_line), N_REF),
    ]).astype(np.float32)

    scores_1d   = pred_a1.predict(X_line)
    analytic_1d = compute_soft_labels(line_df, "A1_point_particles")

    # ------------------------------------------------------------------
    # Analytical A1 boundary: (V/n) / VDW_B = FREE_VOL_THRESHOLD
    #   => V = FREE_VOL_THRESHOLD * VDW_B * n_ref
    # ------------------------------------------------------------------
    V_analytic = FREE_VOL_THRESHOLD * VDW_B * N_REF   # 10 × 0.0391 L = 0.391 L

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Ideal Gas — A1 Validity Predicate (Point Particles / No Molecular Volume)\n"
        r"Criterion: $(V/n)\,/\,b > 10$"
        f"    |    Trained on P = {P_TRAIN_LOW:.0f}–{P_TRAIN_HIGH:.0f} atm only",
        fontsize=11,
    )

    # ---- LEFT: scatter + boundary contours --------------------------------
    ax_l.scatter(
        tr_P, tr_V,
        s=8, alpha=0.45, color="steelblue", zorder=2,
    )
    ax_l.scatter(
        ho_P[~ho_fl], ho_V[~ho_fl],
        s=8, alpha=0.55, color="forestgreen", zorder=3,
    )
    ax_l.scatter(
        ho_P[ho_fl], ho_V[ho_fl],
        s=8, alpha=0.55, color="crimson", zorder=3,
    )

    # Learned decision boundary (score = 0.5 contour at T_REF)
    cs = ax_l.contour(PP, VV, scores_2d, levels=[0.5],
                      colors=["navy"], linewidths=2.5, zorder=4)

    # Analytical boundary: horizontal line at V = V_analytic
    ax_l.axhline(
        V_analytic, color="black", linewidth=2.0, linestyle="--", zorder=4,
    )

    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlim(0.5, 260)
    ax_l.set_ylim(0.05, 55)
    ax_l.set_xlabel("Pressure  P  (atm)", fontsize=11)
    ax_l.set_ylabel("Volume  V  (L)",     fontsize=11)
    ax_l.set_title(
        f"P–V space  (log–log)\nBoundary evaluated at T = {T_REF:.0f} K, n = {N_REF:.0f} mol",
        fontsize=10,
    )

    legend_l = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=7, label=f"Training  (P = {P_TRAIN_LOW:.0f}–{P_TRAIN_HIGH:.0f} atm)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="forestgreen",
               markersize=7, label="Held-out: valid / not flagged"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="crimson",
               markersize=7, label="Held-out: flagged by A1"),
        Line2D([0],[0], color="navy",  linewidth=2.5,
               label="Learned boundary  (score = 0.5)"),
        Line2D([0],[0], color="black", linewidth=2.0, linestyle="--",
               label=f"Analytical  V = {FREE_VOL_THRESHOLD:.0f}×b×n = {V_analytic:.3f} L"),
    ]
    ax_l.legend(handles=legend_l, fontsize=8, loc="upper right")

    # ---- RIGHT: 1D score vs. pressure -------------------------------------
    # Shaded regime bands
    ax_r.axvspan(P_TRAIN_LOW, P_TRAIN_HIGH, alpha=0.13, color="steelblue",
                 label=f"Training  ({P_TRAIN_LOW:.0f}–{P_TRAIN_HIGH:.0f} atm)")
    ax_r.axvspan(P_TEST_LOW,  P_TEST_HIGH,  alpha=0.10, color="tomato",
                 label=f"Held-out  ({P_TEST_LOW:.0f}–{P_TEST_HIGH:.0f} atm)")

    # Decision threshold
    ax_r.axhline(0.5, color="red", linewidth=1.5, linestyle=":",
                 label="Threshold  0.5", zorder=3)

    # Score curves
    ax_r.plot(P_line, scores_1d,   color="steelblue", linewidth=2.5,
              label="Learned score (A1 predicate)", zorder=4)
    ax_r.plot(P_line, analytic_1d, color="black",     linewidth=1.8, linestyle="--",
              label="Analytical soft label", zorder=4)

    ax_r.set_xscale("log")
    ax_r.set_xlim(0.5, 260)
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.set_xlabel("Pressure  P  (atm)", fontsize=11)
    ax_r.set_ylabel("Validity score",     fontsize=11)
    ax_r.set_title(
        f"1D slice along ideal-gas isotherm\nT = {T_REF:.0f} K,  n = {N_REF:.0f} mol  (V = nRT/P)",
        fontsize=10,
    )
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.tight_layout()
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
