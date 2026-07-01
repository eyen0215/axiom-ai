"""
Provenance graph for the ideal gas axiom derivation.

Saves figures/pilot1_provenance.png at 300 dpi.

Nodes are colored:
    red   -- assumption flagged, or derived result whose provenance contains
             a flagged assumption (flag propagated downstream)
    green -- assumption not flagged / derived result with clean provenance
    gray  -- assumption with no operationalizable criterion (n/a)

Edge colors follow the source node so the "contamination stream" is visible.

Uses a real held-out example state where A1 and A2 both fire clearly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from data.generate import generate_dataset
from axiom_graph.graph import build_ideal_gas_graph
from validity_predicates.train import train_all_predicates
from reasoner.forward_chain import run_forward_chain

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "pilot1_provenance.png",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_RED   = "#C0392B"   # flagged
C_GREEN = "#27AE60"   # valid / clean provenance
C_GRAY  = "#7F8C8D"   # n/a (no predicate attached)
C_WHITE = "#FFFFFF"
C_BG    = "#F2F3F4"

# ---------------------------------------------------------------------------
# Graph topology (short keys → (premise_shorts, conclusion_short))
# ---------------------------------------------------------------------------
#   E1: [A3]            → D1
#   E2: [A1, A2, A3]    → D2
#   E3: [A4]            → D3
#   E4: [D1, D2, D3]    → D4
#   E5: [D4, A1]        → D5   ← A1 used again (second occurrence)
#   E6: [D5]            → D6
EDGES: list[tuple[str, str]] = [
    ("A3", "D1"),
    ("A1", "D2"),
    ("A2", "D2"),
    ("A3", "D2"),
    ("A4", "D3"),
    ("D1", "D4"),
    ("D2", "D4"),
    ("D3", "D4"),
    ("D4", "D5"),
    ("A1", "D5"),   # second use of A1 (full free volume)
    ("D5", "D6"),
]

# Full node IDs as used by the forward chain
FULL_ID: dict[str, str] = {
    "A1": "A1_point_particles",
    "A2": "A2_no_forces",
    "A3": "A3_elastic_collisions",
    "A4": "A4_thermal_equilibrium",
    "D1": "D1_momentum_transfer",
    "D2": "D2_collision_frequency",
    "D3": "D3_mean_kinetic_energy",
    "D4": "D4_single_particle_pressure",
    "D5": "D5_pressure_ideal",
    "D6": "D6_ideal_gas_law",
}

# Node positions in data coordinates (x, y)
POS: dict[str, tuple[float, float]] = {
    "A1": (1.4,  5.0),
    "A2": (3.8,  5.0),
    "A3": (6.4,  5.0),
    "A4": (8.4,  5.0),
    "D1": (6.4,  3.3),
    "D2": (2.6,  3.3),
    "D3": (8.4,  3.3),
    "D4": (5.0,  1.6),
    "D5": (5.0,  0.1),
    "D6": (5.0, -1.4),
}

# Labels shown inside each node (short, multi-line)
LABEL: dict[str, str] = {
    "A1": "A1 — Point particles\nno molecular volume",
    "A2": "A2 — No forces\nno intermolecular forces",
    "A3": "A3 — Elastic collisions\n[no criterion]",
    "A4": "A4 — Thermal equilibrium\n[no criterion]",
    "D1": "D1\nMomentum transfer\nΔp = 2mv_x",
    "D2": "D2\nCollision frequency\nf = v_x / (2L)",
    "D3": "D3\nMean kinetic energy\nm⟨v_x²⟩ = kT",
    "D4": "D4\nParticle pressure\nP₁ = m⟨v_x²⟩ / V",
    "D5": "D5\nIdeal pressure\nPV = NkT",
    "D6": "D6  ★ PRIMARY\nIdeal Gas Law\nPV = nRT",
}

# Arc curvature for each edge (rad=0 → straight)
EDGE_RAD: dict[tuple[str, str], float] = {
    ("A1", "D5"): -0.38,   # skip-connection; curves left to avoid graph interior
    ("A3", "D2"):  0.18,   # slight curve to distinguish from A3→D1
    ("D3", "D4"):  0.14,
    ("D1", "D4"): -0.10,
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_node(
    ax: plt.Axes,
    sid: str,
    color: str,
    score_line: str | None = None,
) -> None:
    """Draw a rounded rectangle with label (and optional score line)."""
    x, y = POS[sid]
    is_assumption = sid.startswith("A")
    is_primary    = sid == "D6"

    w = 2.10 if is_assumption else 2.20
    h = 0.90 if score_line is None else 1.05
    if is_primary:
        h = 0.95

    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.07",
        facecolor=color,
        edgecolor=C_WHITE if not is_primary else "#F39C12",
        linewidth=3.0 if is_primary else 1.8,
        zorder=3,
    )
    ax.add_patch(box)

    label = LABEL[sid]
    label_y = y if score_line is None else y + 0.13
    ax.text(
        x, label_y, label,
        ha="center", va="center",
        fontsize=7.5, color=C_WHITE, fontweight="bold",
        linespacing=1.35, zorder=4,
    )

    if score_line:
        ax.text(
            x, y - h / 2 + 0.16, score_line,
            ha="center", va="bottom",
            fontsize=7.2, color=C_WHITE, alpha=0.95,
            style="italic", zorder=4,
        )


def _draw_edge(
    ax: plt.Axes,
    src: str,
    tgt: str,
    color: str,
    rad: float = 0.0,
) -> None:
    """Draw a curved directed arrow from src to tgt."""
    lw = 2.6 if color == C_RED else 1.6
    alpha = 1.0 if color == C_RED else 0.75
    ax.annotate(
        "",
        xy=POS[tgt],
        xytext=POS[src],
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=14,
        ),
        zorder=2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating data and training predicates ...")
    train_df, held_out_df = generate_dataset(n_train=5000, n_held_out=2000, seed=42)
    graph = build_ideal_gas_graph()
    train_all_predicates(graph, train_df, verbose=False)

    print("Running forward chain on held-out set ...")
    result = run_forward_chain(graph, held_out_df)

    # ------------------------------------------------------------------
    # Pick example state: both A1 and A2 clearly flagged, highest P
    # ------------------------------------------------------------------
    s_a1 = result.assumption_scores["A1_point_particles"]
    s_a2 = result.assumption_scores["A2_no_forces"]
    # Relax threshold progressively until we find candidates
    for thr in (0.05, 0.1, 0.2, 0.4, 0.5):
        both = (s_a1 < thr) & (s_a2 < thr)
        candidates = np.where(both)[0]
        if len(candidates) > 0:
            break
    idx = candidates[np.argmax(held_out_df.iloc[candidates]["P"].values)]

    state  = held_out_df.iloc[idx]
    sc_a1  = float(s_a1[idx])
    sc_a2  = float(s_a2[idx])
    nf     = result.node_flagged

    print(
        f"Example: P={state['P']:.1f} atm, V={state['V']:.4f} L, "
        f"T={state['T']:.0f} K, n={state['n']:.0f} mol\n"
        f"  A1={sc_a1:.4f}  A2={sc_a2:.4f}"
    )

    # ------------------------------------------------------------------
    # Assign colours
    # ------------------------------------------------------------------
    def color_of(sid: str) -> str:
        fid = FULL_ID[sid]
        if sid == "A1":
            return C_RED
        if sid == "A2":
            return C_RED
        if sid in ("A3", "A4"):
            return C_GRAY
        return C_RED if nf.get(fid, np.zeros(1, bool))[idx] else C_GREEN

    colors = {sid: color_of(sid) for sid in POS}

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-2.5, 6.3)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Faint horizontal layer bands
    for y_band, label, alpha in [
        (5.0, "Assumptions", 0.08),
        (3.3, "Derived — 1st order", 0.05),
        (1.6, "Derived — 2nd order", 0.05),
        (0.1, "Derived — 3rd order", 0.05),
        (-1.4,"Primary output", 0.07),
    ]:
        band = mpatches.FancyBboxPatch(
            (-0.3, y_band - 0.65), 10.6, 1.30,
            boxstyle="round,pad=0.05",
            facecolor="#AEB6BF", edgecolor="none",
            alpha=alpha, zorder=0,
        )
        ax.add_patch(band)
        ax.text(
            -0.1, y_band, label,
            ha="right", va="center",
            fontsize=8, color="#555",
            style="italic",
        )

    # Edges (drawn before nodes so arrows appear under boxes)
    for src, tgt in EDGES:
        rad = EDGE_RAD.get((src, tgt), 0.0)
        _draw_edge(ax, src, tgt, colors[src], rad=rad)

    # Nodes
    for sid in POS:
        score_line = None
        if sid == "A1":
            score_line = f"score = {sc_a1:.3f}  — FLAGGED"
        elif sid == "A2":
            score_line = f"score = {sc_a2:.3f}  — FLAGGED"
        _draw_node(ax, sid, colors[sid], score_line=score_line)

    # ------------------------------------------------------------------
    # Example state box (bottom centre)
    # ------------------------------------------------------------------
    state_str = (
        f"Example held-out state\n"
        f"P = {state['P']:.1f} atm    V = {state['V']:.4f} L"
        f"    T = {state['T']:.0f} K    n = {state['n']:.0f} mol\n"
        f"Free-volume ratio (V/n)/b = "
        f"{(state['V']/state['n'])/0.0391:.1f}  (threshold 10)    "
        f"Interaction ratio = "
        f"{1.39*state['n']/(state['V']*0.08206*state['T']):.3f}  (threshold 0.10)"
    )
    ax.text(
        4.8, -2.05, state_str,
        ha="center", va="center",
        fontsize=8.5, color="#2C3E50",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=C_WHITE, edgecolor="#AEB6BF",
            linewidth=1.5, alpha=0.97,
        ),
        zorder=5,
        linespacing=1.5,
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_patches = [
        mpatches.Patch(facecolor=C_RED,   edgecolor=C_WHITE, linewidth=1.5,
                       label="Flagged  (score < 0.5 or flag propagated)"),
        mpatches.Patch(facecolor=C_GREEN, edgecolor=C_WHITE, linewidth=1.5,
                       label="Not flagged  (clean provenance)"),
        mpatches.Patch(facecolor=C_GRAY,  edgecolor=C_WHITE, linewidth=1.5,
                       label="No criterion  (n/a — treated as always valid)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.97,
        edgecolor="#BDC3C7",
        bbox_to_anchor=(1.01, -0.01),
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        "Ideal Gas Axiom Graph — Provenance & Flag Propagation\n"
        "A1 and A2 fire at high pressure; flag propagates to D2, D4, D5, D6 via provenance union",
        fontsize=11.5,
        pad=10,
        color="#2C3E50",
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.tight_layout()
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight", facecolor=C_BG)
    print(f"\nSaved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
