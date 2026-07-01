"""
3×3 colored-cell grid showing attribution accuracy vs training distance.

Rows:    Scenario X (A1 breaks), Y (A2 breaks), Z (A3 breaks)
Columns: M = 1.05 (near), 10 (far), 20 (very far)

Each cell:
  Background  — green if attribution correct, red if wrong/missed
  Text rows   — A1 / A2 / A3 with ✓/✗ per predicate + fire rate %
  Footer      — "CORRECT" or "MISSED" verdict

Imports scenario generators and model loaders from attribution_degraded.py
so results are always consistent with that script.

Saves: figures/attribution_accuracy.png  (300 dpi)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))   # experiments/ dir for direct import

from attribution_degraded import (
    generate_scenario_X,
    generate_scenario_Y,
    generate_scenario_Z,
    load_A1,
    load_A2,
    load_A3,
    fire_rate,
    N,
    SEED,
    FIRE_THR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIG_PATH = ROOT / "figures" / "attribution_accuracy.png"

Ms_TEST    = [1.05, 10, 20]
SCENARIOS  = ["X", "Y", "Z"]

# Which assumption is expected to fire for each scenario (index into A1/A2/A3)
EXPECTED = {
    "X": (True,  False, False),
    "Y": (False, True,  False),
    "Z": (False, False, True),
}
EXPECTED_IDX = {"X": 0, "Y": 1, "Z": 2}

ASSUMP_NAMES = ["A1", "A2", "A3"]

COL_LABELS = [
    "M = 1.05×\n(near boundary)",
    "M = 10×\n(far)",
    "M = 20×\n(very far)",
]
ROW_LABELS = [
    "Scenario X\n(A1 breaks:\nturbulent flow)",
    "Scenario Y\n(A2 breaks:\nentrance region)",
    "Scenario Z\n(A3 breaks:\ncompressible flow)",
]

# Colors — strong enough to read at 300 dpi in print
C_CORRECT = "#2e7d32"   # material green 800
C_WRONG   = "#b71c1c"   # material red 900
C_TEXT    = "white"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_results() -> dict:
    """Return {M: {scenario: (fire_A1, fire_A2, fire_A3)}} for all cells."""
    pred_A1 = load_A1()
    pred_A3 = load_A3()

    sc_data = {
        "X": generate_scenario_X(N, SEED),
        "Y": generate_scenario_Y(N, SEED + 1),
        "Z": generate_scenario_Z(N, SEED + 2),
    }

    out: dict = {}
    for M in Ms_TEST:
        pred_A2 = load_A2(M)
        out[M] = {}
        for sc_name, sc in sc_data.items():
            out[M][sc_name] = (
                fire_rate(pred_A1, sc["A1_feats"]),
                fire_rate(pred_A2, sc["A2_feats"]),
                fire_rate(pred_A3, sc["A3_feats"]),
            )
    return out


def cell_correct(rates: tuple, scenario: str) -> bool:
    exp = EXPECTED[scenario]
    return all((r > FIRE_THR) == e for r, e in zip(rates, exp))


# ---------------------------------------------------------------------------
# Cell rendering
# ---------------------------------------------------------------------------

def draw_cell(ax: plt.Axes, rates: tuple, scenario: str, correct: bool) -> None:
    """Fill ax with background + assumption rows + verdict footer."""
    ax.set_facecolor(C_CORRECT if correct else C_WRONG)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(2.5)
    ax.set_xticks([])
    ax.set_yticks([])

    exp     = EXPECTED[scenario]
    exp_idx = EXPECTED_IDX[scenario]
    T       = ax.transAxes

    # ── Header: expected assumption name ──────────────────────────────────
    exp_name = ASSUMP_NAMES[exp_idx]
    ax.text(0.50, 0.91, f"expects {exp_name} to fire",
            transform=T, color=C_TEXT, fontsize=7.5, alpha=0.72,
            va="top", ha="center", style="italic")

    # ── Per-assumption rows ───────────────────────────────────────────────
    # y positions for three rows
    ys = [0.72, 0.50, 0.28]

    for i, (name, rate, y) in enumerate(zip(ASSUMP_NAMES, rates, ys)):
        fired    = rate > FIRE_THR
        expected = exp[i]
        matches  = fired == expected
        marker   = "✓" if matches else "✗"

        is_exp  = (i == exp_idx)
        fw_name = "bold" if is_exp else "normal"
        alpha   = 1.0   if is_exp else 0.80

        # assumption name
        ax.text(0.10, y, name,
                transform=T, color=C_TEXT, fontsize=11,
                fontweight=fw_name, va="center", ha="left", alpha=alpha)

        # ✓ / ✗
        ax.text(0.36, y, marker,
                transform=T, color=C_TEXT, fontsize=13,
                fontweight="bold", va="center", ha="center")

        # fire rate %
        ax.text(0.60, y, f"{rate * 100:.0f}%",
                transform=T, color=C_TEXT, fontsize=10,
                va="center", ha="center", alpha=alpha)

        # fired / silent label (small)
        status = "fired" if fired else "silent"
        ax.text(0.95, y, f"({status})",
                transform=T, color=C_TEXT, fontsize=7.5,
                va="center", ha="right", alpha=0.65)

    # ── Thin divider above verdict ────────────────────────────────────────
    ax.plot([0.05, 0.95], [0.16, 0.16],
            transform=T, color="white", alpha=0.30,
            linewidth=0.8, solid_capstyle="round")

    # ── Verdict footer ────────────────────────────────────────────────────
    verdict = "CORRECT" if correct else "MISSED"
    ax.text(0.50, 0.08, verdict,
            transform=T, color=C_TEXT, fontsize=9,
            fontweight="bold", va="center", ha="center",
            alpha=0.90, style="italic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = compute_results()

    fig, axes = plt.subplots(
        3, 3, figsize=(11.5, 8.2),
        gridspec_kw={"hspace": 0.08, "wspace": 0.08},
    )

    # Leave room: left for row labels, top for col labels + suptitle, bottom for caption
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.12)

    # ── Draw cells ────────────────────────────────────────────────────────
    for i, sc_name in enumerate(SCENARIOS):
        for j, M in enumerate(Ms_TEST):
            rates   = results[M][sc_name]
            correct = cell_correct(rates, sc_name)
            draw_cell(axes[i, j], rates, sc_name, correct)

    # ── Column labels: fig.text just above each top-row axes ──────────────
    # Must call get_position() AFTER subplots_adjust for correct coordinates
    for j, (label, ax_col) in enumerate(zip(COL_LABELS, axes[0, :])):
        pos   = ax_col.get_position()
        x_ctr = (pos.x0 + pos.x1) / 2
        fig.text(x_ctr, pos.y1 + 0.018, label,
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                 color="#111111", linespacing=1.35)

    # ── Row labels: fig.text to the left of each left-column axes ─────────
    for i, (label, ax_row) in enumerate(zip(ROW_LABELS, axes[:, 0])):
        pos     = ax_row.get_position()
        y_ctr   = (pos.y0 + pos.y1) / 2
        x_label = pos.x0 - 0.018
        fig.text(x_label, y_ctr, label,
                 ha="right", va="center", fontsize=9.5, color="#111111",
                 linespacing=1.4)

    # ── Main title (above column labels) ──────────────────────────────────
    fig.suptitle(
        "Attribution accuracy vs training distance from boundary",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Caption ───────────────────────────────────────────────────────────
    caption = (
        "Green = correct assumption identified.   "
        "Red = wrong or missed attribution.\n"
        "Attribution can remain correct even when boundary location is imprecise."
    )
    fig.text(
        0.595, 0.035, caption,
        ha="center", va="bottom", fontsize=9, color="#333333",
        style="italic", linespacing=1.5,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    FIG_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_PATH}")

    # Summary
    print()
    print("Attribution summary:")
    print(f"  {'':6}  {'M=1.05':>8}  {'M=10':>6}  {'M=20':>6}")
    for sc_name in SCENARIOS:
        row = []
        for M in Ms_TEST:
            rates = results[M][sc_name]
            ok    = cell_correct(rates, sc_name)
            row.append("OK" if ok else "FAIL")
        print(f"  Sc. {sc_name}:  {row[0]:>8}  {row[1]:>6}  {row[2]:>6}")
    print()
    for M in Ms_TEST:
        n_ok = sum(cell_correct(results[M][sc], sc) for sc in SCENARIOS)
        print(f"  M={M:g}: {n_ok}/3 correct")


if __name__ == "__main__":
    main()
