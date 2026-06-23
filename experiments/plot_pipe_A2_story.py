"""
1D story plot for the A2 fully_developed predicate.

Collapses the 5D feature space to 1D via the normalised ratio  x / L_entry(v,D,rho,mu).
This single axis reveals training regime, physical boundary, and test detection
in one clean picture.
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO         = 1.2
MU          = 1.81e-5
D_SWEEP     = 0.01
V_SWEEP     = 5.0
ENTRY_COEFF = 0.06

FIG_PATH = Path("figures/pipe_A2_story.png")
DATA_DIR = Path("data/pipe_flow")
SAVE_DIR = Path("validity_predicates/saved")


def compute_L_entry(v: np.ndarray, D: np.ndarray,
                    rho: np.ndarray, mu: np.ndarray) -> np.ndarray:
    Re = rho * v * D / mu
    return ENTRY_COEFF * Re * D


def load_model() -> ValidityPredicate:
    ckpt  = torch.load(SAVE_DIR / "pipe_A2.pt", weights_only=False)
    model = ValidityPredicate(n_features=5, log_transform_cols=(0, 1, 2, 3, 4))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    model = load_model()

    # -----------------------------------------------------------------------
    # 1. Training data
    # -----------------------------------------------------------------------
    d_tr     = np.load(DATA_DIR / "train_A2.npz")
    tr_feats = d_tr["features"].astype(np.float32)   # (5000, 5)
    tr_L     = compute_L_entry(tr_feats[:, 1], tr_feats[:, 2],
                               tr_feats[:, 3], tr_feats[:, 4])
    tr_ratio  = tr_feats[:, 0] / tr_L                # x / L_entry
    tr_scores = model.predict(tr_feats)

    # -----------------------------------------------------------------------
    # 2. Test data (A2break: x < L_entry, all ratios < 1)
    # -----------------------------------------------------------------------
    d_te     = np.load(DATA_DIR / "test_scenario_A2break.npz")
    te_feats = d_te["A2_features"].astype(np.float32)  # (1000, 5)
    te_L     = compute_L_entry(te_feats[:, 1], te_feats[:, 2],
                               te_feats[:, 3], te_feats[:, 4])
    te_ratio  = te_feats[:, 0] / te_L
    te_scores = model.predict(te_feats)

    # -----------------------------------------------------------------------
    # 3. Dense sweep: vary x/L_entry from 0.1 to 3.0 at fixed (v,D,rho,mu)
    # -----------------------------------------------------------------------
    ratios   = np.linspace(0.1, 3.0, 500)
    L_fixed  = compute_L_entry(
        np.array([V_SWEEP]), np.array([D_SWEEP]),
        np.array([RHO]),     np.array([MU]),
    )[0]
    x_sweep  = ratios * L_fixed
    sw_feats = np.column_stack([
        x_sweep,
        np.full(500, V_SWEEP),
        np.full(500, D_SWEEP),
        np.full(500, RHO),
        np.full(500, MU),
    ]).astype(np.float32)
    sw_scores = model.predict(sw_feats)

    # Find where the sweep score crosses 0.5 (first crossing from above)
    above = sw_scores >= 0.5
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]
    if len(crossings) > 0:
        i = crossings[0]
        # linear interpolation between adjacent points
        r0, r1 = ratios[i], ratios[i + 1]
        s0, s1 = sw_scores[i], sw_scores[i + 1]
        cross_ratio = r0 + (0.5 - s0) * (r1 - r0) / (s1 - s0)
    else:
        cross_ratio = float(ratios[np.argmin(np.abs(sw_scores - 0.5))])

    print(f"Learned threshold (score = 0.5 crossing): x/L_entry = {cross_ratio:.4f}")
    print(f"True boundary:                             x/L_entry = 1.0000")
    print(f"Offset from true boundary:                 {cross_ratio - 1.0:+.4f}  "
          f"({(cross_ratio - 1.0) * 100:+.2f}%)")
    print()
    print(f"Training data   — ratio range: [{tr_ratio.min():.3f}, {tr_ratio.max():.3f}]")
    print(f"Test data       — ratio range: [{te_ratio.min():.3f}, {te_ratio.max():.3f}]")
    print(f"  test fire rate (score<0.5): {(te_scores < 0.5).mean()*100:.1f}%")

    # -----------------------------------------------------------------------
    # 4. Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    # Background shading
    ax.axvspan(0.1,  1.0,  alpha=0.12, color="red",   zorder=0,
               label="violation region  (x < L_entry)")
    ax.axvspan(1.05, 3.0,  alpha=0.10, color="green",  zorder=0,
               label="training region  (x ≥ 1.05·L_entry)")

    # Reference verticals
    ax.axvline(1.0,  color="black",  ls="--", lw=1.8, zorder=2,
               label="true boundary  (x / L_entry = 1.0)")
    ax.axvline(1.05, color="gray",   ls="--", lw=1.1, zorder=2,
               label="training cutoff  (1.05×)")

    # Decision threshold
    ax.axhline(0.5, color="darkorange", ls="--", lw=1.2, alpha=0.85, zorder=2,
               label="decision threshold  (score = 0.5)")

    # Dense sweep score curve
    ax.plot(ratios, sw_scores, color="steelblue", lw=2.4, zorder=4,
            label=f"predicate score  (v={V_SWEEP}, D={D_SWEEP})")

    # Learned threshold annotation
    ax.axvline(cross_ratio, color="steelblue", ls=":", lw=1.6, zorder=5)
    ax.annotate(
        f"learned threshold\n({cross_ratio:.3f}×)",
        xy=(cross_ratio, 0.5),
        xytext=(cross_ratio + 0.12, 0.30),
        fontsize=8.5, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.9),
    )

    # Training scatter (subsample 200, only within x-axis range)
    rng    = np.random.default_rng(42)
    in_view = (tr_ratio >= 0.1) & (tr_ratio <= 3.0)
    pool    = np.where(in_view)[0]
    if len(pool) > 0:
        idx = rng.choice(pool, size=min(200, len(pool)), replace=False)
        ax.scatter(tr_ratio[idx], tr_scores[idx],
                   c="green", s=10, alpha=0.45, zorder=3,
                   label=f"training samples (n≤200, shown in-range)")
    else:
        ax.text(0.98, 0.95, "no training points in x-range",
                transform=ax.transAxes, ha="right", fontsize=8, color="green")

    # Test scatter (subsample 200)
    te_idx = rng.choice(len(te_ratio), size=min(200, len(te_ratio)), replace=False)
    ax.scatter(te_ratio[te_idx], te_scores[te_idx],
               c="red", s=10, alpha=0.55, zorder=3,
               label=f"A2break test samples (n=200)")

    # Axes & labels
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("x / L_entry   (1.0 = true physical boundary)", fontsize=12)
    ax.set_ylabel("Predicate score   (< 0.5 = FIRES)", fontsize=12)
    ax.set_title(
        "A2 fully developed flow — training regime, boundary, and test detection",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.88, ncol=2)
    ax.grid(True, alpha=0.20)

    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIG_PATH}")


if __name__ == "__main__":
    main()
