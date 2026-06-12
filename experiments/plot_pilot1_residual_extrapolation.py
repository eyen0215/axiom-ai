"""
Headline figure: predicted vs true log(residual), training and held-out regimes.

Shows whether the model successfully extrapolated the residual trend learned
from low-pressure training data (P = 1-10 atm) into the high-pressure held-out
regime (P = 50-200 atm) it never saw during training.

Usage: python experiments/plot_pilot1_residual_extrapolation.py
  (requires train_residual.py to have been run first)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from validity_predicates.residual_predicate import ResidualPredicate
from validity_predicates.train_residual import (
    make_features, make_log_targets, MODEL_PATH, TRAIN_NPZ, TEST_NPZ
)

FIGURES_DIR = Path("figures")


def _pearson_r(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model at {MODEL_PATH}. Run validity_predicates/train_residual.py first."
        )

    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    predicate = ResidualPredicate()
    predicate.load_state_dict(checkpoint["state_dict"])
    predicate.eval()

    train_data = dict(np.load(TRAIN_NPZ))
    test_data = dict(np.load(TEST_NPZ))

    y_tr_true = make_log_targets(train_data)
    y_tr_pred = predicate.predict_raw(make_features(train_data))

    y_ho_true = make_log_targets(test_data)
    y_ho_pred = predicate.predict_raw(make_features(test_data))

    r = _pearson_r(y_ho_true, y_ho_pred)
    r2 = _r2(y_ho_true, y_ho_pred)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.scatter(y_tr_true, y_tr_pred, s=14, alpha=0.5, color="steelblue",
               label=f"Training  P = 1-10 atm  (N={len(y_tr_true)})")
    ax.scatter(y_ho_true, y_ho_pred, s=14, alpha=0.5, color="tomato",
               label=f"Held-out  P = 50-200 atm  (N={len(y_ho_true)})")

    all_vals = np.concatenate([y_tr_true, y_ho_true, y_tr_pred, y_ho_pred])
    lo = all_vals.min() - 0.4
    hi = all_vals.max() + 0.4
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.8, zorder=5,
            label="y = x  (perfect prediction)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True  log(|PV/nRT - 1|)", fontsize=12)
    ax.set_ylabel("Predicted  log(|PV/nRT - 1|)", fontsize=12)
    ax.set_title(
        "Residual Predicate -- Extrapolation from Low to High Pressure\n"
        f"Held-out:  Pearson r = {r:.3f},  R^2 = {r2:.3f}\n"
        "Inputs: raw [P, V, T, n].  No breakdown criterion or vdW parameters given.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")

    FIGURES_DIR.mkdir(exist_ok=True)
    save_path = FIGURES_DIR / "pilot1_residual_extrapolation.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
