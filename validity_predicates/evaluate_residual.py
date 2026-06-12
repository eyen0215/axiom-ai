"""
Evaluate the residual-based validity predicate for Pilot 1 (ideal gas).

Produces:
  1. Predicted vs true log(residual) as a function of pressure (line + scatter)
  2. Extrapolation quality metrics: Pearson r and R² on held-out set
  3. Calibrated breakdown detection: recall, precision, AUROC
  4. Honesty check: prints what information was and was not given to the model

Breakdown criterion:
  "Broken" = true_residual > 0.10  (10% deviation from ideal gas PV/nRT)
  This threshold is chosen physically (10% is a reasonable engineering limit)
  and is NOT given to the model during training.

Detection threshold:
  pred_log_res > (train_mean + 3 * train_std)
  Calibrated entirely from training-set statistics. Justification: training
  data is in the valid regime (small residuals), so 3-sigma above the training
  mean identifies states the model considers anomalously far from the training
  distribution.

Usage: python validity_predicates/evaluate_residual.py
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
from validity_predicates.train_residual import make_features, make_log_targets, MODEL_PATH, TRAIN_NPZ, TEST_NPZ

FIGURES_DIR = Path("figures")
BREAKDOWN_RESIDUAL_THRESHOLD = 0.10  # "broken" if true residual > 10%
DETECTION_SIGMA = 3.0                # detection threshold: mean + k*std (training stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _auroc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true_binary.astype(bool)]
    neg = y_score[~y_true_binary.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"recall": recall, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(save_figures: bool = True) -> dict:
    # ---- Load model --------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model at {MODEL_PATH}. Run train_residual.py first."
        )
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    predicate = ResidualPredicate()
    predicate.load_state_dict(checkpoint["state_dict"])
    predicate.eval()

    train_log_res_mean = checkpoint["train_log_res_mean"]
    train_log_res_std = checkpoint["train_log_res_std"]
    detection_threshold = train_log_res_mean + DETECTION_SIGMA * train_log_res_std

    # ---- Load data ---------------------------------------------------------
    if not TRAIN_NPZ.exists() or not TEST_NPZ.exists():
        raise FileNotFoundError(
            "Data files not found. Run train_residual.py first to generate and save data."
        )
    train_data = dict(np.load(TRAIN_NPZ))
    test_data = dict(np.load(TEST_NPZ))

    X_tr = make_features(train_data)
    y_tr_true = make_log_targets(train_data)
    y_tr_pred = predicate.predict_raw(X_tr)

    X_ho = make_features(test_data)
    y_ho_true = make_log_targets(test_data)
    y_ho_pred = predicate.predict_raw(X_ho)

    # ---- Honesty check printout --------------------------------------------
    print("=" * 70)
    print("HONESTY CHECK -- what information was and was not given to the model")
    print("=" * 70)
    print("\nGiven to the model:")
    print("  Features : raw [P, V, T, n] (no pre-computed criteria)")
    print("  Target   : log(|PV/nRT - 1|) -- requires only the ideal gas equation")
    print("  Training regime: P = 1-10 atm  (valid regime, small residuals)")
    print("\nNOT given to the model:")
    print("  - Van der Waals parameters a, b")
    print("  - Breakdown threshold (10% residual)")
    print("  - Which combination of P, V, T, n determines breakdown")
    print("  - Any held-out data or its residual values")
    print()

    # ---- Training-set stats ------------------------------------------------
    true_residual_tr = train_data["residual"]
    print("-" * 70)
    print("Training-set true_residual (raw, valid regime only):")
    print(f"  min = {true_residual_tr.min():.6f}  max = {true_residual_tr.max():.6f}")
    print("Training-set log(residual) statistics (valid regime only):")
    print(f"  mean  = {train_log_res_mean:.4f}  (residual ~ {np.exp(train_log_res_mean):.5f})")
    print(f"  std   = {train_log_res_std:.4f}")
    print(f"  min   = {y_tr_true.min():.4f}  max = {y_tr_true.max():.4f}")
    print(f"\nCalibrated detection threshold  (mean + {DETECTION_SIGMA:.0f} * std):")
    print(f"  {train_log_res_mean:.4f} + {DETECTION_SIGMA:.0f} * {train_log_res_std:.4f}"
          f" = {detection_threshold:.4f}  (residual ~ {np.exp(detection_threshold):.4f})")
    print(f"\nBreakdown label threshold (physical choice, NOT given to model):")
    print(f"  true residual > {BREAKDOWN_RESIDUAL_THRESHOLD:.2f}  ->"
          f"  log(residual) > {np.log(BREAKDOWN_RESIDUAL_THRESHOLD):.3f}")
    print()

    # ---- Held-out residual range -------------------------------------------
    true_residual_ho = test_data["residual"]
    print("-" * 70)
    print("Held-out true_residual (raw, P = 50-200 atm):")
    print(f"  min = {true_residual_ho.min():.6f}  max = {true_residual_ho.max():.6f}")
    print()

    # ---- Extrapolation quality (held-out) ----------------------------------
    r = _pearson_r(y_ho_true, y_ho_pred)
    r2 = _r2(y_ho_true, y_ho_pred)
    print("-" * 70)
    print("Extrapolation quality (held-out P = 50-200 atm):")
    print(f"  Pearson r  = {r:.4f}")
    print(f"  R^2        = {r2:.4f}")
    print(f"  mean(true log-residual)      = {y_ho_true.mean():.4f}")
    print(f"  mean(predicted log-residual) = {y_ho_pred.mean():.4f}")
    print(f"  bias (pred - true)           = {(y_ho_pred - y_ho_true).mean():.4f}")
    print()

    # ---- Calibrated breakdown detection ------------------------------------
    broken_true = test_data["residual"] > BREAKDOWN_RESIDUAL_THRESHOLD
    broken_pred = y_ho_pred > detection_threshold

    m = _binary_metrics(broken_true, broken_pred)
    auroc = _auroc(broken_true.astype(int), y_ho_pred)

    print("-" * 70)
    print(f"Calibrated breakdown detection (threshold: pred > {detection_threshold:.4f}):")
    print(f"  N held-out  = {len(broken_true)}")
    print(f"  N broken    = {broken_true.sum()}  ({broken_true.mean():.1%})")
    print(f"  N detected  = {broken_pred.sum()}")
    print(f"  Recall      = {m['recall']:.4f}")
    print(f"  Precision   = {m['precision']:.4f}")
    print(f"  F1          = {m['f1']:.4f}")
    print(f"  AUROC       = {auroc:.4f}")
    print()

    # ---- Training-set fit quality (sanity check) ---------------------------
    r_tr = _pearson_r(y_tr_true, y_tr_pred)
    r2_tr = _r2(y_tr_true, y_tr_pred)
    print("-" * 70)
    print("Training-set fit quality (sanity check):")
    print(f"  Pearson r = {r_tr:.4f}  R^2 = {r2_tr:.4f}")
    print()

    # ---- Linear baseline: log(P) -> log(residual) --------------------------
    log_P_tr = np.log(train_data["P"]).astype(np.float32)
    log_P_ho = np.log(test_data["P"]).astype(np.float32)
    slope, intercept = np.polyfit(log_P_tr, y_tr_true, 1)
    baseline_tr_pred = slope * log_P_tr + intercept
    baseline_ho_pred = slope * log_P_ho + intercept

    bl_mean = float(baseline_tr_pred.mean())
    bl_std = float(baseline_tr_pred.std())
    bl_threshold = bl_mean + DETECTION_SIGMA * bl_std
    bl_auroc = _auroc(broken_true.astype(int), baseline_ho_pred)

    print("-" * 70)
    print("Linear baseline  (log P -> log residual, fit on training set only):")
    print(f"  slope = {slope:.4f}  intercept = {intercept:.4f}")
    print(f"  train pred mean = {bl_mean:.4f}  std = {bl_std:.4f}")
    print(f"  detection threshold (mean + {DETECTION_SIGMA:.0f}*std) = {bl_threshold:.4f}")
    print(f"  AUROC = {bl_auroc:.4f}")
    print()

    # ---- Figures -----------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)

    # Figure 1: P vs log(residual) -- true residuals + predicted trend
    fig1, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(train_data["P"], y_tr_true, s=8, alpha=0.4, color="steelblue",
               label="True log(residual) -- training (P = 1-10 atm)", zorder=3)
    ax.scatter(test_data["P"], y_ho_true, s=8, alpha=0.4, color="tomato",
               label="True log(residual) -- held-out (P = 50-200 atm)", zorder=3)

    ax.scatter(train_data["P"], y_tr_pred, s=8, alpha=0.35, color="navy",
               marker="x", label="Predicted -- training", zorder=4)
    ax.scatter(test_data["P"], y_ho_pred, s=8, alpha=0.35, color="darkred",
               marker="x", label="Predicted -- held-out", zorder=4)

    ax.axhline(np.log(BREAKDOWN_RESIDUAL_THRESHOLD), color="black", lw=1.5, ls="--",
               label=f"Breakdown boundary  residual = {BREAKDOWN_RESIDUAL_THRESHOLD:.0%}")
    ax.axhline(detection_threshold, color="purple", lw=1.2, ls=":",
               label=f"Detection threshold  (train mean + {DETECTION_SIGMA:.0f}*std)")
    ax.axvline(10.0, color="gray", lw=1.0, ls="--", alpha=0.7,
               label="Regime split  P = 10 atm")

    ax.set_xlabel("Pressure P (atm)")
    ax.set_ylabel("log(|PV/nRT - 1|)")
    ax.set_xscale("log")
    ax.set_title(
        "Residual-Based Predicate -- Training vs Held-out\n"
        "Features: raw [P,V,T,n]. Target: log(|PV/nRT-1|). "
        "No breakdown criterion given to model.",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig1.tight_layout()
    fig1_path = FIGURES_DIR / "pilot1_residual_pressure.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"  Figure 1 saved -> {fig1_path}")
    plt.show()

    # Figure 2: True vs predicted log(residual) scatter
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    ax2.scatter(y_tr_true, y_tr_pred, s=12, alpha=0.45, color="steelblue",
                label=f"Training  (P = 1-10 atm, N={len(y_tr_true)})")
    ax2.scatter(y_ho_true, y_ho_pred, s=12, alpha=0.45, color="tomato",
                label=f"Held-out  (P = 50-200 atm, N={len(y_ho_true)})")

    lo = min(y_tr_true.min(), y_ho_true.min(), y_tr_pred.min(), y_ho_pred.min()) - 0.3
    hi = max(y_tr_true.max(), y_ho_true.max(), y_tr_pred.max(), y_ho_pred.max()) + 0.3
    ax2.plot([lo, hi], [lo, hi], "k-", lw=1.5, label="y = x  (perfect prediction)")

    ax2.set_xlabel("True log(|PV/nRT - 1|)")
    ax2.set_ylabel("Predicted log(|PV/nRT - 1|)")
    ax2.set_title(
        f"Extrapolation Quality\n"
        f"Held-out: Pearson r = {r:.3f},  R^2 = {r2:.3f}\n"
        f"(model trained only on low-pressure data, P = 1-10 atm)",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2_path = FIGURES_DIR / "pilot1_residual_scatter.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"  Figure 2 saved -> {fig2_path}")
    plt.show()

    return {
        "r_holdout": r, "r2_holdout": r2,
        "recall": m["recall"], "precision": m["precision"],
        "f1": m["f1"], "auroc": auroc,
        "n_broken": int(broken_true.sum()), "n_holdout": len(broken_true),
        "detection_threshold": detection_threshold,
        "train_log_res_mean": train_log_res_mean,
        "train_log_res_std": train_log_res_std,
    }


if __name__ == "__main__":
    results = evaluate(save_figures=True)
    print("=" * 70)
    print("Summary:")
    print(f"  Held-out Pearson r : {results['r_holdout']:.4f}")
    print(f"  Held-out R^2       : {results['r2_holdout']:.4f}")
    print(f"  Recall             : {results['recall']:.4f}")
    print(f"  Precision          : {results['precision']:.4f}")
    print(f"  AUROC              : {results['auroc']:.4f}")
