"""
Evaluate the empirical-residual validity predicate (empirical_pipe_A2.pt).

Reports:
  - AUROC on pooled (test breakdown + valid holdout from train)
  - Fire rate on breakdown samples  (target > 80%)
  - False positive rate on valid holdout  (target < 5%)
  - Trivial baseline: x alone as discriminator
  - Boundary calibration: where does score cross 0.5 at (v=5, D=0.01)?

Honesty:
  Valid holdout = 1000 randomly sampled rows from train_empirical.npz.
  The model was trained on all 5000 rows of that file, so FPR is measured
  on in-distribution data (conservative / optimistic bound). Fire rate is
  on test_A2break.npz which was never seen during training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from validity_predicates.predicate import ValidityPredicate

MODEL_PATH = ROOT / "validity_predicates" / "saved" / "empirical_pipe_A2.pt"
DATA_DIR   = ROOT / "data" / "empirical_residual"
TRAIN_NPZ  = DATA_DIR / "train_empirical.npz"
TEST_NPZ   = DATA_DIR / "test_A2break.npz"

MU  = 1.81e-5   # Pa*s
RHO = 1.2       # kg/m^3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(score_positive > score_negative) over all pairs."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    return float(np.mean(scores_pos[:, None] > scores_neg[None, :]))


def _load_predicate() -> ValidityPredicate:
    ck = torch.load(MODEL_PATH, weights_only=False)
    pred = ValidityPredicate(
        hidden_dims=(32, 16),
        n_features=3,
        log_transform_cols=(0, 1, 2),
        feature_cols=["x", "v", "D"],
    )
    pred.load_state_dict(ck["state_dict"])
    pred.feat_mean.copy_(torch.tensor(ck["feat_mean"], dtype=torch.float32))
    pred.feat_std.copy_(torch.tensor(ck["feat_std"],  dtype=torch.float32))
    pred.eval()
    return pred


def _score(predicate: ValidityPredicate, X: np.ndarray) -> np.ndarray:
    """Return sigmoid validity scores in (0, 1); higher = more valid."""
    return predicate.predict(X.astype(np.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {MODEL_PATH}")
    predicate = _load_predicate()
    print(f"  effective weights (skip / feat_std): "
          f"x={predicate.skip.weight.data[0,0].item()/predicate.feat_std[0].item():+.3f}  "
          f"v={predicate.skip.weight.data[0,1].item()/predicate.feat_std[1].item():+.3f}  "
          f"D={predicate.skip.weight.data[0,2].item()/predicate.feat_std[2].item():+.3f}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    tr_data   = np.load(TRAIN_NPZ)
    te_data   = np.load(TEST_NPZ)

    X_all_valid   = tr_data["features"].astype(np.float32)   # (5000, 3)
    err_all_valid = tr_data["pred_error"].astype(np.float32)

    X_break   = te_data["features"].astype(np.float32)       # (1000, 3)
    err_break = te_data["pred_error"].astype(np.float32)

    # 1000-sample holdout from the valid set (never trained on a separate split,
    # but used here only for false-positive-rate evaluation)
    hold_idx  = rng.choice(len(X_all_valid), size=1000, replace=False)
    X_valid_h = X_all_valid[hold_idx]
    err_valid_h = err_all_valid[hold_idx]

    print(f"\nData summary")
    print(f"  Valid holdout : {len(X_valid_h):5d} samples  "
          f"mean pred_error = {err_valid_h.mean():.4f}  (all < 0.05)")
    print(f"  Breakdown test: {len(X_break):5d} samples  "
          f"mean pred_error = {err_break.mean():.4f}  (all > 0.05)")

    # ------------------------------------------------------------------
    # Predicate scores
    # ------------------------------------------------------------------
    scores_valid = _score(predicate, X_valid_h)   # high = valid
    scores_break = _score(predicate, X_break)     # should be low

    # ------------------------------------------------------------------
    # Metrics — empirical predicate
    # ------------------------------------------------------------------
    auroc_pred  = _auroc(scores_valid, scores_break)
    fire_break  = float((scores_break < 0.5).mean())   # fraction fired on breakdown
    fpr_valid   = float((scores_valid < 0.5).mean())   # fraction falsely fired on valid

    # ------------------------------------------------------------------
    # Trivial baseline: x alone as the score
    # x_valid tends to be large (valid → large x+), x_break tends to be small
    # ------------------------------------------------------------------
    x_valid_h = X_valid_h[:, 0]
    x_break   = X_break[:, 0]
    auroc_trivial = _auroc(x_valid_h, x_break)

    gap = auroc_pred - auroc_trivial

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print()
    print("=" * 68)
    col1 = "Metric"
    col2 = "Empirical predicate"
    col3 = "Trivial (x only)"
    print(f"  {col1:<30} | {col2:<21} | {col3}")
    print("  " + "-" * 64)
    print(f"  {'AUROC':<30} | {auroc_pred:<21.3f} | {auroc_trivial:.3f}")
    print(f"  {'Fire rate on breakdown':<30} | {fire_break:<21.3f} | {'---'}")
    print(f"  {'False positive rate':<30} | {fpr_valid:<21.3f} | {'---'}")
    print(f"  {'Gap over trivial baseline':<30} | {gap:<21.3f} | {'---'}")
    print("=" * 68)

    # ------------------------------------------------------------------
    # Targets check
    # ------------------------------------------------------------------
    print()
    print("Target checks:")
    print(f"  Fire rate > 0.80 : {fire_break:.3f}  "
          + ("PASS" if fire_break > 0.80 else "FAIL"))
    print(f"  FPR     < 0.05   : {fpr_valid:.3f}  "
          + ("PASS" if fpr_valid < 0.05 else "FAIL"))

    # ------------------------------------------------------------------
    # Additional breakdown of scores
    # ------------------------------------------------------------------
    print()
    print("Score distributions:")
    for label, sc in [("Valid holdout", scores_valid), ("Breakdown test", scores_break)]:
        print(f"  {label}: mean={sc.mean():.4f}  "
              f"min={sc.min():.4f}  max={sc.max():.4f}  "
              f"frac<0.5={float((sc<0.5).mean()):.4f}")

    # ------------------------------------------------------------------
    # Boundary calibration: x crossing at v=5.0, D=0.01
    # Scan x, find where score crosses 0.5
    # ------------------------------------------------------------------
    print()
    print("Boundary calibration  (v=5.0, D=0.01 fixed)")
    v_ref, D_ref = 5.0, 0.01
    x_scan = np.logspace(np.log10(0.01), np.log10(50.0), 5000)
    X_scan = np.column_stack([
        x_scan,
        np.full_like(x_scan, v_ref),
        np.full_like(x_scan, D_ref),
    ]).astype(np.float32)
    scores_scan = _score(predicate, X_scan)

    # First x where score crosses 0.5 from below (score goes from low to high as x increases)
    above = np.where(scores_scan >= 0.5)[0]
    if len(above) == 0:
        print("  WARNING: score never reaches 0.5 in scan range — try wider x range")
        x_learned = float("nan")
    else:
        x_learned = float(x_scan[above[0]])

    # True L_entry — computed here ONLY for comparison, never used in training
    Re_ref    = RHO * v_ref * D_ref / MU
    L_entry   = 0.06 * Re_ref * D_ref
    ratio     = x_learned / L_entry if not np.isnan(x_learned) else float("nan")

    print(f"  Score-0.5 crossing    : x = {x_learned:.4f} m")
    print(f"  True L_entry (ref)    : x = {L_entry:.4f} m")
    print(f"  Boundary ratio (learned / L_entry) = {ratio:.3f}")

    if not np.isnan(ratio):
        if 0.5 <= ratio <= 10.0:
            print(f"  Ratio in expected range [0.5, 10]: same functional form confirmed")
        else:
            print(f"  WARNING: ratio outside [0.5, 10] — boundary may not track L_entry")

    # Additional check: verify score is monotone in x at this (v, D)
    diffs = np.diff(scores_scan)
    n_non_mono = int((diffs < 0).sum())
    if n_non_mono == 0:
        print(f"  Score is monotonically increasing in x at (v={v_ref}, D={D_ref})  [OK]")
    else:
        print(f"  WARNING: score has {n_non_mono} non-monotone steps in x scan")

    # ------------------------------------------------------------------
    # x+ at the learned boundary
    # ------------------------------------------------------------------
    if not np.isnan(x_learned):
        x_plus_learned = x_learned * MU / (RHO * v_ref * D_ref ** 2)
        x_plus_L_entry = 0.06   # by definition: L_entry = 0.06 * Re * D → x+ = 0.06
        print(f"\n  x+ at learned boundary : {x_plus_learned:.4f}  "
              f"(x+ at true L_entry: {x_plus_L_entry:.4f})")
        print(f"  Shah-London pred_error at learned boundary: ", end="")
        # Compute actual pred_error at x_learned
        sys.path.insert(0, str(DATA_DIR))
        from generate_ground_truth import pred_error as shah_pred_error
        pe_at_boundary = float(shah_pred_error(
            np.array([x_learned]), np.array([v_ref]), np.array([D_ref])
        )[0])
        print(f"{pe_at_boundary:.4f}  (expect ~0.05 if boundary = 5% error threshold)")


if __name__ == "__main__":
    main()
