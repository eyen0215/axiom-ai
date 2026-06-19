"""
Rebuild the three predicates that failed the feature-circularity audit.

Predicates rebuilt to use ONLY independent state variables:
  le_A2       [eps_eq]    (was: [eps_eq, |sigma_vm/(E*eps_eq) - 1|])
  maxwell_A1  [E_field]   (was: [|D/(epsilon*E) - 1|])
  maxwell_A2  [frequency] (was: [omega*epsilon/sigma_eff])

For each rebuilt predicate this script prints:
  1. AUROC on a continuous test sweep that crosses the physical boundary
  2. Trivial-threshold baseline AUROC (raw feature value as discriminator)
  3. WARNING if |neural - trivial| < 0.02
  4. 20-point table: feature value -> predicate score

Overwrites: le_A2.pt, maxwell_A1.pt, maxwell_A2.pt

Training data is generated inline; the existing train_*.npz files are NOT used
so that the new feature format does not break the old scenario files mid-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Linear Elasticity (steel)
LE_E           = 200e9          # Pa
LE_SIGMA_YIELD = 250e6          # Pa
LE_EPS_YIELD   = LE_SIGMA_YIELD / LE_E   # = 0.00125

# Maxwell (lossy glass dielectric)
MX_EPSILON_0   = 8.854e-12
MX_EPSILON_R   = 2.25
MX_EPSILON     = MX_EPSILON_0 * MX_EPSILON_R   # ≈ 1.992e-11 F/m
MX_SIGMA_EFF   = 1e-3           # S/m
MX_E_SAT       = 1e8            # V/m  (nonlinear onset; boundary for log_criterion)
MX_F_BOUNDARY  = 0.01 * MX_SIGMA_EFF / (2.0 * np.pi * MX_EPSILON)  # ≈ 79,900 Hz

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

LR           = 1e-3
WD_SKIP      = 0.0
WD_MLP       = 5.0
EPOCHS       = 300
BATCH_SIZE   = 256
HOLDOUT_FRAC = 0.20
FIRE_THRESH  = 0.5
N_TRAIN      = 5000
N_TEST       = 1000

SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Shared training helper
# ---------------------------------------------------------------------------

def compute_norm_stats(X_raw: np.ndarray, log_cols: tuple) -> tuple:
    X = X_raw.copy().astype(np.float64)
    for col in log_cols:
        X[:, col] = np.log(np.clip(X[:, col], 1e-12, None))
    mean = X.mean(axis=0).astype(np.float32)
    std  = (X.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def train_predicate(
    name: str,
    features_raw: np.ndarray,
    log_criterion_raw: np.ndarray,
    log_cols: tuple,
    feature_col_names: list[str],
) -> tuple[ValidityPredicate, float]:
    """Train one ValidityPredicate. Returns (predicate, shift)."""
    N       = len(features_raw)
    n_hold  = int(N * HOLDOUT_FRAC)
    n_train = N - n_hold
    n_feat  = features_raw.shape[1]

    X_train = features_raw[:n_train].astype(np.float32)
    lc_raw  = log_criterion_raw[:n_train]

    raw_mean = float(np.mean(log_criterion_raw))
    shift    = raw_mean if raw_mean > 10.0 else 0.0
    y_train  = (lc_raw - shift).astype(np.float32)

    print(f"\n--- Training {name} ---")
    print(f"  features: {feature_col_names}   log_cols={log_cols}")
    print(f"  mean log_criterion: {raw_mean:.3f}  std: {float(np.std(log_criterion_raw)):.3f}")
    if raw_mean > 10.0:
        print(f"  Re-centering applied (shift={shift:.4f})")
    else:
        print(f"  No re-centering needed")

    feat_mean, feat_std = compute_norm_stats(X_train, log_cols)

    predicate = ValidityPredicate(
        n_features=n_feat,
        log_transform_cols=log_cols,
        feature_cols=feature_col_names,
    )
    predicate.set_normalization(feat_mean, feat_std)

    optimizer = torch.optim.Adam(
        [
            {"params": list(predicate.skip.parameters()), "weight_decay": WD_SKIP},
            {"params": list(predicate.mlp.parameters()),  "weight_decay": WD_MLP},
        ],
        lr=LR,
    )
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)

    predicate.train()
    for epoch in range(EPOCHS):
        perm       = torch.randperm(n_train)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            logits = predicate(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/n_batches:.4f}")

    predicate.eval()
    w = predicate.skip.weight.detach().numpy().flatten()
    b = float(predicate.skip.bias.detach().numpy().item())
    print(f"  skip weight(s): {w}  bias: {b:.4f}")

    return predicate, shift


def score_predicate(pred: ValidityPredicate, X: np.ndarray, shift: float) -> np.ndarray:
    with torch.no_grad():
        logits = pred(torch.from_numpy(X.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-(logits + shift)))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_and_report(
    name: str,
    pred: ValidityPredicate,
    shift: float,
    features_test: np.ndarray,
    y_broken: np.ndarray,          # 1 = broken (assumption fails), 0 = valid
    raw_feature_for_trivial: np.ndarray,  # monotonically increasing with brokenness
    table_feat_vals: np.ndarray,   # 20 points for the table
    table_feat_label: str,
) -> None:
    print(f"\n{'='*62}")
    print(f"EVALUATION: {name}")
    print(f"{'='*62}")

    # Neural AUROC
    scores = score_predicate(pred, features_test, shift)
    broken_signal = 1.0 - scores   # high when broken
    neural_auroc  = float(roc_auc_score(y_broken, broken_signal))

    # Trivial-threshold baseline
    trivial_auroc = float(roc_auc_score(y_broken, raw_feature_for_trivial))

    print(f"  Neural AUROC:              {neural_auroc:.4f}")
    print(f"  Trivial-threshold AUROC:   {trivial_auroc:.4f}")
    gap = abs(neural_auroc - trivial_auroc)
    if gap < 0.02:
        print(f"  WARNING: gap = {gap:.4f} < 0.02 -- "
              f"neural model adds nothing over a simple threshold")
    else:
        print(f"  Gap: {gap:.4f} (> 0.02; neural model shows extra structure)")

    n_broken = int(np.sum(y_broken))
    print(f"  Test set: {len(y_broken)} samples  "
          f"({n_broken} broken, {len(y_broken)-n_broken} valid)")

    # 20-point table
    tab_feats = table_feat_vals[:, np.newaxis]
    tab_scores = score_predicate(pred, tab_feats, shift)

    print(f"\n  {'Feature value':>18}  {'Score (validity)':>18}  {'Fires?':>8}")
    print(f"  {'-'*18}  {'-'*18}  {'-'*8}")
    for fv, sc in zip(table_feat_vals, tab_scores):
        fires = "YES" if sc < FIRE_THRESH else "no"
        print(f"  {fv:>18.4e}  {sc:>18.4f}  {fires:>8}")


# ---------------------------------------------------------------------------
# le_A2 rebuild
# ---------------------------------------------------------------------------

def rebuild_le_a2() -> None:
    print("\n" + "#" * 62)
    print("# REBUILD: le_A2 (linearity)")
    print("# Old features: [eps_eq, |sigma_vm/(E*eps_eq) - 1|]")
    print("# New feature:  [eps_eq]  (independent state variable)")
    print("# Physical boundary: eps_yield = sigma_yield/E = 0.00125")
    print("#" * 62)

    eps_yield = LE_EPS_YIELD   # 0.00125

    # -- Training data: eps_eq uniform in [0.0001, 0.001] (sub-yield) --
    eps_eq_tr = rng.uniform(0.0001, 0.001, N_TRAIN)
    lc_tr     = np.clip(np.log(eps_yield / eps_eq_tr), -20.0, 20.0)
    X_tr      = eps_eq_tr[:, np.newaxis].astype(np.float32)
    log_cols  = ()   # linear scale; 10x range is fine without log

    pred_a2, shift = train_predicate(
        name="le_A2",
        features_raw=X_tr,
        log_criterion_raw=lc_tr,
        log_cols=log_cols,
        feature_col_names=["eps_eq"],
    )

    # -- Save --
    out_path = SAVE_DIR / "le_A2.pt"
    if shift == 0.0:
        torch.save(pred_a2.state_dict(), out_path)
    else:
        torch.save({"model": pred_a2.state_dict(), "shift": shift}, out_path)
    print(f"\n  Saved -> {out_path}")

    # -- Test sweep: log-uniform [0.0005, 0.01], crosses eps_yield = 0.00125 --
    eps_eq_te = np.exp(rng.uniform(np.log(0.0005), np.log(0.01), N_TEST))
    y_broken  = (eps_eq_te >= eps_yield).astype(int)
    X_te      = eps_eq_te[:, np.newaxis].astype(np.float32)

    table_pts = np.exp(np.linspace(np.log(0.0005), np.log(0.01), 20))

    evaluate_and_report(
        name="le_A2 (linearity)  [feature: eps_eq, boundary: eps_yield = 0.00125]",
        pred=pred_a2,
        shift=shift,
        features_test=X_te,
        y_broken=y_broken,
        raw_feature_for_trivial=eps_eq_te,
        table_feat_vals=table_pts,
        table_feat_label="eps_eq",
    )


# ---------------------------------------------------------------------------
# maxwell_A1 rebuild
# ---------------------------------------------------------------------------

def rebuild_maxwell_a1() -> None:
    print("\n" + "#" * 62)
    print("# REBUILD: maxwell_A1 (linear_media)")
    print("# Old feature:  [|D/(epsilon*E) - 1|]  (theory residual)")
    print("# New feature:  [E_field]               (independent state variable)")
    print("# Physical boundary: E_sat = 1e8 V/m")
    print("#" * 62)

    E_sat = MX_E_SAT   # 1e8

    # -- Training data: E_field log-uniform in [1e2, 1e7] (valid regime) --
    E_tr = np.exp(rng.uniform(np.log(1e2), np.log(1e7), N_TRAIN))
    lc_tr = np.clip(np.log(E_sat / E_tr), -20.0, 20.0)
    X_tr  = E_tr[:, np.newaxis].astype(np.float32)
    log_cols = (0,)   # E_field spans 5 orders of magnitude

    pred_a1, shift = train_predicate(
        name="maxwell_A1",
        features_raw=X_tr,
        log_criterion_raw=lc_tr,
        log_cols=log_cols,
        feature_col_names=["E_field"],
    )

    # -- Save --
    out_path = SAVE_DIR / "maxwell_A1.pt"
    if shift == 0.0:
        torch.save(pred_a1.state_dict(), out_path)
    else:
        torch.save({"model": pred_a1.state_dict(), "shift": shift}, out_path)
    print(f"\n  Saved -> {out_path}")

    # -- Test sweep: log-uniform [1e6, 5e9], crosses E_sat = 1e8 --
    E_te     = np.exp(rng.uniform(np.log(1e6), np.log(5e9), N_TEST))
    y_broken = (E_te >= E_sat).astype(int)
    X_te     = E_te[:, np.newaxis].astype(np.float32)

    table_pts = np.exp(np.linspace(np.log(1e6), np.log(5e9), 20))

    evaluate_and_report(
        name="maxwell_A1 (linear_media)  [feature: E_field, boundary: E_sat = 1e8 V/m]",
        pred=pred_a1,
        shift=shift,
        features_test=X_te,
        y_broken=y_broken,
        raw_feature_for_trivial=E_te,
        table_feat_vals=table_pts,
        table_feat_label="E_field (V/m)",
    )


# ---------------------------------------------------------------------------
# maxwell_A2 rebuild
# ---------------------------------------------------------------------------

def rebuild_maxwell_a2() -> None:
    print("\n" + "#" * 62)
    print("# REBUILD: maxwell_A2 (quasi_static)")
    print("# Old feature:  [omega*epsilon/sigma_eff]  (criterion ratio)")
    print("# New feature:  [frequency]                (independent state variable)")
    print(f"# Physical boundary: f_boundary = {MX_F_BOUNDARY:.1f} Hz (~79.9 kHz)")
    print("#" * 62)

    f_bnd = MX_F_BOUNDARY

    # -- Training data: frequency log-uniform in [1, 1e4] Hz (sub-threshold) --
    f_tr  = np.exp(rng.uniform(np.log(1.0), np.log(1e4), N_TRAIN))
    lc_tr = np.clip(np.log(f_bnd / f_tr), -20.0, 20.0)
    X_tr  = f_tr[:, np.newaxis].astype(np.float32)
    log_cols = (0,)   # frequency spans orders of magnitude

    pred_a2, shift = train_predicate(
        name="maxwell_A2",
        features_raw=X_tr,
        log_criterion_raw=lc_tr,
        log_cols=log_cols,
        feature_col_names=["frequency"],
    )

    # -- Save --
    out_path = SAVE_DIR / "maxwell_A2.pt"
    if shift == 0.0:
        torch.save(pred_a2.state_dict(), out_path)
    else:
        torch.save({"model": pred_a2.state_dict(), "shift": shift}, out_path)
    print(f"\n  Saved -> {out_path}")

    # -- Test sweep: log-uniform [1e4, 1e8] Hz, crosses f_boundary --
    f_te     = np.exp(rng.uniform(np.log(1e4), np.log(1e8), N_TEST))
    y_broken = (f_te >= f_bnd).astype(int)
    X_te     = f_te[:, np.newaxis].astype(np.float32)

    table_pts = np.exp(np.linspace(np.log(1e4), np.log(1e8), 20))

    evaluate_and_report(
        name=f"maxwell_A2 (quasi_static)  [feature: frequency, boundary: {f_bnd:.0f} Hz]",
        pred=pred_a2,
        shift=shift,
        features_test=X_te,
        y_broken=y_broken,
        raw_feature_for_trivial=f_te,
        table_feat_vals=table_pts,
        table_feat_label="frequency (Hz)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("PREDICATE REBUILD — independent features only")
    print("=" * 62)
    print(f"\nLE  eps_yield    = {LE_EPS_YIELD:.5f}")
    print(f"MW  E_sat        = {MX_E_SAT:.2e} V/m")
    print(f"MW  f_boundary   = {MX_F_BOUNDARY:.1f} Hz")

    rebuild_le_a2()
    rebuild_maxwell_a1()
    rebuild_maxwell_a2()

    print("\n" + "=" * 62)
    print("All three predicates rebuilt and saved.")
    print("  le_A2.pt       -- [eps_eq], log_cols=()")
    print("  maxwell_A1.pt  -- [E_field], log_cols=(0,)")
    print("  maxwell_A2.pt  -- [frequency], log_cols=(0,)")
    print("=" * 62)


if __name__ == "__main__":
    main()
