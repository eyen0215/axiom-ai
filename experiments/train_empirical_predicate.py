"""
Train the empirical-residual validity predicate for pipe-flow assumption A2.

What the model sees
-------------------
  Features : raw [x, v, D]  (log-transformed inside forward())
  Target   : log(0.05 / pred_error)  — empirical prediction-error margin
             computed from Shah & London vs Hagen-Poiseuille, NO L_entry

What the model does NOT see
----------------------------
  - L_entry or Re at any point
  - Any formula for the validity boundary
  - Which combination of x, v, D matters

Architecture : ValidityPredicate (skip-connection MLP)
  n_features=3, log_transform_cols=[0, 1, 2]

Training config (fixed, no early stopping)
  lr=1e-3, weight_decay_mlp=5.0, weight_decay_skip=0.0
  epochs=600, batch_size=256, loss=MSE

The key diagnostic: after training, the skip connection effective weights
  effective_weight[i] = skip.weight[i] / feat_std[i]
should match the formula-based weights [+1, -1, -2] (up to an overall scale).

This is because log(0.05/pred_error) ≈ log(x) − log(v) − 2*log(D) + const
when pred_error is approximated by the large-x+ Shah & London asymptote.
Recovering [+, −, −] signs and |w_D|/|w_v| ≈ 2 means the model has
discovered the same boundary as the formula-based predicate — without
being told L_entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from validity_predicates.predicate import ValidityPredicate

DATA_PATH  = ROOT / "data" / "empirical_residual" / "train_empirical.npz"
SAVE_PATH  = ROOT / "validity_predicates" / "saved" / "empirical_pipe_A2.pt"

FEATURE_COLS   = ["x", "v", "D"]
LOG_COLS       = (0, 1, 2)
N_FEATURES     = 3
HIDDEN_DIMS    = (32, 16)

LR                  = 1e-3
WEIGHT_DECAY_MLP    = 5.0
WEIGHT_DECAY_SKIP   = 0.0
N_EPOCHS            = 600
BATCH_SIZE          = 256


def _log_transform(X: np.ndarray) -> np.ndarray:
    """Apply log to all three columns (x, v, D are all log-transformed)."""
    X = X.copy()
    for col in LOG_COLS:
        X[:, col] = np.log(np.clip(X[:, col], 1e-9, None))
    return X


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading  {DATA_PATH}")
    data = np.load(DATA_PATH)
    X_raw    = data["features"].astype(np.float32)       # (5000, 3)  [x, v, D]
    y_all    = data["log_criterion"].astype(np.float32)  # (5000,)
    err_all  = data["pred_error"].astype(np.float32)     # (5000,)

    print(f"  {len(X_raw)} training samples")
    print(f"  log_criterion range : [{y_all.min():.3f}, {y_all.max():.3f}]"
          f"  mean = {y_all.mean():.3f}")
    print(f"  pred_error range    : [{err_all.min():.4f}, {err_all.max():.4f}]"
          f"  (all < 0.05 by construction)")

    # ------------------------------------------------------------------
    # Fit normalisation on log-transformed training features
    # ------------------------------------------------------------------
    X_log     = _log_transform(X_raw)
    feat_mean = X_log.mean(axis=0).astype(np.float32)
    feat_std  = (X_log.std(axis=0) + 1e-8).astype(np.float32)

    print(f"\nFeature log-space statistics (used for normalisation):")
    for i, name in enumerate(FEATURE_COLS):
        print(f"  log({name}) : mean = {feat_mean[i]:+.3f}  std = {feat_std[i]:.3f}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    predicate = ValidityPredicate(
        hidden_dims=HIDDEN_DIMS,
        n_features=N_FEATURES,
        log_transform_cols=LOG_COLS,
        feature_cols=FEATURE_COLS,
    )
    predicate.set_normalization(feat_mean, feat_std)

    optimizer = torch.optim.Adam([
        {"params": predicate.skip.parameters(), "weight_decay": WEIGHT_DECAY_SKIP},
        {"params": predicate.mlp.parameters(),  "weight_decay": WEIGHT_DECAY_MLP},
    ], lr=LR)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_raw)
    y_t = torch.from_numpy(y_all)
    n_tr = len(X_t)

    # ------------------------------------------------------------------
    # Training loop — exactly N_EPOCHS epochs, mini-batch size BATCH_SIZE
    # ------------------------------------------------------------------
    print(f"\nTraining for {N_EPOCHS} epochs  "
          f"(batch={BATCH_SIZE}, lr={LR}, wd_mlp={WEIGHT_DECAY_MLP}) ...")
    print(f"  {'Epoch':>6}  {'train MSE':>10}")
    print(f"  {'-'*6}  {'-'*10}")

    for epoch in range(1, N_EPOCHS + 1):
        predicate.train()
        perm       = torch.randperm(n_tr)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_tr, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            x_b, y_b = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(predicate(x_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % 100 == 0:
            avg = epoch_loss / n_batches
            print(f"  {epoch:6d}  {avg:10.5f}")

    # Final full-batch loss
    predicate.eval()
    with torch.no_grad():
        final_loss = loss_fn(predicate(X_t), y_t).item()
    print(f"\nFinal full-batch MSE : {final_loss:.5f}")

    # ------------------------------------------------------------------
    # Effective skip weights
    # ------------------------------------------------------------------
    w_skip = predicate.skip.weight.data.numpy().ravel()   # shape (3,)
    b_skip = predicate.skip.bias.data.item()

    eff_w = w_skip / feat_std   # effective weight in log-feature space

    # Formula-based reference weights (from log(x / L_entry) = log(x) - log(v) - 2*log(D) + const)
    formula_w = np.array([+1.0, -1.0, -2.0])

    print()
    print("=== Skip Connection Effective Weights ===")
    print("(effective_weight[i] = skip.weight[i] / feat_std[i])")
    print()

    signs_match = True
    for i, name in enumerate(FEATURE_COLS):
        sign_ok = np.sign(eff_w[i]) == np.sign(formula_w[i])
        tag     = "SIGN MATCH" if sign_ok else "SIGN MISMATCH"
        if not sign_ok:
            signs_match = False
        print(f"  {name}: effective weight = {eff_w[i]:+.3f}  "
              f"(formula-based was {formula_w[i]:+.3f})  {tag}")

    print()
    ratio = abs(eff_w[2]) / (abs(eff_w[1]) + 1e-12)
    print(f"  |w_D| / |w_v|  = {ratio:.3f}  (formula-based: 2.000)")
    if 1.3 <= ratio <= 3.0:
        print("  D^2 RECOVERED")
    else:
        print("  D^2 ratio outside expected range [1.3, 3.0]")

    print()
    print(f"  skip bias  b = {b_skip:+.3f}")
    print(f"  raw skip weights : {w_skip}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict" : predicate.state_dict(),
        "feat_mean"  : feat_mean.tolist(),
        "feat_std"   : feat_std.tolist(),
        "feature_cols": FEATURE_COLS,
        "log_cols"   : list(LOG_COLS),
        "eff_weights": eff_w.tolist(),
        "final_mse"  : final_loss,
    }, SAVE_PATH)
    print(f"\nSaved -> {SAVE_PATH}")
    print("\nDone. Run experiments/plot_empirical_boundary.py for boundary plots.")


if __name__ == "__main__":
    main()
