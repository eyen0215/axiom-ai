"""
Train the residual-based validity predicate for Pilot 1 (ideal gas).

What is given to the model:
  Features : raw [P, V, T, n] observables, log-transformed inside the model
  Target   : log(|PV/nRT - 1|) — ideal-gas theory residual
  Loss     : MSE regression

What is NOT given to the model:
  - Van der Waals parameters (a, b)
  - Any breakdown criterion or threshold
  - Which combination of P, V, T, n matters

Training regime  : P = 1-10 atm  (van der Waals CO2 data, residuals ~0.3-3%)
Held-out regime  : P = 50-200 atm (residuals ~5-40%, breakdown boundary ~90 atm)

Usage: python validity_predicates/train_residual.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate import generate_ideal_gas_residual_data
from validity_predicates.residual_predicate import ResidualPredicate, RESIDUAL_LOG_COLS

SAVE_DIR = Path(__file__).parent / "saved_models"
TRAIN_NPZ = Path("data/ideal_gas_residual_train.npz")
TEST_NPZ = Path("data/ideal_gas_residual_test.npz")
MODEL_PATH = SAVE_DIR / "residual_pilot1.pt"


def make_features(data: dict) -> np.ndarray:
    return np.column_stack([data["P"], data["V"], data["T"], data["n"]]).astype(np.float32)


def make_log_targets(data: dict) -> np.ndarray:
    residual = np.clip(data["residual"], 1e-12, None)
    return np.log(residual).astype(np.float32)


def train_residual_predicate(
    train_data: dict,
    *,
    hidden_dims: tuple = (32, 16),
    lr: float = 1e-2,
    n_epochs: int = 800,
    val_frac: float = 0.15,
    patience: int = 60,
    seed: int = 0,
    verbose: bool = True,
) -> ResidualPredicate:
    X_raw = make_features(train_data)
    y_all = make_log_targets(train_data)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_raw))
    n_val = max(1, int(val_frac * len(X_raw)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_raw, y_tr = X_raw[tr_idx], y_all[tr_idx]
    X_val_raw, y_val = X_raw[val_idx], y_all[val_idx]

    # Normalise on log-transformed training features
    X_tr_log = X_tr_raw.copy()
    for col in RESIDUAL_LOG_COLS:
        X_tr_log[:, col] = np.log(np.clip(X_tr_raw[:, col], 1e-9, None))
    feat_mean = X_tr_log.mean(axis=0).astype(np.float32)
    feat_std = (X_tr_log.std(axis=0) + 1e-8).astype(np.float32)

    predicate = ResidualPredicate(hidden_dims=hidden_dims)
    predicate.set_normalization(feat_mean, feat_std)

    # skip: weight_decay=0 (free to learn linear trend)
    # MLP:  weight_decay=5 (kept near zero OOD so skip dominates extrapolation)
    optimizer = torch.optim.Adam([
        {"params": predicate.skip.parameters(), "weight_decay": 0.0},
        {"params": predicate.mlp.parameters(), "weight_decay": 5.0},
    ], lr=lr)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.from_numpy(X_tr_raw)
    y_tr_t = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val_raw)
    y_val_t = torch.from_numpy(y_val)

    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        predicate.train()
        optimizer.zero_grad()
        loss = loss_fn(predicate(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

        predicate.eval()
        with torch.no_grad():
            val_loss = loss_fn(predicate(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_sd = copy.deepcopy(predicate.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}  best val MSE = {best_val:.5f}")
                break

        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}  train={loss.item():.5f}  val={val_loss:.5f}")

    if best_sd is not None:
        predicate.load_state_dict(best_sd)

    return predicate


def main():
    print("=== Residual-Based Predicate Training -- Pilot 1 (Ideal Gas) ===\n")
    print("What the model sees:")
    print("  Features : raw [P, V, T, n] (log-transformed inside forward())")
    print("  Target   : log(|PV/nRT - 1|)")
    print("  NOT given: van der Waals a, b; breakdown criterion; threshold\n")

    print("Generating training data  (P = 1-10 atm, N=2000, CO2 vdW EOS) ...")
    train_data = generate_ideal_gas_residual_data(n_samples=2000, P_range=(1, 10), seed=0)
    n_tr = len(train_data["P"])
    res_tr = train_data["residual"]
    print(f"  {n_tr} samples  |  residual range: {res_tr.min():.5f} - {res_tr.max():.4f}"
          f"  |  mean: {res_tr.mean():.5f}")

    print("Generating held-out data  (P = 50-200 atm, N=1000) ...")
    test_data = generate_ideal_gas_residual_data(n_samples=1000, P_range=(50, 200), seed=1)
    n_ho = len(test_data["P"])
    res_ho = test_data["residual"]
    print(f"  {n_ho} samples  |  residual range: {res_ho.min():.4f} - {res_ho.max():.4f}"
          f"  |  mean: {res_ho.mean():.4f}")

    Path("data").mkdir(exist_ok=True)
    np.savez(TRAIN_NPZ, **train_data)
    np.savez(TEST_NPZ, **test_data)
    print(f"\n  Saved: {TRAIN_NPZ}")
    print(f"  Saved: {TEST_NPZ}")

    y_tr_all = make_log_targets(train_data)
    train_log_res_mean = float(y_tr_all.mean())
    train_log_res_std = float(y_tr_all.std())
    print(f"\nTraining log(residual) stats:")
    print(f"  mean = {train_log_res_mean:.4f}  ({train_log_res_mean:.2f})")
    print(f"  std  = {train_log_res_std:.4f}")
    print(f"  (All training states are in the valid-regime by construction.)")

    print("\nTraining residual predicate ...")
    predicate = train_residual_predicate(train_data, verbose=True)

    print("\nSkip connection weights (log P, log V, log T, log n):")
    w = predicate.skip.weight.data.numpy().ravel()
    b = predicate.skip.bias.data.item()
    print(f"  w = {w}  b = {b:.4f}")
    print("  (Positive weight on log P / negative on log V -> higher P predicts "
          "higher residual -- correct physical direction.)")

    SAVE_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": predicate.state_dict(),
        "train_log_res_mean": train_log_res_mean,
        "train_log_res_std": train_log_res_std,
        "feat_mean": predicate.feat_mean.numpy().tolist(),
        "feat_std": predicate.feat_std.numpy().tolist(),
    }, MODEL_PATH)
    print(f"\n  Saved model -> {MODEL_PATH}")
    print("\nDone. Run validity_predicates/evaluate_residual.py for metrics and plots.")


if __name__ == "__main__":
    main()
