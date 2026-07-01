"""
Train a ResidualPredicate on PR-EOS ground truth data and evaluate on two held-out sets.

Architecture: same skip+MLP as validity_predicates/residual_predicate.py.
Inputs  : raw [P, V, T, n], log-transformed inside forward()
Target  : log(r_vdw), where r_vdw = |(P + a*(n/V)^2)*(V/n - b)/(R*T) - 1|
No sigmoid. Calibration threshold = training mean + 3*std of log(r_vdw).
Breakdown criterion: r_vdw > 0.05 (5% deviation from vdW EOS).

Data (Peng-Robinson CO2 ground truth):
  Training   : T=350-500 K, P=1-50 atm   (gas, vdW accurate)
  Held-out 1 : T=290-310 K, P=60-80 atm  (near-critical)
  Held-out 2 : T=350 K,     P=100-200 atm (high-density gas)

Usage:
    python experiments/train_vdw_predicate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_vdw_residual import generate_vdw_residual_data
from validity_predicates.train_residual import train_residual_predicate

BROKEN_THRESHOLD = 0.05   # r_vdw > 0.05 -> vdW breakdown
N_TRAIN = 2000
N_HO    = 500


def make_features(data: dict) -> np.ndarray:
    return np.column_stack([data["P"], data["V"], data["T"], data["n"]]).astype(np.float32)


def eval_held_out(predicate, data: dict, train_log_mean: float, train_log_std: float,
                  label: str) -> None:
    X      = make_features(data)
    y_true = np.log(np.clip(data["r_vdw"], 1e-12, None))
    y_pred = predicate.predict_raw(X)

    r, _   = pearsonr(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2     = 1.0 - ss_res / ss_tot

    broken = (data["r_vdw"] > BROKEN_THRESHOLD).astype(int)
    n_broken = int(broken.sum())
    if 0 < n_broken < len(broken):
        auroc = roc_auc_score(broken, y_pred)
    else:
        auroc = float("nan")

    det_threshold = train_log_mean + 3 * train_log_std
    n_detected = int((y_pred > det_threshold).sum())

    print(f"\n{label}")
    print(f"  N = {len(y_true)},  broken (r_vdw > {BROKEN_THRESHOLD}) = {n_broken} ({n_broken/len(y_true):.1%})")
    print(f"  Detection threshold (train mean+3std) = {det_threshold:.4f}")
    print(f"  N detected above threshold = {n_detected}")
    print(f"  Pearson r = {r:+.4f}")
    print(f"  R²        = {r2:+.4f}")
    print(f"  AUROC     = {auroc:.4f}")


def main():
    print("=== vdW Residual Predicate (PR-EOS ground truth, CO2) ===\n")

    print("Generating data ...")
    train = generate_vdw_residual_data(N_TRAIN, P_range=(1,   50),  T_range=(350, 500), seed=0)
    ho1   = generate_vdw_residual_data(N_HO,   P_range=(60,  80),  T_range=(290, 310), seed=1)
    ho2   = generate_vdw_residual_data(N_HO,   P_range=(100, 200), T_range=(350, 350), seed=2)

    r_tr = train["r_vdw"]
    print(f"  Training N={len(r_tr)},  r_vdw [{r_tr.min():.5f}, {r_tr.max():.5f}]")
    r1 = ho1["r_vdw"]
    print(f"  HO-1     N={len(r1)},  r_vdw [{r1.min():.5f}, {r1.max():.5f}]")
    r2 = ho2["r_vdw"]
    print(f"  HO-2     N={len(r2)},  r_vdw [{r2.min():.5f}, {r2.max():.5f}]")

    # train_residual_predicate reads data["residual"] for the log target;
    # alias r_vdw so it works without modifying the existing function
    train_for_fit = {**train, "residual": train["r_vdw"]}

    log_r_tr = np.log(np.clip(r_tr, 1e-12, None))
    train_log_mean = float(log_r_tr.mean())
    train_log_std  = float(log_r_tr.std())
    print(f"\nTraining log(r_vdw): mean={train_log_mean:.4f}  std={train_log_std:.4f}")
    print(f"Detection threshold (mean + 3*std) = {train_log_mean + 3*train_log_std:.4f}")
    print(f"  log({BROKEN_THRESHOLD}) = {np.log(BROKEN_THRESHOLD):.4f}  (breakdown boundary)\n")

    print("Training ...")
    predicate = train_residual_predicate(train_for_fit, verbose=True)

    w = predicate.skip.weight.data.numpy().ravel()
    b_val = predicate.skip.bias.data.item()
    print(f"\nSkip weights [log P, log V, log T, log n]: {w.round(4)}  bias={b_val:.4f}")

    print("\n--- Evaluation ---")
    eval_held_out(predicate, ho1, train_log_mean, train_log_std,
                  "Held-out 1: near-critical (T=290-310 K, P=60-80 atm)")
    eval_held_out(predicate, ho2, train_log_mean, train_log_std,
                  "Held-out 2: high-density gas (T=350 K, P=100-200 atm)")


if __name__ == "__main__":
    main()
