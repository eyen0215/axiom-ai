"""
Generate training and evaluation data for the empirical-residual predicate.

Imports pred_error from generate_ground_truth — does NOT reimplement it.

Files produced
--------------
train_empirical.npz   5 000 valid-regime samples (pred_error < 0.05)
test_A2break.npz      1 000 breakdown samples    (pred_error > 0.05)
grid_boundary.npz     2 500 grid points at D=0.01 for boundary inspection

Feature vector: [x, v, D]
Label (train): log_criterion = log(0.05 / (pred_error + 1e-10)), clipped [-20, 20]
"""

import os
import sys

import numpy as np

# Locate generate_ground_truth in the same directory as this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ground_truth import pred_error   # Shah & London — no L_entry

OUTDIR   = os.path.dirname(os.path.abspath(__file__))
RNG      = np.random.default_rng(42)
EPSILON  = 0.05
CLIP_LO, CLIP_HI = -20.0, 20.0

# Parameter ranges
V_LO,  V_HI  = 0.5,   15.0
D_LO,  D_HI  = 0.005, 0.05
X_LO,  X_HI  = 0.01,  5.0

BATCH = 100_000   # candidates per rejection-sampling iteration


def _log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def _sample(n_target, valid: bool):
    """
    Draw n_target samples that satisfy the regime criterion.
    valid=True  → pred_error < EPSILON  (training data)
    valid=False → pred_error > EPSILON  (breakdown data)
    """
    xs, vs, Ds, errs = [], [], [], []
    collected = 0

    while collected < n_target:
        x = _log_uniform(RNG, X_LO, X_HI, BATCH)
        v = _log_uniform(RNG, V_LO, V_HI, BATCH)
        D = _log_uniform(RNG, D_LO, D_HI, BATCH)
        err = pred_error(x, v, D)

        if valid:
            mask = err < EPSILON
        else:
            mask = err > EPSILON

        n_keep = min(mask.sum(), n_target - collected)
        idx    = np.where(mask)[0][:n_keep]

        xs.append(x[idx]);  vs.append(v[idx])
        Ds.append(D[idx]);  errs.append(err[idx])
        collected += n_keep

    return (np.concatenate(xs),
            np.concatenate(vs),
            np.concatenate(Ds),
            np.concatenate(errs))


# ------------------------------------------------------------------
# 1.  Training data — valid regime
# ------------------------------------------------------------------
print("Generating train_empirical.npz (5 000 valid samples) …")
x_tr, v_tr, D_tr, err_tr = _sample(5_000, valid=True)

features_tr   = np.stack([x_tr, v_tr, D_tr], axis=1)     # (5000, 3)
log_crit_tr   = np.clip(
    np.log(EPSILON / (err_tr + 1e-10)), CLIP_LO, CLIP_HI  # (5000,)
)
is_valid_tr   = np.ones(len(x_tr), dtype=bool)

np.savez(
    os.path.join(OUTDIR, "train_empirical.npz"),
    features      = features_tr,
    log_criterion = log_crit_tr,
    is_valid      = is_valid_tr,
    pred_error    = err_tr,
)

# ------------------------------------------------------------------
# 2.  Test data — breakdown regime
# ------------------------------------------------------------------
print("Generating test_A2break.npz (1 000 breakdown samples) …")
x_te, v_te, D_te, err_te = _sample(1_000, valid=False)

features_te   = np.stack([x_te, v_te, D_te], axis=1)
is_valid_te   = np.zeros(len(x_te), dtype=bool)

np.savez(
    os.path.join(OUTDIR, "test_A2break.npz"),
    features   = features_te,
    is_valid   = is_valid_te,
    pred_error = err_te,
)

# ------------------------------------------------------------------
# 3.  Boundary grid — D=0.01 fixed, 50×50 (x, v)
# ------------------------------------------------------------------
print("Generating grid_boundary.npz (2 500 grid points) …")
D_grid  = 0.01
x_grid  = np.logspace(np.log10(0.01), np.log10(3.0), 50)
v_grid  = np.logspace(np.log10(0.5),  np.log10(10.0), 50)

XX, VV  = np.meshgrid(x_grid, v_grid)          # (50, 50) each
x_flat  = XX.ravel()                            # (2500,)
v_flat  = VV.ravel()
D_flat  = np.full_like(x_flat, D_grid)

err_grid = pred_error(x_flat, v_flat, D_flat)   # (2500,)
features_grid     = np.stack([x_flat, v_flat, D_flat], axis=1)
true_valid_grid   = err_grid < EPSILON

np.savez(
    os.path.join(OUTDIR, "grid_boundary.npz"),
    features         = features_grid,
    pred_error_grid  = err_grid,
    true_valid_grid  = true_valid_grid,
    x_grid           = x_grid,
    v_grid           = v_grid,
)

# ------------------------------------------------------------------
# 4.  Diagnostics
# ------------------------------------------------------------------
print()
print("=== Diagnostics ===")
print(f"train_empirical : {len(x_tr):>6} samples | "
      f"mean pred_error = {err_tr.mean():.4f}  (want < 0.05)")
print(f"test_A2break    : {len(x_te):>6} samples | "
      f"mean pred_error = {err_te.mean():.4f}  (want > 0.05)")
print(f"grid_boundary   : {len(x_flat):>6} points  | "
      f"valid fraction  = {true_valid_grid.mean():.3f}")

mean_lc = log_crit_tr.mean()
print(f"mean log_criterion (train) = {mean_lc:.2f}")
if mean_lc > 10:
    print("  CALIBRATION WARNING: mean log_criterion > 10 — training data "
          "may be too far from the boundary; consider sampling closer to "
          "pred_error = 0.05 or reducing the valid threshold.")

assert features_tr.shape  == (5000, 3), f"train features shape: {features_tr.shape}"
assert features_te.shape  == (1000, 3), f"test features shape: {features_te.shape}"
assert features_grid.shape == (2500, 3), f"grid features shape: {features_grid.shape}"
assert is_valid_tr.all(),  "All train samples should be valid"
assert (~is_valid_te).all(), "All test samples should be breakdown"

print()
print("All shape assertions passed. Files written to:")
for fname in ("train_empirical.npz", "test_A2break.npz", "grid_boundary.npz"):
    fpath = os.path.join(OUTDIR, fname)
    print(f"  {fpath}  ({os.path.getsize(fpath)//1024} KB)")
