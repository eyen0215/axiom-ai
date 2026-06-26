"""
Train three ValidityPredicate instances for Dittus-Boelter assumptions A1, A2, A3.

What each predicate sees (no Re, Pr, Nu, L/D ever given as features):
  A1: [v, D, rho, mu] log-transformed  target = log(0.05 / pred_error(Re, Pr))
  A2: [mu, cp, k]     log-transformed  target = log(0.05 / pred_error(Re, Pr))
  A3: [L, D]          log-transformed  target = log(0.05 / pred_error_A3(L))

Design notes:
  - In train_A1, only v varies; D/rho/mu are fixed constants.
    The skip can only learn from v; weights on D/rho/mu are unlearnable.
    pred_error is U-shaped in Re (Prompt 1a): error INCREASES with Re in
    [12k, 80k] at Pr=6.2, so the A1 log_criterion is mostly negative and the
    skip may learn the WRONG direction (higher v -> more invalid).
  - In train_A2, only mu varies; cp/k are fixed.
    A2 has a U-shaped validity region (Pr too low OR too high), so a linear
    skip cannot represent the boundary. MLP/skip ratio is the key diagnostic.
  - In train_A3, only L varies; D is fixed.
    log_criterion = log(L) - log(D) + const -> all-positive targets, clean
    signal. Effective weight on L should be ~+1 (integer-exponent control case).

Training hyper-parameters (fixed as requested):
  lr=1e-3, weight_decay_mlp=5.0, weight_decay_skip=0.0, epochs=600, batch=256
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from validity_predicates.predicate import ValidityPredicate

SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"
DATA_DIR = Path(__file__).parent.parent / "data" / "dittus_boelter"

LR                = 1e-3
WEIGHT_DECAY_MLP  = 5.0
WEIGHT_DECAY_SKIP = 0.0
EPOCHS            = 600
BATCH             = 256
VAL_FRAC          = 0.15
PATIENCE          = 60


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_one(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_features: int,
    log_transform_cols: tuple,
    feature_names: list,
    *,
    seed: int = 0,
    verbose: bool = True,
) -> tuple:
    """Train one ValidityPredicate; return (predicate, feat_std_log_train)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_raw))
    n_val = max(1, int(VAL_FRAC * len(X_raw)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr, y_tr   = X_raw[tr_idx], y[tr_idx]
    X_val, y_val = X_raw[val_idx], y[val_idx]

    # Normalization on log-transformed training features
    X_tr_log = X_tr.copy()
    for col in log_transform_cols:
        X_tr_log[:, col] = np.log(np.clip(X_tr[:, col], 1e-9, None))
    feat_mean = X_tr_log.mean(axis=0).astype(np.float32)
    feat_std  = (X_tr_log.std(axis=0) + 1e-8).astype(np.float32)

    pred = ValidityPredicate(
        n_features=n_features,
        log_transform_cols=log_transform_cols,
        feature_cols=feature_names,
    )
    pred.set_normalization(feat_mean, feat_std)

    opt = torch.optim.Adam([
        {"params": pred.skip.parameters(), "weight_decay": WEIGHT_DECAY_SKIP},
        {"params": pred.mlp.parameters(),  "weight_decay": WEIGHT_DECAY_MLP},
    ], lr=LR)
    loss_fn = nn.MSELoss()

    X_tr_t  = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t  = torch.from_numpy(y_tr.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        pred.train()
        perm = torch.randperm(len(X_tr_t))
        for start in range(0, len(X_tr_t), BATCH):
            bi = perm[start : start + BATCH]
            opt.zero_grad()
            loss_fn(pred(X_tr_t[bi]), y_tr_t[bi]).backward()
            opt.step()

        pred.eval()
        with torch.no_grad():
            val_loss = loss_fn(pred(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val, best_sd, no_improve = val_loss, copy.deepcopy(pred.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                if verbose:
                    print(f"    early stop @ epoch {epoch:4d}  best val MSE = {best_val:.5f}")
                break

        if verbose and epoch % 100 == 0:
            print(f"    epoch {epoch:4d}  val MSE = {val_loss:.5f}")

    if best_sd is not None:
        pred.load_state_dict(best_sd)

    return pred, feat_std


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def decompose_skip_mlp(pred: ValidityPredicate, X_raw: np.ndarray):
    """Return (skip_outputs, mlp_outputs) arrays for all samples in X_raw."""
    pred.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_raw.astype(np.float32)).clone()
        for col in pred._log_transform_cols:
            x[..., col] = torch.log(x[..., col].clamp(min=1e-9))
        x_norm   = (x - pred.feat_mean) / (pred.feat_std + 1e-8)
        skip_out = pred.skip(x_norm).squeeze(-1).numpy()
        mlp_out  = pred.mlp(x_norm).squeeze(-1).numpy()
    return skip_out, mlp_out


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def report_skip_weights(pred: ValidityPredicate, feat_names: list,
                        feat_std_log: np.ndarray) -> np.ndarray:
    """Print raw and effective skip weights; return raw weight array."""
    w = pred.skip.weight.data.numpy().ravel()
    b = pred.skip.bias.data.item()
    print("  Skip weights (raw = coefficient on normalised log-feature):")
    print(f"    {'Feature':>6}  {'raw_w':>8}  {'std_log':>10}  eff_w (raw/std)")
    for name, wi, si in zip(feat_names, w, feat_std_log):
        raw_std = float(si)
        if raw_std > 1e-4:
            eff = wi / raw_std
            note = f"{eff:>10.4f}"
        else:
            note = "      nan  <-- constant in training, unlearnable"
        print(f"    {name:>6}  {wi:>8.4f}  {raw_std:>10.6f}  {note}")
    print(f"  Bias: {b:.4f}")
    return w


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------

def train_A1() -> dict:
    sep = "-" * 65
    print(sep)
    print("A1  turbulent_flow  features=[v, D, rho, mu]  log_cols=[0,1,2,3]")
    print(sep)
    print("  Theoretical: Re = rho*v*D/mu -> all four effective weights ~+0.8")
    print("  except mu which should be ~-0.8.")
    print("  CAVEAT: D/rho/mu are FIXED in training -- weights on them")
    print("  get zero gradient and cannot be learned from this dataset.")
    print("  Also: pred_error is U-shaped in Re (Prompt 1a), so the criterion")
    print("  INCREASES with Re in [12k, 80k] -- many targets are negative.")
    print()

    d     = np.load(DATA_DIR / "train_A1.npz")
    X, lc = d["X"].astype(np.float32), d["log_criterion"].astype(np.float32)
    print(f"  n={len(X)}, X.shape={X.shape}")
    print(f"  log_criterion: mean={lc.mean():.4f}  std={lc.std():.4f}  "
          f"min={lc.min():.4f}  max={lc.max():.4f}")
    pct_neg = float(np.mean(lc < 0)) * 100
    print(f"  Fraction of targets < 0: {pct_neg:.1f}%")
    if pct_neg > 50:
        print("  NOTE: majority of training targets are NEGATIVE -- corrupted signal.")
    print()

    pred, fstd = train_one(X, lc, 4, (0, 1, 2, 3), ["v", "D", "rho", "mu"])

    print()
    w = report_skip_weights(pred, ["v", "D", "rho", "mu"], fstd)

    skip_out, mlp_out = decompose_skip_mlp(pred, X)
    rms_s, rms_m = rms(skip_out), rms(mlp_out)
    ratio = rms_m / (rms_s + 1e-9)
    print(f"\n  MLP/skip RMS ratio: {ratio:.3f}  "
          f"(rms_skip={rms_s:.4f}, rms_mlp={rms_m:.4f})")

    # Balance metric (raw weights)
    balance = abs(w[0]) + abs(w[1]) + abs(w[2]) - abs(w[3])
    print(f"  Balance |w_v|+|w_D|+|w_rho|-|w_mu| = {balance:.4f}  "
          "(near 0 if Re discovered from all four; not expected here)")

    signs_ok = w[0] > 0 and w[1] > 0 and w[2] > 0 and w[3] < 0
    print(f"  Signs correct (v+, D+, rho+, mu-): {signs_ok}")
    if not signs_ok:
        sv = "+" if w[0] > 0 else "-"
        sD = "+" if w[1] > 0 else "-"
        sr = "+" if w[2] > 0 else "-"
        sm = "+" if w[3] > 0 else "-"
        print(f"  Actual signs: v={sv}  D={sD}  rho={sr}  mu={sm}")
        if w[0] < 0:
            print("  w_v is NEGATIVE: consistent with pred_error increasing with Re")
            print("  in [12k, 80k] (U-shape from Prompt 1a). The skip correctly fits")
            print("  the criterion but the criterion itself has inverted direction.")

    SAVE_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": pred.state_dict(),
                "feature_cols": ["v", "D", "rho", "mu"],
                "log_transform_cols": (0, 1, 2, 3)}, SAVE_DIR / "db_A1.pt")
    print(f"\n  Saved -> {SAVE_DIR / 'db_A1.pt'}")

    return {"pred": pred, "w": w, "ratio": ratio, "balance": balance,
            "signs_ok": signs_ok, "pct_neg": pct_neg}


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------

def train_A2() -> dict:
    print()
    sep = "-" * 65
    print(sep)
    print("A2  moderate_prandtl  features=[mu, cp, k]  log_cols=[0,1,2]")
    print(sep)
    print("  U-shaped validity: Pr < 0.6 (liquid metal) OR Pr > 160 (oil).")
    print("  A single linear skip CANNOT represent a non-monotone boundary.")
    print("  Only mu varies in training (cp, k fixed).")
    print("  Key diagnostic: MLP/skip ratio -- expect >> 1 for U-shaped case.")
    print()

    d     = np.load(DATA_DIR / "train_A2.npz")
    X, lc = d["X"].astype(np.float32), d["log_criterion"].astype(np.float32)
    Pr    = d["Pr"] if "Pr" in d else None
    print(f"  n={len(X)}, X.shape={X.shape}")
    print(f"  log_criterion: mean={lc.mean():.4f}  std={lc.std():.4f}  "
          f"min={lc.min():.4f}  max={lc.max():.4f}")
    if Pr is not None:
        print(f"  Pr range in training: [{Pr.min():.2f}, {Pr.max():.2f}]")
    print()

    pred, fstd = train_one(X, lc, 3, (0, 1, 2), ["mu", "cp", "k"])

    print()
    w = report_skip_weights(pred, ["mu", "cp", "k"], fstd)

    skip_out, mlp_out = decompose_skip_mlp(pred, X)
    rms_s, rms_m = rms(skip_out), rms(mlp_out)
    ratio = rms_m / (rms_s + 1e-9)
    print(f"\n  MLP/skip RMS ratio: {ratio:.3f}  "
          f"(rms_skip={rms_s:.4f}, rms_mlp={rms_m:.4f})")

    if ratio < 2.0:
        print("  WARNING: MLP/skip ratio < 2.0 -- skip may be dominating.")
        print("           U-shaped boundary may not be learned.")
    else:
        print(f"  MLP dominates (ratio={ratio:.2f}): MLP carries U-shaped correction.")

    print(f"  |w_mu| = {abs(w[0]):.4f}  "
          "(large if skip still attempts linear fit; small if it gave up)")

    SAVE_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": pred.state_dict(),
                "feature_cols": ["mu", "cp", "k"],
                "log_transform_cols": (0, 1, 2)}, SAVE_DIR / "db_A2.pt")
    print(f"\n  Saved -> {SAVE_DIR / 'db_A2.pt'}")

    return {"pred": pred, "w": w, "ratio": ratio}


# ---------------------------------------------------------------------------
# A3
# ---------------------------------------------------------------------------

def train_A3() -> dict:
    print()
    sep = "-" * 65
    print(sep)
    print("A3  developed_flow  features=[L, D]  log_cols=[0,1]")
    print(sep)
    print("  Theoretical: log_criterion = log(L) - log(D) + const")
    print("  -> eff_w_L = +1, eff_w_D = -1  (integer exponents, control case).")
    print("  D is fixed in training; only L varies. All targets are positive.")
    print()

    d     = np.load(DATA_DIR / "train_A3.npz")
    X, lc = d["X"].astype(np.float32), d["log_criterion"].astype(np.float32)
    print(f"  n={len(X)}, X.shape={X.shape}")
    print(f"  log_criterion: mean={lc.mean():.4f}  std={lc.std():.4f}  "
          f"min={lc.min():.4f}  max={lc.max():.4f}")
    if np.all(lc > 0):
        print("  All targets positive -- clean, consistent valid-regime signal.")
    print()

    pred, fstd = train_one(X, lc, 2, (0, 1), ["L", "D"])

    print()
    w = report_skip_weights(pred, ["L", "D"], fstd)

    skip_out, mlp_out = decompose_skip_mlp(pred, X)
    rms_s, rms_m = rms(skip_out), rms(mlp_out)
    ratio = rms_m / (rms_s + 1e-9)
    print(f"\n  MLP/skip RMS ratio: {ratio:.3f}  "
          f"(rms_skip={rms_s:.4f}, rms_mlp={rms_m:.4f})")

    # Effective weight on L (variable feature)
    eff_L = float(w[0]) / (float(fstd[0]) + 1e-8)
    print(f"\n  Effective log-space weight on L: {eff_L:.4f}  (expect ~+1.0)")

    std_D = float(fstd[1])
    if std_D > 1e-4:
        eff_D = float(w[1]) / std_D
        print(f"  Effective log-space weight on D: {eff_D:.4f}  (expect ~-1.0)")
        print(f"  Ratio eff_L / |eff_D| = {eff_L / (abs(eff_D)+1e-9):.4f}  (expect ~1.0)")
        result_eff_D = eff_D
    else:
        print(f"  D constant in training (std_log_D={std_D:.2e}) -- eff_D unlearnable.")
        result_eff_D = float("nan")

    sign_L_ok = w[0] > 0
    print(f"  Sign of w_L correct (+): {sign_L_ok}")

    SAVE_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": pred.state_dict(),
                "feature_cols": ["L", "D"],
                "log_transform_cols": (0, 1)}, SAVE_DIR / "db_A3.pt")
    print(f"\n  Saved -> {SAVE_DIR / 'db_A3.pt'}")

    return {"pred": pred, "w": w, "ratio": ratio,
            "eff_L": eff_L, "eff_D": result_eff_D, "sign_L_ok": sign_L_ok}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(r1: dict, r2: dict, r3: dict) -> None:
    print()
    print("=" * 75)
    print("SUMMARY TABLE")
    print(f"{'Predicate':<12}  {'Key weights (raw)':^36}  {'MLP/skip':>8}  Notes")
    print("-" * 75)

    w1 = r1["w"]
    w1_str = f"v={w1[0]:+.3f} D={w1[1]:+.3f} rho={w1[2]:+.3f} mu={w1[3]:+.3f}"
    n1 = "signs OK" if r1["signs_ok"] else "SIGN ISSUE (U-shape artifact)"
    print(f"{'A1 turbulent':<12}  {w1_str:<36}  {r1['ratio']:>8.3f}  {n1}")

    w2 = r2["w"]
    w2_str = f"mu={w2[0]:+.3f} cp={w2[1]:+.3f} k={w2[2]:+.3f}"
    n2 = "MLP>>skip" if r2["ratio"] >= 2.0 else "WARNING: skip dominating"
    print(f"{'A2 mod-Pr':<12}  {w2_str:<36}  {r2['ratio']:>8.3f}  {n2}")

    w3 = r3["w"]
    eff_L_s = f"{r3['eff_L']:.3f}" if not np.isnan(r3["eff_L"]) else "nan"
    eff_D_s = f"{r3['eff_D']:.3f}" if not np.isnan(r3["eff_D"]) else "N/A"
    w3_str = f"L={w3[0]:+.3f} D={w3[1]:+.3f} (eff_L={eff_L_s})"
    n3 = f"eff_D={eff_D_s}; D const"
    print(f"{'A3 developed':<12}  {w3_str:<36}  {r3['ratio']:>8.3f}  {n3}")

    print("=" * 75)
    print()
    print("Key findings:")
    pct = r1.get("pct_neg", float("nan"))
    print(f"  A1: {pct:.0f}% of training targets are negative (pred_error > 0.05")
    print("      across Re=[12k,80k] at Pr=6.2 due to U-shape from Prompt 1a).")
    print("      The skip fits this inverted criterion; weights on D/rho/mu are")
    print("      unlearnable (constant in training).")
    print("  A2: U-shaped Pr validity cannot be represented by one linear skip.")
    print("      MLP/skip ratio reveals how much the MLP compensated.")
    print("  A3: Clean all-positive log-linear signal; effective w_L should be ~+1.")
    print("      This is the control case confirming the architecture works.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Training Dittus-Boelter validity predicates")
    print()
    r1 = train_A1()
    r2 = train_A2()
    r3 = train_A3()
    print_summary(r1, r2, r3)
