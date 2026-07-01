"""
Train ValidityPredicate for A2 (fully_developed) at five distances from breakdown boundary.

The key question: do skip weights degrade (lose sign or magnitude) as training
data moves farther from the breakdown boundary?

Theoretical log-space decomposition:
  log(x / L_entry) = log(x) - log(v) - 2*log(D) - log(rho) + log(mu) + const
  => variable-feature signs: +1 (x), -1 (v), -2 (D)
  => |eff_w_D| / |eff_w_v| should converge to 2.0

Note: rho=1.2 and mu=1.81e-5 are constants in every training file.
The model cannot learn from them; their skip weights are not updated by training.

Honesty: [x, v, D, rho, mu] are given; Re and L_entry are NOT given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

DATA_DIR = Path("data/degradation_curve")
SAVE_DIR  = Path("validity_predicates/saved")

MULTIPLIERS   = [1.05, 2, 5, 10, 20]
LOG_COLS      = (0, 1, 2, 3, 4)          # log-transform all 5: x, v, D, rho, mu
FEATURE_NAMES = ["x", "v", "D", "rho", "mu"]

# Theoretical log-space gradient of log(x/L_entry) w.r.t. each log-feature
THEORY_SIGNS = [+1, -1, -2, -1, +1]
# Only x, v, D vary in training; rho and mu are constant -> their weights are noise
VARIABLE_IDX = [0, 1, 2]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _log_transform(features: np.ndarray, log_cols: tuple) -> np.ndarray:
    x = features.copy().astype(np.float32)
    for c in log_cols:
        x[:, c] = np.log(np.clip(x[:, c], 1e-9, None))
    return x


def train_one(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    lr: float = 1e-3,
    weight_decay_mlp: float = 5.0,
    weight_decay_skip: float = 0.0,
    epochs: int = 600,
    batch_size: int = 256,
    seed: int = 0,
    label: str = "",
) -> ValidityPredicate:
    torch.manual_seed(seed)

    model = ValidityPredicate(
        n_features=5,
        log_transform_cols=LOG_COLS,
        feature_cols=FEATURE_NAMES,
    )

    x_log     = _log_transform(features, LOG_COLS)
    feat_mean = x_log.mean(axis=0).astype(np.float32)
    feat_std  = (x_log.std(axis=0) + 1e-8).astype(np.float32)
    model.set_normalization(feat_mean, feat_std)

    opt = torch.optim.Adam([
        {"params": model.skip.parameters(), "weight_decay": weight_decay_skip},
        {"params": model.mlp.parameters(),  "weight_decay": weight_decay_mlp},
    ], lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(features.astype(np.float32))
    y_t = torch.from_numpy(targets.astype(np.float32))
    n   = len(features)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])   # tensor -> forward() -> raw logit
            loss.backward()
            opt.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                full_loss = loss_fn(model(X_t), y_t).item()
            model.train()
            print(f"    [{label}] epoch {epoch:3d}  train_mse={full_loss:.5f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Weight analysis
# ---------------------------------------------------------------------------

def get_weights(model: ValidityPredicate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_w, feat_std, eff_w).

    eff_w[i] = raw_w[i] / feat_std[i] is the effective coefficient on log(x_i):
      skip_out = Σ raw_w[i] * (log(x_i) - mean_i) / std_i
               = Σ eff_w[i] * log(x_i) + const

    For constant features (std < 1e-4) eff_w[i] is set to nan.
    """
    raw_w = model.skip.weight.data.numpy().ravel()
    std   = model.feat_std.numpy()
    eff_w = np.where(std > 1e-4, raw_w / std, np.nan)
    return raw_w, std, eff_w


def sign_check(eff_w: np.ndarray) -> tuple[bool, str]:
    """Check variable-feature signs (x, v, D) against theoretical direction."""
    ok = True
    for i in VARIABLE_IDX:
        if np.isnan(eff_w[i]) or ((eff_w[i] > 0) != (THEORY_SIGNS[i] > 0)):
            ok = False
    parts = []
    for v in eff_w:
        parts.append("N/A" if np.isnan(v) else ("+" if v >= 0 else "-"))
    return ok, "[" + ", ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Degradation curve — A2 (fully_developed) skip weight analysis")
    print("=" * 62)
    print()
    print("Features given : x, v, D, rho, mu  (5 raw; NOT Re or L_entry)")
    print("Target         : log(x / L_entry),  L_entry = 0.06*Re*D")
    print()
    print("rho=1.2 and mu=1.81e-5 are constant in every training file.")
    print("Gradient w.r.t. their skip weights is always 0 — excluded from analysis.")
    print()
    print("Theoretical effective coefficients in log-feature space:")
    print("  d(log_crit)/d(log x) = +1   [more x  => more valid]")
    print("  d(log_crit)/d(log v) = -1   [more v  => larger L_entry]")
    print("  d(log_crit)/d(log D) = -2   [D^2 in L_entry = 0.06*rho*v*D^2/mu]")
    print("  => |eff_w_D|/|eff_w_v| should be ~2.0")
    print()

    summary: list[dict] = []

    for M in MULTIPLIERS:
        label = f"M={M:g}"
        fname = DATA_DIR / f"train_M{M:g}.npz"

        print(f"\n{'='*62}")
        print(f"  {label}  —  {fname.name}")
        print(f"{'='*62}")

        d     = np.load(fname)
        feats = d["features"].astype(np.float32)
        tgts  = d["log_criterion"].astype(np.float32)
        print(f"  n={len(feats)}  "
              f"log_crit in [{tgts.min():.3f}, {tgts.max():.3f}]  "
              f"mean={tgts.mean():.3f}")

        seed  = int(M * 100) % (2**31)
        model = train_one(feats, tgts, label=label, seed=seed)

        raw_w, feat_std, eff_w = get_weights(model)
        ok, signs_str          = sign_check(eff_w)
        ratio = (abs(eff_w[2]) / (abs(eff_w[1]) + 1e-12)
                 if not (np.isnan(eff_w[2]) or np.isnan(eff_w[1])) else np.nan)

        print(f"\n  Raw skip weights   [x, v, D, rho, mu]:")
        print(f"    [{', '.join(f'{w:>8.4f}' for w in raw_w)}]")
        print(f"  Feature std (log)  [x, v, D, rho, mu]:")
        print(f"    [{', '.join(f'{s:>8.4f}' for s in feat_std)}]")
        print(f"  Eff weights w/std  [x, v, D, rho*, mu*]:")
        eff_str = ", ".join(f"{v:>7.3f}" if not np.isnan(v) else "    N/A" for v in eff_w)
        print(f"    [{eff_str}]")
        print(f"  Signs: {signs_str}  ->  {'ALL CORRECT (x,v,D)' if ok else 'SIGN MISMATCH'}")
        ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"
        print(f"  |eff_w_D|/|eff_w_v| = {ratio_str}  (expect ~2.0)")

        save_path = SAVE_DIR / f"pipe_A2_M{M:g}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "n_features": 5,
            "log_cols":   LOG_COLS,
            "feat_mean":  model.feat_mean.numpy().tolist(),
            "feat_std":   model.feat_std.numpy().tolist(),
            "M":          M,
        }, save_path)
        print(f"  Saved -> {save_path}")

        summary.append({
            "M": M,
            "eff_x": float(eff_w[0]), "eff_v": float(eff_w[1]), "eff_D": float(eff_w[2]),
            "signs_ok": ok,
            "ratio": float(ratio) if not np.isnan(ratio) else float("nan"),
        })

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print()
    print("=" * 72)
    print("SUMMARY — Skip weight degradation by training distance from boundary")
    print("=" * 72)
    hdr = (f"{'M':<6} | {'w_x':>7} | {'w_v':>7} | {'w_D':>7} | "
           f"{'signs correct?':<14} | {'|wD|/|wv|':>9}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary:
        ratio_s  = f"{row['ratio']:>9.3f}" if not np.isnan(row["ratio"]) else "       N/A"
        signs_s  = "YES" if row["signs_ok"] else "NO"
        print(
            f"{row['M']:<6g} | {row['eff_x']:>7.3f} | {row['eff_v']:>7.3f} | "
            f"{row['eff_D']:>7.3f} | {signs_s:<14} | {ratio_s}"
        )
    print("-" * len(hdr))
    print(
        f"{'theory':<6} | {'+1.000':>7} | {'-1.000':>7} | {'-2.000':>7} | "
        f"{'EXPECTED':<14} | {'  2.000':>9}"
    )
    print("=" * len(hdr))
    print()
    print("* eff_w = raw_skip_weight / feat_std  (log-space coefficient on each log-feature)")
    print("  rho and mu columns are constant; their eff_w is undefined and excluded.")
    print()
    print("Key question: do w_x, w_v, w_D maintain correct signs as M increases?")
    print("If YES at M=20, extrapolation direction is learned even from far-boundary data.")


if __name__ == "__main__":
    main()
