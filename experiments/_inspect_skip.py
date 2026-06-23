"""Inspect pipe_A2 skip connection weights vs theoretical physics."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

# ---- 1. Load ---------------------------------------------------------------
ckpt  = torch.load("validity_predicates/saved/pipe_A2.pt", weights_only=False)
model = ValidityPredicate(n_features=5, log_transform_cols=(0, 1, 2, 3, 4))
model.load_state_dict(ckpt["state_dict"])
model.eval()

w    = model.skip.weight.detach().numpy().flatten()   # (5,)
b    = float(model.skip.bias.detach().numpy())
mean = model.feat_mean.numpy()
std  = model.feat_std.numpy()

FEATURES = ["x", "v", "D", "rho", "mu"]

print("=" * 62)
print("STEP 1 — Raw stored tensors")
print("=" * 62)
print(f"skip.weight : {w}")
print(f"skip.bias   : {b:.6f}")
print(f"feat_mean   : {mean}")
print(f"feat_std    : {std}")

# ---- 2. Effective weights --------------------------------------------------
# model: skip(x_norm) = w @ ((log(x) - mean) / std) + b
# =>  effective_weight[i] = w[i] / std[i]   (weight in original log-feature space)
eff_w = w / std

print()
print("=" * 62)
print("STEP 2 — Effective weights in log-feature space  (w_i / std_i)")
print("=" * 62)
for feat, ew, wi, si in zip(FEATURES, eff_w, w, std):
    flag = "  ** constant feature (std ~ 0) **" if si < 1e-3 else ""
    print(f"  {feat:4s}  eff_weight = {ew:+.4f}   "
          f"(raw_w = {wi:+.6f},  std = {si:.6e}){flag}")

# ---- 3. Theoretical weights ------------------------------------------------
# log_criterion = log(x / L_entry)
#   L_entry = 0.06 * (rho*v*D/mu) * D  =  0.06 * rho * v * D^2 / mu
# => log(x/L_entry) = log(x) - log(0.06) - log(rho) - log(v) - 2*log(D) + log(mu)
# Theoretical weights: x=+1, v=-1, D=-2, rho=-1, mu=+1
THEO = np.array([+1.0, -1.0, -2.0, -1.0, +1.0])

print()
print("=" * 62)
print("STEP 3 — Theoretical weights in log-feature space")
print("  log(x/L_entry) = log(x) - log(v) - 2*log(D) - log(rho) + log(mu)")
print("  (rho, mu are constants in training; absorbed into bias)")
print("=" * 62)
for feat, tw in zip(FEATURES, THEO):
    print(f"  {feat:4s}  theoretical = {tw:+.3f}")

# ---- 4 & 5. Comparison + sign check ----------------------------------------
print()
print("=" * 62)
print("STEP 4 & 5 — Comparison table + sign agreement")
print("=" * 62)
hdr = (f"{'Feature':8s} | {'Theoretical':>12s} | "
       f"{'Learned eff.':>14s} | {'Ratio L/T':>10s} | Sign")
print(hdr)
print("-" * len(hdr))
for feat, tw, ew, si in zip(FEATURES, THEO, eff_w, std):
    if si < 1e-3:
        ratio_str = "N/A (const)"
        sign_str  = "N/A (const)"
    else:
        ratio     = ew / tw
        ratio_str = f"{ratio:+.3f}"
        sign_str  = "CORRECT" if (ew * tw > 0) else "WRONG"
    print(f"{feat:8s} | {tw:+12.3f} | {ew:+14.4f} | {ratio_str:>10s} | {sign_str}")

# ---- 6. |w_D| / |w_v| ratio ------------------------------------------------
print()
print("=" * 62)
print("STEP 6 — |w_D| / |w_v|  (expect ~2.0 if model found D^2 in L_entry)")
print("=" * 62)
ew_v    = eff_w[1]
ew_D    = eff_w[2]
ratio_Dv = abs(ew_D) / (abs(ew_v) + 1e-12)
close   = "YES — close to 2.0" if abs(ratio_Dv - 2.0) < 0.4 else "NO  — not close to 2.0"
print(f"  |eff_w_D| = {abs(ew_D):.4f}")
print(f"  |eff_w_v| = {abs(ew_v):.4f}")
print(f"  |w_D| / |w_v| = {ratio_Dv:.4f}   ->   {close}")

# ---- summary ---------------------------------------------------------------
print()
print("=" * 62)
print("SUMMARY")
print("=" * 62)
var_idx   = [i for i, s in enumerate(std) if s >= 1e-3]
var_feats = [FEATURES[i] for i in var_idx]
n_correct = sum(1 for i in var_idx if eff_w[i] * THEO[i] > 0)
print(f"  Variable features (std > 1e-3):  {var_feats}")
print(f"  Sign correct:  {n_correct}/{len(var_idx)} variable features")
print(f"  |w_D|/|w_v| = {ratio_Dv:.4f}   (theoretical = 2.000)")
if n_correct == len(var_idx) and abs(ratio_Dv - 2.0) < 0.4:
    print("  VERDICT: skip discovered the correct physics structure.")
elif n_correct == len(var_idx):
    print("  VERDICT: signs all correct; D^2 scaling is approximate.")
else:
    print("  VERDICT: at least one sign is wrong — partial discovery only.")
