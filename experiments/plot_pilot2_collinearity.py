"""
Collinearity failure visualization for Pilot 2 (Hooke's Law).

Left  — Pairwise scatter plots of all three training features
        (stress_ratio, strain_energy_ratio, epsilon).  Each pair lies
        exactly on a deterministic curve: the features are not independent
        predictors — they are all functions of the same underlying strain ε.

Right — Grouped bar chart comparing breakdown-detection recall under two
        training strategies:
          • Shared / collinear: every predicate sees all three features.
            The skip regression has infinitely many solutions; gradient
            descent finds random sign combinations → recall is non-deterministic
            (0.0 or 1.0 depending on seed).
          • Per-criterion isolation (the fix): each predicate sees only the
            one feature in its validity criterion → unique solution with
            forced correct sign → recall = 1.00 always.

Saves figures/pilot2_collinearity.png at 300 dpi.
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data.generate import (
    generate_hooke_dataset,
    EPSILON_Y,
)
from axiom_graph.graph import build_hooke_law_graph
from validity_predicates.train import (
    train_all_hooke_predicates,
    compute_hooke_log_criterion,
)
from validity_predicates.predicate import ValidityPredicate, HOOKE_LOG_COLS

_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "pilot2_collinearity.png",
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
N_SEEDS        = 8          # seeds for the collinear multi-run
ALL_FEATURES   = ["stress_ratio", "strain_energy_ratio", "epsilon"]
ASSUMPTION_IDS = ["A1_linearity", "A2_elasticity", "A3_small_strain"]
LABEL_COLS     = {
    "A1_linearity":    "valid_linearity",
    "A2_elasticity":   "valid_elasticity",
    "A3_small_strain": "valid_small_strain",
}
SHORT_NAMES = ["A1\nLinearity", "A2\nElasticity", "A3\nSmall strain"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recall(y_violated: np.ndarray, scores: np.ndarray) -> float:
    """Recall = TP / (TP + FN).  y_violated: True means assumption violated."""
    flagged = scores < 0.5
    tp = int((flagged & y_violated).sum())
    fn = int((~flagged & y_violated).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _train_collinear(
    assumption_id: str,
    train_df,
    seed: int,
) -> ValidityPredicate | None:
    """Train one predicate using all three Hooke features (collinear strategy)."""
    log_targets = compute_hooke_log_criterion(train_df, assumption_id)
    if log_targets is None:
        return None

    X_raw = train_df[ALL_FEATURES].values.astype(np.float32)

    torch.manual_seed(seed)
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_raw))
    n_val = max(1, int(0.15 * len(X_raw)))
    vi, ti = idx[:n_val], idx[n_val:]

    X_tr, y_tr   = X_raw[ti], log_targets[ti]
    X_val, y_val = X_raw[vi], log_targets[vi]

    mean = X_tr.mean(0).astype(np.float32)
    std  = (X_tr.std(0) + 1e-8).astype(np.float32)

    pred = ValidityPredicate(
        hidden_dims=(32, 16),
        n_features=3,
        log_transform_cols=HOOKE_LOG_COLS,   # () — no log-transform
        feature_cols=ALL_FEATURES,
    )
    pred.set_normalization(mean, std)

    opt = torch.optim.Adam([
        {"params": pred.skip.parameters(), "weight_decay": 0.0},
        {"params": pred.mlp.parameters(),  "weight_decay": 5.0},
    ], lr=1e-2)
    loss_fn = nn.MSELoss()

    Xtr  = torch.from_numpy(X_tr);   ytr  = torch.from_numpy(y_tr)
    Xval = torch.from_numpy(X_val);  yval = torch.from_numpy(y_val)

    best_val, best_sd, no_imp = float("inf"), None, 0
    for _ in range(600):
        pred.train();  opt.zero_grad()
        loss_fn(pred(Xtr), ytr).backward();  opt.step()
        pred.eval()
        with torch.no_grad():
            vl = loss_fn(pred(Xval), yval).item()
        if vl < best_val - 1e-8:
            best_val, best_sd, no_imp = vl, copy.deepcopy(pred.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= 80:
                break
    if best_sd:
        pred.load_state_dict(best_sd)
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating Hooke's data ...")
    train_df, held_out_df = generate_hooke_dataset(n_train=5000, n_held_out=2000, seed=42)

    # All held-out states are post-yield, so all three assumptions are violated
    violated = {
        aid: ~held_out_df[LABEL_COLS[aid]].values.astype(bool)
        for aid in ASSUMPTION_IDS
    }

    # ----------------------------------------------------------------
    # 1. Collinear experiment — N_SEEDS seeds
    # ----------------------------------------------------------------
    print(f"Running collinear experiment ({N_SEEDS} seeds) ...")
    coll_recalls: dict[str, list[float]] = {a: [] for a in ASSUMPTION_IDS}

    for seed in range(N_SEEDS):
        line_parts = []
        for aid in ASSUMPTION_IDS:
            pred = _train_collinear(aid, train_df, seed=seed)
            X_ho = held_out_df[ALL_FEATURES].values.astype(np.float32)
            r = _recall(violated[aid], pred.predict(X_ho))
            coll_recalls[aid].append(r)
            # Show skip weight sign for diagnosis
            w = pred.skip.weight.data.numpy().ravel()
            line_parts.append(f"{aid.split('_')[0]} R={r:.2f} w={w}")
        print(f"  seed {seed}: " + "  |  ".join(line_parts))

    coll_mean = {a: float(np.mean(coll_recalls[a])) for a in ASSUMPTION_IDS}
    coll_std  = {a: float(np.std( coll_recalls[a])) for a in ASSUMPTION_IDS}

    # ----------------------------------------------------------------
    # 2. Per-criterion experiment (the fix)
    # ----------------------------------------------------------------
    print("Training per-criterion predicates (the fix) ...")
    graph = build_hooke_law_graph()
    preds = train_all_hooke_predicates(graph, train_df, verbose=False)

    fix_recalls: dict[str, float] = {}
    for aid in ASSUMPTION_IDS:
        p = preds.get(aid)
        if p is None:
            fix_recalls[aid] = 0.0
            continue
        X_ho = held_out_df[p.feature_cols].values.astype(np.float32)
        fix_recalls[aid] = _recall(violated[aid], p.predict(X_ho))

    print("Recall summary (collinear mean ± std  |  fix):")
    for aid in ASSUMPTION_IDS:
        print(f"  {aid:30s}  {coll_mean[aid]:.3f}±{coll_std[aid]:.3f}"
              f"  ->  {fix_recalls[aid]:.3f}")

    # ----------------------------------------------------------------
    # 3. Figure
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.35)

    # -- LEFT: pairwise scatter (3 mini panels stacked) ---------------
    gs_left = gs[0].subgridspec(3, 1, hspace=0.70)
    ax0 = fig.add_subplot(gs_left[0])   # SR vs SER
    ax1 = fig.add_subplot(gs_left[1])   # ε vs SR
    ax2 = fig.add_subplot(gs_left[2])   # ε vs SER

    rng = np.random.default_rng(0)
    sub = train_df.iloc[rng.choice(len(train_df), size=500, replace=False)]
    sr  = sub["stress_ratio"].values
    ser = sub["strain_energy_ratio"].values
    eps = sub["epsilon"].values

    _sc = dict(s=9, alpha=0.55, edgecolors="none", zorder=3)

    # Panel 0 — SR vs SER  (SER ≡ SR², constructed exactly, zero scatter)
    xth = np.linspace(sr.min() * 0.95, sr.max() * 1.05, 200)
    ax0.scatter(sr, ser, color="#2980B9", **_sc)
    ax0.plot(xth, xth ** 2, color="#1A5276", lw=1.8, ls="--",
             label="SER = SR²  (exact)", zorder=4)
    ax0.set_xlabel("stress_ratio", fontsize=8)
    ax0.set_ylabel("strain_energy_ratio", fontsize=8)
    ax0.set_title(
        "stress_ratio  vs  strain_energy_ratio\n"
        "SER ≡ SR²  (zero scatter — exact algebraic identity)",
        fontsize=8.5, pad=3,
    )
    ax0.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax0.tick_params(labelsize=7)

    # Panel 1 — ε vs SR  (SR = ε/ε_y, linear, near-zero scatter)
    xth2 = np.linspace(eps.min() * 0.95, eps.max() * 1.05, 200)
    ax1.scatter(eps, sr, color="#C0392B", **_sc)
    ax1.plot(xth2, xth2 / EPSILON_Y, color="#7B241C", lw=1.8, ls="--",
             label="SR = ε/ε_y  (linear)", zorder=4)
    ax1.set_xlabel("epsilon", fontsize=8)
    ax1.set_ylabel("stress_ratio", fontsize=8)
    ax1.set_title(
        "epsilon  vs  stress_ratio\n"
        "SR ≡ ε/ε_y  (near-zero scatter — 0.2% measurement noise only)",
        fontsize=8.5, pad=3,
    )
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax1.tick_params(labelsize=7)
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Panel 2 — ε vs SER  (SER = (ε/ε_y)², quadratic)
    ax2.scatter(eps, ser, color="#27AE60", **_sc)
    ax2.plot(xth2, (xth2 / EPSILON_Y) ** 2, color="#1E8449", lw=1.8, ls="--",
             label="SER = (ε/ε_y)²  (quadratic)", zorder=4)
    ax2.set_xlabel("epsilon", fontsize=8)
    ax2.set_ylabel("strain_energy_ratio", fontsize=8)
    ax2.set_title(
        "epsilon  vs  strain_energy_ratio\n"
        "SER ≡ (ε/ε_y)²  (near-zero scatter)",
        fontsize=8.5, pad=3,
    )
    ax2.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax2.tick_params(labelsize=7)
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Left-panel header
    fig.text(
        0.265, 0.985,
        "All three Hooke features are deterministic functions of  ε",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#1A252F",
    )
    fig.text(
        0.265, 0.960,
        "The skip regression has infinitely many solutions when features are collinear",
        ha="center", va="top", fontsize=8.5, color="#555",
    )

    # -- RIGHT: grouped bar chart -------------------------------------
    ax_bar = fig.add_subplot(gs[1])

    x = np.arange(3)
    w = 0.34

    means = [coll_mean[a] for a in ASSUMPTION_IDS]
    stds  = [coll_std[a]  for a in ASSUMPTION_IDS]
    fixes = [fix_recalls[a] for a in ASSUMPTION_IDS]

    bars_c = ax_bar.bar(
        x - w / 2, means, w,
        yerr=stds, capsize=5,
        color="#E67E22", alpha=0.88, edgecolor="white", linewidth=1.3,
        label=f"Shared features  (collinear, n = {N_SEEDS} seeds)",
        error_kw=dict(elinewidth=1.6, ecolor="#784212"),
    )
    bars_f = ax_bar.bar(
        x + w / 2, fixes, w,
        color="#27AE60", alpha=0.88, edgecolor="white", linewidth=1.3,
        label="Per-criterion isolation  (the fix)",
    )

    # Value labels
    for bar, val, std in zip(bars_c, means, stds):
        label = f"{val:.2f}\n±{std:.2f}"
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + (std or 0) + 0.04, 1.13),
            label,
            ha="center", va="bottom", fontsize=7.8, color="#784212",
        )
    for bar, val in zip(bars_f, fixes):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8, color="#1E8449",
            fontweight="bold",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax_bar.set_ylim(0, 1.28)
    ax_bar.set_ylabel("Breakdown detection recall", fontsize=10.5)
    ax_bar.set_title(
        "Recall: collinear shared features  vs.  per-criterion isolation\n"
        "Orange bars: skip weight sign is random → recall unpredictable\n"
        "Green bars: single-feature skip has forced correct sign → always 1.00",
        fontsize=9, pad=6,
    )
    ax_bar.legend(fontsize=8.5, loc="upper right")
    ax_bar.axhline(1.0, color="#27AE60", lw=1.0, ls=":", alpha=0.5)
    ax_bar.grid(True, axis="y", alpha=0.28)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ----------------------------------------------------------------
    # 4. Save
    # ----------------------------------------------------------------
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
