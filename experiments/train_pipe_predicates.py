"""
Train ValidityPredicate for pipe flow assumptions A1, A2, A3.
Evaluate AUROC (neural vs trivial single-variable threshold baseline) for each.

Honesty invariant: each predicate receives only raw independent variables.
Re and L_entry are NEVER given as inputs; the model must discover any
necessary combinations itself.
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

DATA_DIR = Path("data/pipe_flow")
SAVE_DIR  = Path("validity_predicates/saved")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _log_transform(features: np.ndarray, log_cols: tuple) -> np.ndarray:
    """Return a log-transformed copy (same as what ValidityPredicate.forward does)."""
    x = features.copy().astype(np.float32)
    for c in log_cols:
        x[:, c] = np.log(np.clip(x[:, c], 1e-9, None))
    return x


def train_pipe_predicate(
    features: np.ndarray,
    targets: np.ndarray,
    n_features: int,
    log_cols: tuple,
    *,
    lr: float = 1e-3,
    weight_decay_mlp: float = 5.0,
    weight_decay_skip: float = 0.0,
    epochs: int = 300,
    batch_size: int = 256,
    seed: int = 0,
    label: str = "",
) -> ValidityPredicate:
    torch.manual_seed(seed)

    model = ValidityPredicate(n_features=n_features, log_transform_cols=log_cols)

    # Normalization: computed on log-transformed features to match forward()
    x_log = _log_transform(features, log_cols)
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
        batch_losses: list[float] = []
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            # tensor input -> nn.Module.__call__ -> forward() -> raw logit
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"    [{label}] epoch {epoch:3d}  train_mse={np.mean(batch_losses):.5f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_auroc(
    valid_feats: np.ndarray,
    break_feats: np.ndarray,
    model: ValidityPredicate,
    trivial_col: int,
) -> tuple[float, float]:
    """Return (neural_auroc, trivial_auroc).

    y_true = 1 for valid (assumption holds), 0 for breaking.
    Neural score  = predicate validity score in (0, 1); higher => more valid.
    Trivial score = single feature column; best of both monotone directions taken.
    """
    n_v = len(valid_feats)
    n_b = len(break_feats)
    y_true   = np.concatenate([np.ones(n_v), np.zeros(n_b)])
    all_feats = np.vstack([
        valid_feats.astype(np.float32),
        break_feats.astype(np.float32),
    ])

    # Neural
    neural_scores = model.predict(all_feats)
    neural_auroc  = roc_auc_score(y_true, neural_scores)

    # Trivial: single-variable threshold (best monotone direction)
    col_vals      = all_feats[:, trivial_col]
    raw_auroc     = roc_auc_score(y_true, col_vals)
    trivial_auroc = max(raw_auroc, 1.0 - raw_auroc)

    return neural_auroc, trivial_auroc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[float, float]] = {}

    # -----------------------------------------------------------------------
    # A1 — laminar_flow
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("A1: laminar_flow")
    print("  Features given to model : v, D, rho, mu  (4 raw variables)")
    print("  Training target         : log(2300 / Re)  clipped to [-20, 20]")
    print("  NOT given               : Re = rho*v*D/mu  (model must discover)")
    print("=" * 60)

    d          = np.load(DATA_DIR / "train_A1.npz")
    feats_A1   = d["features"].astype(np.float32)      # (5000, 4)
    targets_A1 = d["log_criterion"].astype(np.float32)  # (5000,)
    print(f"  Train: {len(feats_A1)} samples  shape={feats_A1.shape}")
    print(f"  log_criterion  min={targets_A1.min():.3f}  "
          f"max={targets_A1.max():.3f}  mean={targets_A1.mean():.3f}")

    model_A1 = train_pipe_predicate(
        feats_A1, targets_A1,
        n_features=4, log_cols=(0, 1, 2, 3),
        label="A1",
    )
    torch.save({
        "state_dict": model_A1.state_dict(),
        "n_features": 4,
        "log_cols":   (0, 1, 2, 3),
        "feat_mean":  model_A1.feat_mean.numpy().tolist(),
        "feat_std":   model_A1.feat_std.numpy().tolist(),
    }, SAVE_DIR / "pipe_A1.pt")
    print(f"  Saved -> {SAVE_DIR / 'pipe_A1.pt'}")

    d_break        = np.load(DATA_DIR / "test_scenario_A1break.npz")
    break_feats_A1 = d_break["A1_features"].astype(np.float32)   # (1000, 4)
    valid_hold_A1  = feats_A1[:1000]                              # 1000 valid holdout

    neural_A1, trivial_A1 = compute_auroc(
        valid_hold_A1, break_feats_A1, model_A1, trivial_col=0   # col 0 = v
    )
    results["A1"] = (neural_A1, trivial_A1)
    print(f"  AUROC  neural={neural_A1:.4f}  trivial-v={trivial_A1:.4f}  "
          f"gap={neural_A1 - trivial_A1:.4f}")

    # -----------------------------------------------------------------------
    # A2 — fully_developed
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("A2: fully_developed  [KEY — diagonal boundary in (x,v) space]")
    print("  Features given to model : x, v, D, rho, mu  (5 raw variables)")
    print("  Training target         : log(x / L_entry)  clipped to [-20, 20]")
    print("  NOT given               : L_entry = 0.06*Re*D  (model must discover)")
    print("=" * 60)

    d          = np.load(DATA_DIR / "train_A2.npz")
    feats_A2   = d["features"].astype(np.float32)      # (5000, 5)
    targets_A2 = d["log_criterion"].astype(np.float32)  # (5000,)
    print(f"  Train: {len(feats_A2)} samples  shape={feats_A2.shape}")
    print(f"  log_criterion  min={targets_A2.min():.3f}  "
          f"max={targets_A2.max():.3f}  mean={targets_A2.mean():.3f}")

    model_A2 = train_pipe_predicate(
        feats_A2, targets_A2,
        n_features=5, log_cols=(0, 1, 2, 3, 4),
        label="A2",
        epochs=600,   # MSE still dropping at epoch 300 in Attempt 1; extra steps help converge
    )
    torch.save({
        "state_dict": model_A2.state_dict(),
        "n_features": 5,
        "log_cols":   (0, 1, 2, 3, 4),
        "feat_mean":  model_A2.feat_mean.numpy().tolist(),
        "feat_std":   model_A2.feat_std.numpy().tolist(),
    }, SAVE_DIR / "pipe_A2.pt")
    print(f"  Saved -> {SAVE_DIR / 'pipe_A2.pt'}")

    d_break        = np.load(DATA_DIR / "test_scenario_A2break.npz")
    break_feats_A2 = d_break["A2_features"].astype(np.float32)   # (1000, 5)
    valid_hold_A2  = feats_A2[:1000]                              # 1000 valid holdout

    neural_A2, trivial_A2 = compute_auroc(
        valid_hold_A2, break_feats_A2, model_A2, trivial_col=0   # col 0 = x
    )
    results["A2"] = (neural_A2, trivial_A2)
    print(f"  AUROC  neural={neural_A2:.4f}  trivial-x={trivial_A2:.4f}  "
          f"gap={neural_A2 - trivial_A2:.4f}")

    # -----------------------------------------------------------------------
    # A3 — incompressible
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("A3: incompressible  [control — single variable, gap ~0 expected]")
    print("  Features given to model : v  (1 raw variable)")
    print("  Training target         : log(102.9 / v)  clipped to [-20, 20]")
    print("  NOT given               : c_sound=343 or threshold 102.9 explicitly")
    print("=" * 60)

    d          = np.load(DATA_DIR / "train_A3.npz")
    feats_A3   = d["features"].astype(np.float32)      # (5000, 1)
    targets_A3 = d["log_criterion"].astype(np.float32)  # (5000,)
    print(f"  Train: {len(feats_A3)} samples  shape={feats_A3.shape}")
    print(f"  log_criterion  min={targets_A3.min():.3f}  "
          f"max={targets_A3.max():.3f}  mean={targets_A3.mean():.3f}")

    model_A3 = train_pipe_predicate(
        feats_A3, targets_A3,
        n_features=1, log_cols=(0,),
        label="A3",
    )
    torch.save({
        "state_dict": model_A3.state_dict(),
        "n_features": 1,
        "log_cols":   (0,),
        "feat_mean":  model_A3.feat_mean.numpy().tolist(),
        "feat_std":   model_A3.feat_std.numpy().tolist(),
    }, SAVE_DIR / "pipe_A3.pt")
    print(f"  Saved -> {SAVE_DIR / 'pipe_A3.pt'}")

    d_break        = np.load(DATA_DIR / "test_scenario_A3break.npz")
    break_feats_A3 = d_break["A3_features"].astype(np.float32)   # (1000, 1)
    valid_hold_A3  = feats_A3[:1000]                              # 1000 valid holdout

    neural_A3, trivial_A3 = compute_auroc(
        valid_hold_A3, break_feats_A3, model_A3, trivial_col=0   # col 0 = v
    )
    results["A3"] = (neural_A3, trivial_A3)
    print(f"  AUROC  neural={neural_A3:.4f}  trivial-v={trivial_A3:.4f}  "
          f"gap={neural_A3 - trivial_A3:.4f}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n")
    hdr = f"{'Predicate':<12} {'Neural AUROC':>13} {'Trivial AUROC':>14} {'Gap':>7}  Note"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    notes = {
        "A1": "Re discovered from (v,D,rho,mu); combo predicate",
        "A2": "KEY: diagonal boundary — x threshold insufficient",
        "A3": "control: v is the only input; gap ~0 expected",
    }
    for pred in ("A1", "A2", "A3"):
        neural, trivial = results[pred]
        gap = neural - trivial
        triv_label = "trivial-x" if pred == "A2" else "trivial-v"
        print(
            f"{pred:<12} {neural:>13.4f} {trivial:>14.4f} {gap:>7.4f}  {notes[pred]}"
        )
    print("=" * len(hdr))


if __name__ == "__main__":
    main()
