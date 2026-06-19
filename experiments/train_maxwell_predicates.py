"""Maxwell's equations: build graph, train A1/A2 predicates, evaluate.

Three tasks in one script:
  TASK 1  Build the Maxwell axiom graph and verify footprints.
  TASK 2  Train A1 (linear_media) and A2 (quasi_static) predicates.
  TASK 3  Evaluate on test scenarios; print AUROC, fire rates, FPR,
          and provenance result for Scenario B (the cross-domain key result).

Expected key result (Scenario B, A2 fires at high frequency):
  D1 wave_speed    SUSPECT   (A2 in D1 footprint)
  D2 impedance     TRUSTED   (A2 not in D2 footprint)
  D3 energy_density TRUSTED  (A2 not in D3 footprint)
  D4 polarization  TRUSTED   (A2 not in D4 footprint)

Cross-domain pattern: same as LE Scenario C (A5 fires -> only D4 SUSPECT).
Same mechanism, different physics, no new code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom_graph.maxwell_graph import build_maxwell_graph
from validity_predicates.predicate import ValidityPredicate

DATA_DIR = Path(__file__).parent.parent / "data" / "maxwell"
SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"

LR           = 1e-3
WD_SKIP      = 0.0
WD_MLP       = 5.0
EPOCHS       = 300
BATCH_SIZE   = 256
HOLDOUT_FRAC = 0.20
FIRE_THRESH  = 0.5


# ---------------------------------------------------------------------------
# Training helpers (same pattern as train_le_predicates.py)
# ---------------------------------------------------------------------------

def compute_norm_stats(
    X_raw: np.ndarray,
    log_cols: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std computed on log-transformed features (matching forward())."""
    X = X_raw.copy().astype(np.float64)
    for col in log_cols:
        X[:, col] = np.log(np.clip(X[:, col], 1e-9, None))
    mean = X.mean(axis=0).astype(np.float32)
    std  = (X.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def train_one(
    name: str,
    features_raw: np.ndarray,
    log_criterion_raw: np.ndarray,
    log_cols: tuple,
    feature_col_names: list[str],
    recenter: bool,
) -> tuple[ValidityPredicate, float]:
    """Train one ValidityPredicate.  Returns (predicate, shift)."""
    N       = len(features_raw)
    n_hold  = int(N * HOLDOUT_FRAC)
    n_train = N - n_hold
    n_feat  = features_raw.shape[1]

    X_train_raw = features_raw[:n_train].astype(np.float32)
    X_hold_raw  = features_raw[n_train:].astype(np.float32)
    lc_train    = log_criterion_raw[:n_train]

    raw_mean = float(np.mean(log_criterion_raw))
    shift    = raw_mean if recenter else 0.0
    y_train  = (lc_train - shift).astype(np.float32)

    print(f"\n--- {name} ---")
    print(f"  n_features={n_feat}, log_cols={log_cols}")
    print(f"  mean log_criterion (raw):      {raw_mean:.3f}")
    print(f"  mean log_criterion (adjusted): {float(np.mean(y_train)):.3f}")
    if raw_mean > 10.0 and not recenter:
        print(f"  WARNING: mean(log_criterion) > 10 -- calibration bias likely")

    feat_mean, feat_std = compute_norm_stats(X_train_raw, log_cols)

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

    X_t = torch.from_numpy(X_train_raw)
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
            avg = total_loss / n_batches
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.4f}")

    predicate.eval()
    with torch.no_grad():
        logits_hold = predicate(torch.from_numpy(X_hold_raw)).numpy()

    scores = 1.0 / (1.0 + np.exp(-(logits_hold + shift)))
    print(f"  holdout n={len(X_hold_raw)}: score mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")

    # Skip weight (for sanity check: should be negative — larger feature = fires)
    w = predicate.skip.weight.detach().numpy().flatten()
    b = float(predicate.skip.bias.detach().numpy().item())
    print(f"  skip weight(s): {w}  bias: {b:.4f}")

    return predicate, shift


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def score_predicate(
    pred: ValidityPredicate,
    features: np.ndarray,
    shift: float = 0.0,
) -> np.ndarray:
    """Return per-sample sigmoid validity scores with optional log_criterion shift."""
    if shift == 0.0:
        return pred.predict(features)
    with torch.no_grad():
        logits = pred(torch.from_numpy(features.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-(logits + shift)))


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------

def run_provenance(
    g,
    assumption_fired: dict[str, np.ndarray],
) -> dict[str, tuple[str, float]]:
    """SUSPECT/TRUSTED + suspect_frac for each derived node."""
    derived_nodes = [n for n in g.nodes.values() if n.kind == "derived"]
    n = next(iter(assumption_fired.values())).shape[0]
    results = {}
    for node in derived_nodes:
        fp      = g.ancestor_assumptions(node.id)
        any_fired = np.zeros(n, dtype=bool)
        for aid in fp:
            if aid in assumption_fired:
                any_fired |= assumption_fired[aid]
        frac = float(np.mean(any_fired))
        results[node.id] = ("SUSPECT" if frac > FIRE_THRESH else "TRUSTED", frac)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DERIVED_ORDER = [
    "D1_wave_speed",
    "D2_impedance",
    "D3_energy_density",
    "D4_polarization",
]
DERIVED_PARENTS = {
    "D1_wave_speed":     "A1,A2",
    "D2_impedance":      "A1,A3,A4",
    "D3_energy_density": "A1",
    "D4_polarization":   "A1,A4",
}


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # TASK 1 — Build and verify Maxwell axiom graph
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("TASK 1 -- Maxwell axiom graph")
    print("=" * 65)
    g = build_maxwell_graph()

    footprints = {
        node.id: g.ancestor_assumptions(node.id)
        for node in g.nodes.values()
        if node.kind == "derived"
    }
    for did, fp in footprints.items():
        print(f"  {did}: {sorted(fp)}")

    assert footprints["D1_wave_speed"] == frozenset({"A1_linear_media", "A2_quasi_static"})
    assert footprints["D3_energy_density"] == frozenset({"A1_linear_media"})
    print("  Graph assertions passed.")

    # -----------------------------------------------------------------------
    # TASK 2 — Train A1 and A2 predicates
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TASK 2 -- Training predicates")
    print("=" * 65)

    # --- A1: linear_media (log transform col 0; feature = E_field) ---
    d_a1 = np.load(DATA_DIR / "train_A1.npz")
    mean_lc_a1 = float(np.mean(d_a1["log_criterion"]))
    pred_a1, shift_a1 = train_one(
        name="A1 (linear_media)",
        features_raw=d_a1["features"],
        log_criterion_raw=d_a1["log_criterion"],
        log_cols=(0,),
        feature_col_names=["E_field"],
        recenter=(mean_lc_a1 > 10.0),
    )
    path_a1 = SAVE_DIR / "maxwell_A1.pt"
    if shift_a1 == 0.0:
        torch.save(pred_a1.state_dict(), path_a1)
    else:
        torch.save({"model": pred_a1.state_dict(), "shift": shift_a1}, path_a1)
    print(f"  Saved -> {path_a1}")

    # --- A2: quasi_static (log transform col 0; feature = frequency) ---
    d_a2 = np.load(DATA_DIR / "train_A2.npz")
    mean_lc_a2 = float(np.mean(d_a2["log_criterion"]))
    pred_a2, shift_a2 = train_one(
        name="A2 (quasi_static)",
        features_raw=d_a2["features"],
        log_criterion_raw=d_a2["log_criterion"],
        log_cols=(0,),
        feature_col_names=["frequency"],
        recenter=(mean_lc_a2 > 10.0),
    )
    path_a2 = SAVE_DIR / "maxwell_A2.pt"
    if shift_a2 == 0.0:
        torch.save(pred_a2.state_dict(), path_a2)
    else:
        torch.save({"model": pred_a2.state_dict(), "shift": shift_a2}, path_a2)
    print(f"  Saved -> {path_a2}")

    # -----------------------------------------------------------------------
    # TASK 3 — Evaluate on test scenarios
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TASK 3 -- Evaluation on test scenarios")
    print("=" * 65)

    d_a = np.load(DATA_DIR / "test_scenario_A.npz")
    d_b = np.load(DATA_DIR / "test_scenario_B.npz")
    d_c = np.load(DATA_DIR / "test_scenario_C.npz")

    # Score all three scenarios with both predicates
    sc_a1_a = score_predicate(pred_a1, d_a["A1_features"], shift_a1)
    sc_a1_b = score_predicate(pred_a1, d_b["A1_features"], shift_a1)
    sc_a1_c = score_predicate(pred_a1, d_c["A1_features"], shift_a1)

    sc_a2_a = score_predicate(pred_a2, d_a["A2_features"], shift_a2)
    sc_a2_b = score_predicate(pred_a2, d_b["A2_features"], shift_a2)
    sc_a2_c = score_predicate(pred_a2, d_c["A2_features"], shift_a2)

    # ---- Pooled AUROC (Scenario A and B only) ----------------------------
    # A1: label=1 in Scenario A (a1_breaks=True), label=0 in Scenario B
    # A2: label=1 in Scenario B (a2_breaks=True), label=0 in Scenario A
    labels_a1 = [1] * len(sc_a1_a) + [0] * len(sc_a1_b)
    scores_a1 = np.concatenate([1.0 - sc_a1_a, 1.0 - sc_a1_b])
    auroc_a1  = roc_auc_score(labels_a1, scores_a1)

    labels_a2 = [0] * len(sc_a2_a) + [1] * len(sc_a2_b)
    scores_a2 = np.concatenate([1.0 - sc_a2_a, 1.0 - sc_a2_b])
    auroc_a2  = roc_auc_score(labels_a2, scores_a2)

    # ---- Fire rates -------------------------------------------------------
    fr_a1_a = float(np.mean(sc_a1_a < FIRE_THRESH))
    fr_a2_a = float(np.mean(sc_a2_a < FIRE_THRESH))
    fr_a1_b = float(np.mean(sc_a1_b < FIRE_THRESH))
    fr_a2_b = float(np.mean(sc_a2_b < FIRE_THRESH))

    # ---- FPR on Scenario C (valid holdout) --------------------------------
    fpr_a1 = float(np.mean(sc_a1_c < FIRE_THRESH))
    fpr_a2 = float(np.mean(sc_a2_c < FIRE_THRESH))

    print("\nAUROC (pooled Scenario A + B, 2000 samples per predicate):")
    print(f"  A1 linear_media:  {auroc_a1:.4f}")
    print(f"  A2 quasi_static:  {auroc_a2:.4f}")

    print("\nScenario A fire rates (A1 should fire, A2 should be silent):")
    print(f"  A1 fire rate: {fr_a1_a*100:.1f}%  (expected > 80%)")
    print(f"  A2 fire rate: {fr_a2_a*100:.1f}%  (expected < 10%)")

    print("\nScenario B fire rates (A2 should fire, A1 should be silent):")
    print(f"  A1 fire rate: {fr_a1_b*100:.1f}%  (expected < 10%)")
    print(f"  A2 fire rate: {fr_a2_b*100:.1f}%  (expected > 80%)")

    print("\nFalse positive rate on Scenario C (valid holdout, target < 5%):")
    print(f"  A1 FPR: {fpr_a1*100:.1f}%  ({'OK' if fpr_a1 < 0.05 else 'WARNING: exceeds 5%'})")
    print(f"  A2 FPR: {fpr_a2*100:.1f}%  ({'OK' if fpr_a2 < 0.05 else 'WARNING: exceeds 5%'})")

    # ---- Score distribution on Scenario B --------------------------------
    print(f"\nScenario B score distributions:")
    print(f"  A1 scores: mean={np.mean(sc_a1_b):.4f}  std={np.std(sc_a1_b):.4f}  "
          f"(high = valid, expected high since A1 silent)")
    print(f"  A2 scores: mean={np.mean(sc_a2_b):.4f}  std={np.std(sc_a2_b):.4f}  "
          f"(low = broken, expected low since A2 fires)")

    # ---- Provenance for Scenario B ----------------------------------------
    fired_a1_b = sc_a1_b < FIRE_THRESH
    fired_a2_b = sc_a2_b < FIRE_THRESH

    prov_b = run_provenance(g, {
        "A1_linear_media": fired_a1_b,
        "A2_quasi_static": fired_a2_b,
    })

    print("\nProvenance (Scenario B: A2 fires, A1 silent):")
    print(f"  {'Derived quantity':<22}  {'Parents':<12}  {'Status':<8}  suspect_frac")
    for did in DERIVED_ORDER:
        status, frac = prov_b[did]
        print(f"  {did:<22}  [{DERIVED_PARENTS[did]:<10}]  {status:<8}  ({frac:.2f})")

    # ---- KEY RESULT -------------------------------------------------------
    expected_b = {
        "D1_wave_speed":     "SUSPECT",
        "D2_impedance":      "TRUSTED",
        "D3_energy_density": "TRUSTED",
        "D4_polarization":   "TRUSTED",
    }
    all_correct = all(prov_b[k][0] == v for k, v in expected_b.items())

    print(f"\n{'='*65}")
    print("KEY RESULT -- Scenario B: high frequency, A2 fires only")
    print("  Expected: D1 SUSPECT, D2/D3/D4 TRUSTED")
    print()
    for did in DERIVED_ORDER:
        status, frac = prov_b[did]
        exp    = expected_b[did]
        marker = "OK" if status == exp else "FAIL"
        print(f"  {did:<22}  {status:<8}  expected={exp}  [{marker}]")
    print()
    if all_correct:
        print("  PASS: all four derived quantities correctly classified.")
        print("  Cross-domain pattern confirmed: same mechanism as LE Scenario C.")
    else:
        print("  FAIL: one or more derived quantities misclassified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
