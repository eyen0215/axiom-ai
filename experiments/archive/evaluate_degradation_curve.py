"""
Evaluate all five degradation-curve predicates on the shared near-boundary test set.

For each M in [1.05, 2, 5, 10, 20]:
  Load pipe_A2_M{M}.pt (trained on data at M * L_entry from the boundary)
  Score all 1000 test samples from test_near_boundary.npz
  Compute Neural AUROC vs is_valid label
  Compute Trivial AUROC: best single-variable threshold on x alone (col 0)
  Compute Gap = neural - trivial

Honesty note:
  Test set has x in [0.5 * L_entry, 2.0 * L_entry] — straddles the boundary.
  is_valid = (x > L_entry).  Both labels present (~67% valid, ~33% invalid).
  Trivial baseline uses raw x value only — ignores that L_entry varies with v, D.
  Neural predictor can in principle recover x/L_entry from the 5 raw features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

DATA_DIR  = Path("data/degradation_curve")
SAVE_DIR  = Path("validity_predicates/saved")
TEST_FILE = DATA_DIR / "test_near_boundary.npz"

MULTIPLIERS = [1.05, 2, 5, 10, 20]
RHO = 1.2
MU  = 1.81e-5
ENTRY_COEFF = 0.06


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(M: float) -> ValidityPredicate:
    ckpt = torch.load(
        SAVE_DIR / f"pipe_A2_M{M:g}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ValidityPredicate(
        n_features=ckpt["n_features"],
        log_transform_cols=tuple(ckpt["log_cols"]),
        feature_cols=["x", "v", "D", "rho", "mu"],
    )
    model.set_normalization(
        np.array(ckpt["feat_mean"], dtype=np.float32),
        np.array(ckpt["feat_std"],  dtype=np.float32),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def training_mean_x_over_L(M: float) -> float:
    """Compute mean(x / L_entry) from the training file for this M."""
    d    = np.load(DATA_DIR / f"train_M{M:g}.npz")
    f    = d["features"]
    x, v, D = f[:, 0], f[:, 1], f[:, 2]
    Re       = RHO * v * D / MU
    L_entry  = ENTRY_COEFF * Re * D
    return float((x / L_entry).mean())


def compute_auroc(
    features: np.ndarray,
    y_true: np.ndarray,
    model: ValidityPredicate,
) -> tuple[float, float, float]:
    """Return (neural_auroc, trivial_auroc, gap).

    Trivial baseline: best monotone threshold on x alone (column 0).
    y_true: 1 = valid (assumption holds), 0 = invalid.
    """
    neural_scores = model.predict(features.astype(np.float32))
    neural_auroc  = float(roc_auc_score(y_true, neural_scores))

    x_col         = features[:, 0].astype(np.float64)
    raw           = float(roc_auc_score(y_true, x_col))
    trivial_auroc = max(raw, 1.0 - raw)

    gap = neural_auroc - trivial_auroc
    return neural_auroc, trivial_auroc, gap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load shared test set
    # ------------------------------------------------------------------
    d        = np.load(TEST_FILE)
    features = d["features"].astype(np.float32)      # (1000, 5)
    y_true   = d["is_valid"].astype(int)             # 1=valid, 0=invalid
    n_valid  = int(y_true.sum())
    n_inval  = len(y_true) - n_valid

    print("=" * 68)
    print("Degradation curve — AUROC evaluation on shared near-boundary test set")
    print("=" * 68)
    print()
    print("Test set:  test_near_boundary.npz")
    print(f"  n={len(features)}  valid={n_valid}  invalid={n_inval}  "
          f"({100*y_true.mean():.1f}% valid)")
    print("  x in [0.5 * L_entry, 2.0 * L_entry] per sample")
    print("  is_valid = (x > L_entry)  — true physical label")
    print()
    print("Trivial baseline: best monotone threshold on raw x value alone.")
    print("  (Ignores v, D variation in L_entry — correct answer requires x/L_entry)")
    print()

    # ------------------------------------------------------------------
    # Evaluate each predicate
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for M in MULTIPLIERS:
        model             = load_model(M)
        mean_xL           = training_mean_x_over_L(M)
        n_auroc, t_auroc, gap = compute_auroc(features, y_true, model)

        print(f"M={M:g}:  "
              f"train mean(x/L)={mean_xL:.2f}  "
              f"neural={n_auroc:.4f}  trivial={t_auroc:.4f}  gap={gap:+.4f}")

        rows.append({
            "M":            M,
            "mean_xL":      mean_xL,
            "neural_auroc": n_auroc,
            "trivial_auroc":t_auroc,
            "gap":          gap,
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("KEY RESULT — AUROC vs training distance from breakdown boundary")
    print("=" * 72)
    hdr = (f"{'M':<6} | {'Train x/L':>10} | {'Neural AUROC':>12} | "
           f"{'Trivial AUROC':>13} | {'Gap':>7}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['M']:<6g} | {r['mean_xL']:>10.2f} | {r['neural_auroc']:>12.4f} | "
            f"{r['trivial_auroc']:>13.4f} | {r['gap']:>+7.4f}"
        )
    print("=" * len(hdr))

    # ------------------------------------------------------------------
    # Threshold analysis
    # ------------------------------------------------------------------
    neural  = [r["neural_auroc"]  for r in rows]
    gaps    = [r["gap"]           for r in rows]
    Ms      = [r["M"]             for r in rows]

    def first_below(values, threshold, labels) -> str:
        for v, m in zip(values, labels):
            if v < threshold:
                return str(m)
        return "> 20 (not reached in tested range)"

    def first_below_zero(values, labels) -> str:
        for v, m in zip(values, labels):
            if v < 0:
                return str(m)
        return "> 20 (not reached in tested range)"

    m_below_080  = first_below(neural, 0.80, Ms)
    m_gap_below  = first_below(gaps,   0.05, Ms)
    m_below_triv = first_below_zero(gaps, Ms)

    print()
    print("Threshold crossings (practical limits of the system):")
    print(f"  Neural AUROC drops below 0.80 at M = {m_below_080}")
    print(f"  Gap over trivial drops below 0.05 at M = {m_gap_below}")
    print(f"  Neural AUROC drops below trivial at M = {m_below_triv}")
    print()

    # Contextual interpretation
    best_gap_M  = Ms[int(np.argmax(gaps))]
    worst_neural = min(neural)
    print("Interpretation:")
    print(f"  Best neural-vs-trivial gap at M={best_gap_M}")
    print(f"  Worst neural AUROC in tested range = {worst_neural:.4f}  (M=20)")
    if worst_neural >= 0.65:
        print(f"  => Phase 1 success criterion met (AUROC > 0.65 at M=20): YES")
    else:
        print(f"  => Phase 1 success criterion met (AUROC > 0.65 at M=20): NO — "
              f"honest limitation: system is not useful beyond M~{m_below_080}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = DATA_DIR / "auroc_results.npz"
    np.savez(
        out_path,
        M              = np.array(Ms,                       dtype=np.float64),
        mean_xL        = np.array([r["mean_xL"]      for r in rows], dtype=np.float64),
        neural_auroc   = np.array([r["neural_auroc"] for r in rows], dtype=np.float64),
        trivial_auroc  = np.array([r["trivial_auroc"]for r in rows], dtype=np.float64),
        gap            = np.array([r["gap"]           for r in rows], dtype=np.float64),
    )
    print(f"\nSaved -> {out_path}")
    print("  Arrays: M, mean_xL, neural_auroc, trivial_auroc, gap")


if __name__ == "__main__":
    main()
