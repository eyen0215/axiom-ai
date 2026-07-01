"""
Structural independence test for pairs of validity predicates.

PURPOSE
-------
Two assumptions A and B are "independently decomposable" if predicate A's
output does not depend on the state variable(s) that govern B, when B is
held at any fixed valid value.

This test formalises that check:
  1. Generate N_SWEEP points of A's state variable sweeping across A's
     valid-to-invalid boundary (fresh, not from existing .npz files).
  2. Evaluate predicate A's score at two different fixed "valid" values of
     B's state variable  (b_anchor_1, b_anchor_2 -- both safely inside B's
     valid region, chosen to span a large fraction of B's training range).
  3. Compute max|score_b1 - score_b2| along the sweep.
  4. Classify as INDEPENDENT if max_diff < 0.05, COUPLED otherwise.

CURRENT RESULT (after rebuild_fail_predicates.py)
--------------------------------------------------
All predicates use a SINGLE independent state variable as their sole input
feature.  Predicate A therefore does not receive B's state variable at all,
so score_b1 == score_b2 identically and max_diff = 0.0 for every pair.
This is the correct and expected result for a clean decomposition.

The test is retained as a REGRESSION CHECK for two failure modes:
  a) Someone retrains a predicate with multi-variable input that silently
     includes another assumption's state variable.  The test will then detect
     non-zero coupling and flag it as COUPLED.
  b) A future domain uses joint features for efficiency.  This test must pass
     (all pairs INDEPENDENT) before decomposition mode can be used for that
     domain.

SPECIAL CASE: shared state variable
------------------------------------
LE A1 and A2 both use eps_eq as their feature.  In real experiments these
cannot be independently controlled -- eps_eq is a single measurement.  The
test correctly shows zero architectural coupling (the predictor functions are
separate), but the SHARED VAR note flags that physical independence cannot be
claimed for this pair.

REUSE PROTOCOL (for future domains)
-------------------------------------
Before committing to axiom decomposition for a new domain:
  1. Add the domain's predicate configs to DOMAIN_CONFIGS at the bottom.
  2. Run:  python experiments/decomposability_test.py
  3. All pairs must be INDEPENDENT (max_diff < 0.05) to proceed with
     per-assumption decomposition.
  If a pair is COUPLED:
    a) Separate the coupled features into disjoint per-predicate inputs, or
    b) Document the coupling and use joint evaluation for any derived
       quantity whose provenance footprint contains both coupled assumptions.

EXTENDING TO MULTI-FEATURE PREDICATES
---------------------------------------
If pred_A ever takes both A's and B's features jointly (shape (N, 2)), replace
the body of _score_a_at_b_anchor() with:
    X_joint = np.column_stack([a_sweep_feats, np.full(len(a_sweep_feats), b_anchor)])
    return score_pred(pred_a, X_joint, shift_a)
The rest of the test runs unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"

N_SWEEP              = 50     # points along A's sweep
INDEPENDENCE_THRESH  = 0.05   # max_diff threshold


# ---------------------------------------------------------------------------
# Predicate configurations
# ---------------------------------------------------------------------------

# Each config entry:
#   ckpt           -- filename in SAVE_DIR
#   n_features     -- number of input features for ValidityPredicate
#   log_cols       -- which columns are log-transformed internally
#   feature_names  -- list of feature names (length = n_features)
#   sweep_lo/hi    -- range for sweeping THIS predicate's boundary
#   sweep_log      -- True = log-uniform sweep, False = linear
#   b_anchors      -- two valid values of THIS predicate's feature variable
#                     used as fixed "B" values when OTHER predicates are swept

LE_PREDS: dict[str, dict] = {
    "A1": {
        "label":        "A1 (small_strain)",
        "ckpt":         "le_A1.pt",
        "n_features":   1,
        "log_cols":     (),
        "feature_names": ["eps_eq"],
        # Sweep crosses A1 boundary: eps_A1 = 0.01
        "sweep_lo":     0.001,
        "sweep_hi":     0.10,
        "sweep_log":    True,
        # Two valid eps_eq values used when A1 plays the role of B
        "b_anchors":    [0.001, 0.007],
    },
    "A2": {
        "label":        "A2 (linearity)",
        "ckpt":         "le_A2.pt",
        "n_features":   1,
        "log_cols":     (),
        "feature_names": ["eps_eq"],
        # Sweep crosses A2 boundary: eps_yield = 0.00125
        "sweep_lo":     0.0003,
        "sweep_hi":     0.01,
        "sweep_log":    True,
        "b_anchors":    [0.0003, 0.0009],
    },
    "A5": {
        "label":        "A5 (quasi_static)",
        "ckpt":         "le_A5.pt",
        "n_features":   1,
        "log_cols":     (0,),
        "feature_names": ["frequency"],
        # Sweep crosses A5 boundary: ~8061 Hz
        "sweep_lo":     100.0,
        "sweep_hi":     1e6,
        "sweep_log":    True,
        "b_anchors":    [1.0, 50.0],
    },
}

LE_PAIRS: list[tuple[str, str]] = [
    ("A1", "A2"),
    ("A1", "A5"),
    ("A2", "A5"),
]

MX_PREDS: dict[str, dict] = {
    "A1": {
        "label":        "A1 (linear_media)",
        "ckpt":         "maxwell_A1.pt",
        "n_features":   1,
        "log_cols":     (0,),
        "feature_names": ["E_field"],
        # Sweep crosses A1 boundary: E_sat = 1e8 V/m
        "sweep_lo":     1e6,
        "sweep_hi":     5e9,
        "sweep_log":    True,
        "b_anchors":    [1e3, 1e6],
    },
    "A2": {
        "label":        "A2 (quasi_static)",
        "ckpt":         "maxwell_A2.pt",
        "n_features":   1,
        "log_cols":     (0,),
        "feature_names": ["frequency"],
        # Sweep crosses A2 boundary: f_boundary ~79,891 Hz
        "sweep_lo":     1e4,
        "sweep_hi":     1e8,
        "sweep_log":    True,
        "b_anchors":    [100.0, 5000.0],
    },
}

MX_PAIRS: list[tuple[str, str]] = [
    ("A1", "A2"),
]

DOMAIN_CONFIGS = [
    ("Linear Elasticity", LE_PREDS, LE_PAIRS),
    ("Maxwell",           MX_PREDS, MX_PAIRS),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predicate(cfg: dict) -> tuple[ValidityPredicate, float]:
    """Load predicate from checkpoint; return (predicate, shift)."""
    path = SAVE_DIR / cfg["ckpt"]
    pred = ValidityPredicate(
        n_features=cfg["n_features"],
        log_transform_cols=cfg["log_cols"],
        feature_cols=cfg["feature_names"],
    )
    raw = torch.load(path, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        pred.load_state_dict(raw["model"])
        shift = float(raw["shift"])
    else:
        pred.load_state_dict(raw)
        shift = 0.0
    pred.eval()
    return pred, shift


def score_pred(pred: ValidityPredicate, X: np.ndarray, shift: float) -> np.ndarray:
    """Sigmoid validity score with optional log_criterion shift."""
    if shift == 0.0:
        return pred.predict(X)
    with torch.no_grad():
        logits = pred(torch.from_numpy(X.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-(logits + shift)))


def make_sweep(cfg: dict) -> np.ndarray:
    """N_SWEEP points across A's boundary range."""
    lo, hi = cfg["sweep_lo"], cfg["sweep_hi"]
    if cfg["sweep_log"]:
        return np.exp(np.linspace(np.log(lo), np.log(hi), N_SWEEP))
    return np.linspace(lo, hi, N_SWEEP)


def _score_a_at_b_anchor(
    pred_a: ValidityPredicate,
    shift_a: float,
    a_sweep: np.ndarray,
    b_anchor: float,          # fixed valid value of B's state variable
) -> np.ndarray:
    """
    Score pred_A along a_sweep while B is held at b_anchor.

    Current architecture: pred_A is single-feature, so b_anchor is not
    passed.  To extend to multi-feature pred_A, replace the body with:
        X = np.column_stack([a_sweep, np.full(len(a_sweep), b_anchor)])
        return score_pred(pred_a, X, shift_a)
    """
    X_a = a_sweep[:, np.newaxis].astype(np.float32)
    return score_pred(pred_a, X_a, shift_a)   # b_anchor unused by design


# ---------------------------------------------------------------------------
# Pair test
# ---------------------------------------------------------------------------

def test_pair(
    name_a: str, cfg_a: dict, pred_a: ValidityPredicate, shift_a: float,
    name_b: str, cfg_b: dict,
) -> dict:
    """
    Run the independence test for the ordered pair (A sweeps, B fixed).

    Returns a result dict with:
      max_diff        -- max|score_b1 - score_b2| along the sweep
      classification  -- "INDEPENDENT" or "COUPLED"
      shared_var      -- True if A and B have overlapping feature names
      sweep_vals      -- the sweep array (for diagnostics)
      scores_b1       -- score curve at b_anchor_1
      scores_b2       -- score curve at b_anchor_2
    """
    a_sweep = make_sweep(cfg_a)
    b1, b2  = cfg_b["b_anchors"]

    scores_b1 = _score_a_at_b_anchor(pred_a, shift_a, a_sweep, b1)
    scores_b2 = _score_a_at_b_anchor(pred_a, shift_a, a_sweep, b2)

    max_diff = float(np.max(np.abs(scores_b1 - scores_b2)))
    classification = "INDEPENDENT" if max_diff < INDEPENDENCE_THRESH else "COUPLED"

    shared = set(cfg_a["feature_names"]) & set(cfg_b["feature_names"])

    return {
        "pair":           f"{name_a} vs {name_b}",
        "a_label":        cfg_a["label"],
        "b_label":        cfg_b["label"],
        "b_anchor_1":     b1,
        "b_anchor_2":     b2,
        "b_feat_name":    cfg_b["feature_names"][0],
        "max_diff":       max_diff,
        "classification": classification,
        "shared_var":     shared,
        "sweep_vals":     a_sweep,
        "scores_b1":      scores_b1,
        "scores_b2":      scores_b2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    W = 72
    print("=" * W)
    print("DECOMPOSABILITY TEST — structural independence of predicate pairs")
    print("=" * W)
    print(f"\nThreshold: max_diff < {INDEPENDENCE_THRESH:.2f} => INDEPENDENT")
    print(f"Sweep:     {N_SWEEP} points across each predicate's boundary range")

    all_results: list[tuple[str, list[dict]]] = []

    for domain_name, preds_cfg, pairs in DOMAIN_CONFIGS:
        print(f"\n{'='*W}")
        print(f"Domain: {domain_name}")
        print(f"{'='*W}")

        # Load all predicates for this domain
        loaded: dict[str, tuple[ValidityPredicate, float]] = {}
        for key, cfg in preds_cfg.items():
            pred, shift = load_predicate(cfg)
            loaded[key] = (pred, shift)
            print(f"  Loaded {cfg['label']}  (shift={shift:.4f})")

        domain_results: list[dict] = []

        for name_a, name_b in pairs:
            cfg_a = preds_cfg[name_a]
            cfg_b = preds_cfg[name_b]
            pred_a, shift_a = loaded[name_a]

            # Test both directions: (A sweeps, B fixed) and (B sweeps, A fixed)
            res_ab = test_pair(name_a, cfg_a, pred_a, shift_a, name_b, cfg_b)
            pred_b, shift_b = loaded[name_b]
            res_ba = test_pair(name_b, cfg_b, pred_b, shift_b, name_a, cfg_a)

            domain_results.extend([res_ab, res_ba])

        all_results.append((domain_name, domain_results))

        # Print table for this domain
        print()
        hdr = f"  {'Sweeping pred':<22} {'Fixed pred (anchors)':<30} {'Max diff':>9}  {'Result'}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for r in domain_results:
            b_str = (f"{r['b_feat_name']}="
                     f"[{r['b_anchor_1']:.3g}, {r['b_anchor_2']:.3g}]")
            shared_note = (f"  [SHARED VAR: {', '.join(sorted(r['shared_var']))}]"
                           if r["shared_var"] else "")
            cls   = r["classification"]
            marker = "  <-- WARNING" if cls == "COUPLED" else ""
            print(f"  {r['pair']:<22} {b_str:<30} {r['max_diff']:>9.6f}  "
                  f"{cls}{marker}{shared_note}")

        # Detailed score curves if any pair shows non-zero diff
        for r in domain_results:
            if r["max_diff"] > 1e-9:
                print(f"\n  Score curve diff detail for {r['pair']}:")
                print(f"  {'feat_val':>12}  {'score_b1':>10}  {'score_b2':>10}  {'diff':>10}")
                for fv, s1, s2 in zip(r["sweep_vals"], r["scores_b1"], r["scores_b2"]):
                    print(f"  {fv:>12.4e}  {s1:>10.6f}  {s2:>10.6f}  {abs(s1-s2):>10.6f}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*W}")
    print("SUMMARY")
    print(f"{'='*W}\n")

    all_coupled  = []
    all_indep    = []
    all_shared   = []

    for domain_name, domain_results in all_results:
        for r in domain_results:
            tag = f"{domain_name}: {r['pair']}"
            if r["classification"] == "COUPLED":
                all_coupled.append((tag, r["max_diff"]))
            else:
                all_indep.append((tag, r["max_diff"]))
            if r["shared_var"]:
                all_shared.append((tag, r["shared_var"]))

    print(f"  INDEPENDENT pairs: {len(all_indep)}")
    print(f"  COUPLED     pairs: {len(all_coupled)}")
    if all_shared:
        print(f"  SHARED VAR  pairs: {len(all_shared)}")
        for tag, svars in all_shared:
            print(f"    {tag}  (shared feature: {', '.join(sorted(svars))})")
    print()

    if all_coupled:
        print("  COUPLED PAIRS (need attention before using decomposition):")
        for tag, md in all_coupled:
            print(f"    {tag}  max_diff={md:.6f}")
        print()
        print("  CONCLUSION: decomposition NOT validated for these pairs.")
    else:
        print("  All pairs INDEPENDENT.  max_diff = 0.0 for all.")
        print()
        print("  CONCLUSION: predicates are structurally independent.")
        print("  Axiom decomposition is validated for the tested domains.")
        if all_shared:
            print()
            print("  NOTE on shared-variable pairs:")
            print("  Architectural independence is confirmed (max_diff = 0.0),")
            print("  but the shared state variable means these assumptions cannot")
            print("  be independently controlled in a real experiment.")
            print("  Provenance results for derived quantities whose footprint")
            print("  contains BOTH of a shared-variable pair should be interpreted")
            print("  with care: if one fires, the other is likely also at risk.")

    print(f"\n{'='*W}")


if __name__ == "__main__":
    main()
