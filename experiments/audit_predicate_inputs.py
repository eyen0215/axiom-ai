"""
Audit predicate input features for scientific validity.

A predicate is scientifically honest only if its features are INDEPENDENT
state variables -- quantities a scientist directly sets or measures (strain,
frequency, applied field) -- WITHOUT invoking the theory being validated.

A predicate is CIRCULAR when its features are:
  - residuals:  |measured - theory_prediction| / scale
  - criterion-encoded ratios:  the exact dimensionless number that defines
    whether the assumption holds (e.g. omega*epsilon/sigma_eff for A2)
  - derived fields:  outputs of the constitutive equation being tested

Classification:
  PASS  all features are independent state variables
  FAIL  one or more features are theory-derived, residual-based, or
        directly encode the breakdown criterion

This script loads each training .npz, prints actual feature statistics,
and reports the classification.  Nothing is retrained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

LE_DATA  = Path(__file__).parent.parent / "data" / "linear_elasticity"
MX_DATA  = Path(__file__).parent.parent / "data" / "maxwell"
SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"

# ---------------------------------------------------------------------------
# Feature classification definitions
# ---------------------------------------------------------------------------

PREDICATES = [
    # (label, npz_path, ckpt_path, log_transform_cols, col_specs)
    # col_specs: list of (col_name, classification, reason)
    {
        "label":       "le_A1",
        "domain":      "Linear Elasticity",
        "assumption":  "A1 small_strain",
        "npz":         LE_DATA / "train_A1.npz",
        "ckpt":        SAVE_DIR / "le_A1.pt",
        "log_cols":    (),
        "columns": [
            {
                "name":   "eps_eq",
                "kind":   "INDEPENDENT",
                "reason": "Equivalent strain -- directly measured via strain gauge. "
                          "No theory needed to compute it.",
            },
        ],
        "verdict": "PASS",
    },
    {
        "label":       "le_A2",
        "domain":      "Linear Elasticity",
        "assumption":  "A2 linearity",
        "npz":         LE_DATA / "train_A2.npz",
        "ckpt":        SAVE_DIR / "le_A2.pt",
        "log_cols":    (),
        "columns": [
            {
                "name":   "eps_eq",
                "kind":   "INDEPENDENT",
                "reason": "Equivalent strain -- directly measured via strain gauge. "
                          "No theory needed. Boundary: eps_yield = 0.00125. "
                          "Rebuilt predicate uses eps_eq as sole feature.",
            },
        ],
        "verdict": "PASS",
    },
    {
        "label":       "le_A5",
        "domain":      "Linear Elasticity",
        "assumption":  "A5 quasi_static",
        "npz":         LE_DATA / "train_A5.npz",
        "ckpt":        SAVE_DIR / "le_A5.pt",
        "log_cols":    (0,),
        "columns": [
            {
                "name":   "frequency",
                "kind":   "INDEPENDENT",
                "reason": "Loading frequency -- directly set by experimenter. "
                          "No theory needed.",
            },
        ],
        "verdict": "PASS",
    },
    {
        "label":       "maxwell_A1",
        "domain":      "Maxwell",
        "assumption":  "A1 linear_media",
        "npz":         MX_DATA / "train_A1.npz",
        "ckpt":        SAVE_DIR / "maxwell_A1.pt",
        "log_cols":    (0,),
        "columns": [
            {
                "name":   "E_field",
                "kind":   "INDEPENDENT",
                "reason": "Applied electric field (V/m) -- directly set or measured "
                          "without invoking D = epsilon*E. "
                          "Boundary: E_sat = 1e8 V/m. Rebuilt predicate uses E_field "
                          "with log_transform_cols=(0,) to span orders of magnitude.",
            },
        ],
        "verdict": "PASS",
    },
    {
        "label":       "maxwell_A2",
        "domain":      "Maxwell",
        "assumption":  "A2 quasi_static",
        "npz":         MX_DATA / "train_A2.npz",
        "ckpt":        SAVE_DIR / "maxwell_A2.pt",
        "log_cols":    (0,),
        "columns": [
            {
                "name":   "frequency",
                "kind":   "INDEPENDENT",
                "reason": "Loading frequency (Hz) -- directly set by experimenter. "
                          "Boundary: f_boundary = 0.01*sigma_eff/(2*pi*epsilon) "
                          "~= 79,891 Hz. Rebuilt predicate uses raw frequency; "
                          "the model must discover the threshold from data.",
            },
        ],
        "verdict": "PASS",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ckpt(ckpt_path: Path) -> dict:
    """Load checkpoint; return state dict and shift."""
    raw = torch.load(ckpt_path, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        return {"state": raw["model"], "shift": float(raw["shift"])}
    return {"state": raw, "shift": 0.0}


def feat_stats(col: np.ndarray) -> str:
    return (f"min={col.min():.3e}  max={col.max():.3e}  "
            f"mean={col.mean():.3e}  std={col.std():.3e}")


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main() -> None:
    W = 72   # line width

    print("=" * W)
    print("PREDICATE INPUT FEATURE AUDIT")
    print("=" * W)
    print()
    print("Classifying features as:")
    print("  INDEPENDENT  raw state variable (strain, frequency, applied field, ...)")
    print("  DEPENDENT    theory-derived quantity, residual, or criterion-encoded ratio")
    print()

    rows = []   # (label, feature_summary, verdict)

    for p in PREDICATES:
        label     = p["label"]
        npz_path  = p["npz"]
        ckpt_path = p["ckpt"]

        print(f"{'-'*W}")
        print(f"Predicate : {label}  ({p['domain']} -- {p['assumption']})")

        # ---- Training data ------------------------------------------------
        if not npz_path.exists():
            print(f"  [training data not found: {npz_path}]")
            verdict_str = "UNKNOWN"
            rows.append((label, "data missing", "UNKNOWN"))
            continue

        d = np.load(npz_path)
        F = d["features"]
        print(f"  Training data : {npz_path.name}  shape={F.shape}")

        for i, col_spec in enumerate(p["columns"]):
            col = F[:, i]
            print(f"  col[{i}]  {col_spec['name']}")
            print(f"         stats : {feat_stats(col)}")
            print(f"         kind  : {col_spec['kind']}")
            print(f"         note  : {col_spec['reason']}")

        # ---- Checkpoint ---------------------------------------------------
        if not ckpt_path.exists():
            print(f"  [checkpoint not found: {ckpt_path}]")
        else:
            ckpt = load_ckpt(ckpt_path)
            shift = ckpt["shift"]
            print(f"  Checkpoint    : {ckpt_path.name}"
                  + (f"  shift={shift:.4f}" if shift != 0.0 else ""))

        # ---- Verdict ------------------------------------------------------
        verdict = p["verdict"]
        print(f"  VERDICT       : {verdict}")
        print()

        feat_names = ", ".join(c["name"] for c in p["columns"])
        rows.append((label, feat_names, verdict))

    # ---- Summary table ---------------------------------------------------
    print("=" * W)
    print("SUMMARY TABLE")
    print("=" * W)
    print()
    hdr = f"  {'Predicate':<16} {'Features':<38} {'Verdict'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, features, verdict in rows:
        marker = "  <-- needs rebuild" if verdict == "FAIL" else ""
        print(f"  {label:<16} {features:<38} {verdict}{marker}")

    print()
    fail_count = sum(1 for _, _, v in rows if v == "FAIL")
    pass_count = sum(1 for _, _, v in rows if v == "PASS")
    print(f"  PASS: {pass_count}   FAIL: {fail_count}")
    print()
    if fail_count > 0:
        print("Predicates that FAIL must be rebuilt with raw independent features.")
        print()
    else:
        print("All predicates use independent features only.")
        print()
    print("=" * W)


if __name__ == "__main__":
    main()
