"""
Pipe flow pilot: evaluate trained predicates across three test scenarios
and demonstrate provenance-based trust propagation.

Three assumptions:
    A1 laminar_flow     -- Re > 2300 causes breakdown
    A2 fully_developed  -- x < L_entry causes breakdown
    A3 incompressible   -- v > 102.9 m/s causes breakdown

Two derived quantities (different provenance footprints):
    D1 velocity_profile  <- {A1, A2}
    D2 pressure_drop     <- {A1, A2, A3}

Key discriminative scenario: A3break where only A3 fires (and, physically,
A1 also fires at high v -- documented below rather than masked).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom_graph.pipe_flow_graph import build_pipe_graph
from validity_predicates.predicate import ValidityPredicate

DATA_DIR  = Path("data/pipe_flow")
SAVE_DIR  = Path("validity_predicates/saved")
FIRE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Predicate loading
# ---------------------------------------------------------------------------

def load_predicates() -> tuple[ValidityPredicate, ValidityPredicate, ValidityPredicate]:
    ckpt = torch.load(SAVE_DIR / "pipe_A1.pt", weights_only=False)
    pred_a1 = ValidityPredicate(n_features=4, log_transform_cols=(0, 1, 2, 3))
    pred_a1.load_state_dict(ckpt["state_dict"])
    pred_a1.eval()

    ckpt = torch.load(SAVE_DIR / "pipe_A2.pt", weights_only=False)
    pred_a2 = ValidityPredicate(n_features=5, log_transform_cols=(0, 1, 2, 3, 4))
    pred_a2.load_state_dict(ckpt["state_dict"])
    pred_a2.eval()

    ckpt = torch.load(SAVE_DIR / "pipe_A3.pt", weights_only=False)
    pred_a3 = ValidityPredicate(n_features=1, log_transform_cols=(0,))
    pred_a3.load_state_dict(ckpt["state_dict"])
    pred_a3.eval()

    return pred_a1, pred_a2, pred_a3


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def run_provenance(
    g,
    assumption_fired: dict[str, np.ndarray],
) -> dict[str, tuple[str, float]]:
    """
    For each DerivedNode compute suspect_frac = fraction of samples where
    ANY ancestor assumption fired.  SUSPECT if majority (frac > 0.5).
    Assumptions absent from assumption_fired are treated as silent.
    """
    n = next(iter(assumption_fired.values())).shape[0]
    results: dict[str, tuple[str, float]] = {}
    for node in g.nodes.values():
        if node.kind != "derived":
            continue
        fp = g.ancestor_assumptions(node.id)
        any_fired = np.zeros(n, dtype=bool)
        for aid in fp:
            if aid in assumption_fired:
                any_fired |= assumption_fired[aid]
        frac = float(np.mean(any_fired))
        results[node.id] = ("SUSPECT" if frac > FIRE_THRESHOLD else "TRUSTED", frac)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_bool(arr) -> bool:
    """Convert scalar or per-sample ground-truth array to a single bool."""
    return bool(np.mean(np.atleast_1d(arr).astype(float)) > 0.5)


def fire_label(fire_rate: float) -> str:
    return "FIRED" if fire_rate > FIRE_THRESHOLD else "silent"


# ---------------------------------------------------------------------------
# Scenario table
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("A1break", "A1 fires — Re > 2300 (turbulent)"),
    ("A2break", "A2 fires — x < L_entry (entrance region)"),
    ("A3break", "A3 fires — v > 102.9 m/s (compressible); A1 also fires here"),
]

DERIVED_ORDER = ["D1_velocity_profile", "D2_pressure_drop"]
DERIVED_FOOTPRINT = {
    "D1_velocity_profile": "A1,A2",
    "D2_pressure_drop":    "A1,A2,A3",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    g = build_pipe_graph()
    pred_a1, pred_a2, pred_a3 = load_predicates()

    W = 64
    print("=" * W)
    print("PIPE FLOW PILOT -- Validity Predicate Evaluation")
    print("=" * W)

    for scen_file, scen_desc in SCENARIOS:
        d  = np.load(DATA_DIR / f"test_scenario_{scen_file}.npz")
        N  = len(d["A1_features"])

        gt_a1 = majority_bool(d["a1_breaks"])
        gt_a2 = majority_bool(d["a2_breaks"])
        gt_a3 = majority_bool(d["a3_breaks"])

        sc_a1 = pred_a1.predict(d["A1_features"].astype(np.float32))
        sc_a2 = pred_a2.predict(d["A2_features"].astype(np.float32))
        sc_a3 = pred_a3.predict(d["A3_features"].astype(np.float32))

        fired_a1 = sc_a1 < FIRE_THRESHOLD
        fired_a2 = sc_a2 < FIRE_THRESHOLD
        fired_a3 = sc_a3 < FIRE_THRESHOLD

        fr_a1 = float(np.mean(fired_a1))
        fr_a2 = float(np.mean(fired_a2))
        fr_a3 = float(np.mean(fired_a3))

        assumption_fired = {
            "A1_laminar_flow":    fired_a1,
            "A2_fully_developed": fired_a2,
            "A3_incompressible":  fired_a3,
        }
        prov = run_provenance(g, assumption_fired)

        print(f"\nScenario {scen_file}: {scen_desc}")
        print(f"  N = {N}  |  ground truth: a1={gt_a1}, a2={gt_a2}, a3={gt_a3}")
        print()
        print(f"  {'Predicate':<24} {'score_mean':>10}  {'fire_rate':>9}  status")
        print(f"  A1 laminar_flow:         {np.mean(sc_a1):>10.4f}  {fr_a1*100:>8.1f}%  {fire_label(fr_a1)}")
        print(f"  A2 fully_developed:      {np.mean(sc_a2):>10.4f}  {fr_a2*100:>8.1f}%  {fire_label(fr_a2)}")
        print(f"  A3 incompressible:       {np.mean(sc_a3):>10.4f}  {fr_a3*100:>8.1f}%  {fire_label(fr_a3)}")
        print()
        print(f"  Provenance  (footprint -> status, suspect_frac)")
        for did in DERIVED_ORDER:
            status, frac = prov[did]
            print(f"    {did:<26} [{DERIVED_FOOTPRINT[did]:<8}]  {status:<8}  ({frac:.3f})")

    # -----------------------------------------------------------------------
    # False positive rate on training holdouts (all-valid, target < 5%)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("False positive rate on valid training data (all is_valid=True, n=5000 each)")
    print()

    checks = [
        ("A1 laminar_flow",    pred_a1, "train_A1.npz"),
        ("A2 fully_developed", pred_a2, "train_A2.npz"),
        ("A3 incompressible",  pred_a3, "train_A3.npz"),
    ]
    all_fpr_ok = True
    for label, pred, fname in checks:
        d_tr = np.load(DATA_DIR / fname)
        scores = pred.predict(d_tr["features"].astype(np.float32))
        fpr = float(np.mean(scores < FIRE_THRESHOLD))
        ok  = fpr < 0.05
        if not ok:
            all_fpr_ok = False
        print(f"  {label:<22}  FPR = {fpr*100:5.1f}%  "
              f"{'OK (<5%)' if ok else 'WARNING: exceeds 5%'}")

    print()
    if all_fpr_ok:
        print("  All predicates meet the <5% false-positive target.")
    else:
        print("  WARNING: one or more predicates exceed the 5% FPR target.")
    print("=" * W)


if __name__ == "__main__":
    main()
