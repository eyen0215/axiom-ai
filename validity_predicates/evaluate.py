"""
Evaluation metrics for validity predicates on held-out regimes.

Computes, per assumption:
    - Recall    -- primary metric (must not miss genuine breakdowns)
    - Precision
    - F1 score
    - AUROC     -- threshold-free ranking quality

Also reports provenance-propagated flagging at the derived-node level.

Domain configuration is passed as explicit dicts so the same functions
work for both the ideal gas and Hooke's Law domains.

Fits into the system: called from experiments/pilot.py after training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from reasoner.forward_chain import ForwardChainResult
    from reasoner.provenance import ProvenanceRecord

# ---------------------------------------------------------------------------
# Ideal gas domain configuration
# ---------------------------------------------------------------------------

IG_LABEL_COLS: Dict[str, str] = {
    "A1_point_particles": "valid_point_particle",
    "A2_no_forces":       "valid_no_forces",
}

IG_ASSUMPTION_LABELS: Dict[str, str] = {
    "A1_point_particles":    "A1 - Point particles (no molecular volume)",
    "A2_no_forces":          "A2 - No intermolecular forces",
    "A3_elastic_collisions": "A3 - Elastic collisions         [no criterion]",
    "A4_thermal_equilibrium":"A4 - Thermal equilibrium        [no criterion]",
}

IG_DERIVED_ORDER: List[str] = [
    "D1_momentum_transfer",
    "D2_collision_frequency",
    "D3_mean_kinetic_energy",
    "D4_single_particle_pressure",
    "D5_pressure_ideal",
    "D6_ideal_gas_law",
]

IG_NODE_LABELS: Dict[str, str] = {
    "D1_momentum_transfer":       "D1 - Momentum transfer per collision",
    "D2_collision_frequency":     "D2 - Wall collision frequency",
    "D3_mean_kinetic_energy":     "D3 - Mean translational kinetic energy",
    "D4_single_particle_pressure":"D4 - Single-particle pressure",
    "D5_pressure_ideal":          "D5 - Ideal pressure  PV = NkT",
    "D6_ideal_gas_law":           "D6 - Ideal gas law   PV = nRT  [PRIMARY]",
}

IG_ASSUMPTION_ORDER: List[str] = [
    "A1_point_particles", "A2_no_forces",
    "A3_elastic_collisions", "A4_thermal_equilibrium",
]

# ---------------------------------------------------------------------------
# Hooke's Law domain configuration
# ---------------------------------------------------------------------------

HOOKE_LABEL_COLS: Dict[str, str] = {
    "A1_linearity":   "valid_linearity",
    "A2_elasticity":  "valid_elasticity",
    "A3_small_strain":"valid_small_strain",
}

HOOKE_ASSUMPTION_LABELS: Dict[str, str] = {
    "A1_linearity":    "A1 - Linearity (F proportional to x)",
    "A2_elasticity":   "A2 - Elasticity (recoverable deformation)",
    "A3_small_strain": "A3 - Small strain (geometry unchanged)",
    "A4_homogeneity":  "A4 - Homogeneity (uniform material)  [no criterion]",
}

HOOKE_DERIVED_ORDER: List[str] = [
    "D1_linear_response",
    "D2_elastic_stiffness",
    "D3_material_stiffness",
    "D4_hookes_law",
    "D5_deformation",
]

HOOKE_NODE_LABELS: Dict[str, str] = {
    "D1_linear_response":   "D1 - Linear force-displacement response",
    "D2_elastic_stiffness": "D2 - Elastic and constant stiffness",
    "D3_material_stiffness":"D3 - Material stiffness k = EA/L0",
    "D4_hookes_law":        "D4 - Hooke's Law  F = kx       [PRIMARY]",
    "D5_deformation":       "D5 - Deformation  x = FL0/(EA)",
}

HOOKE_ASSUMPTION_ORDER: List[str] = [
    "A1_linearity", "A2_elasticity", "A3_small_strain", "A4_homogeneity",
]

# ---------------------------------------------------------------------------
# Fourier heat conduction domain configuration
# ---------------------------------------------------------------------------

FOURIER_LABEL_COLS: Dict[str, str] = {
    "A1_continuum":          "valid_continuum",
    "A2_steady_state":       "valid_steady_state",
    "A3_linear_response":    "valid_linear_response",
    "A4_local_equilibrium":  "valid_local_equilibrium",
}

FOURIER_ASSUMPTION_LABELS: Dict[str, str] = {
    "A1_continuum":         "A1 - Continuum (Kn = lambda/L < 0.1)",
    "A2_steady_state":      "A2 - Steady state (Fo = alpha*t/L^2 > 1)",
    "A3_linear_response":   "A3 - Linear response (1.65*dT/dx*L/T < 0.1)",
    "A4_local_equilibrium": "A4 - Local equilibrium (t > 1 ps)",
}

FOURIER_DERIVED_ORDER: List[str] = [
    "D1_continuum_field",
    "D2_constitutive_law",
    "D3_heat_equation",
    "D4_fourier_law",
]

FOURIER_NODE_LABELS: Dict[str, str] = {
    "D1_continuum_field":  "D1 - Continuum temperature field",
    "D2_constitutive_law": "D2 - Fourier constitutive law  q = -k*dT/dx",
    "D3_heat_equation":    "D3 - Full heat equation",
    "D4_fourier_law":      "D4 - Fourier's Law (steady state)  [PRIMARY]",
}

FOURIER_ASSUMPTION_ORDER: List[str] = [
    "A1_continuum", "A2_steady_state", "A3_linear_response", "A4_local_equilibrium",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (no sklearn required).

    y_true  : binary (1 = positive = violated)
    y_score : continuous; higher means more likely violated
    """
    pos = y_score[y_true.astype(bool)]
    neg = y_score[~y_true.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"recall": recall, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _should_flag_for_node(
    node_id: str,
    prov_map: Dict[str, "ProvenanceRecord"],
    held_df: pd.DataFrame,
    label_cols: Dict[str, str],
    n: int,
) -> Optional[np.ndarray]:
    """Bool array: states where this node SHOULD be flagged.

    A node should be flagged when at least one ancestor assumption with a
    known ground-truth label is analytically violated.  Returns None if none
    of the node's ancestors have operationalizable criteria.
    """
    if node_id not in prov_map:
        return None
    anc_ids: FrozenSet[str] = prov_map[node_id].assumption_ids
    should = np.zeros(n, dtype=bool)
    has_label = False
    for aid, col in label_cols.items():
        if aid in anc_ids and col in held_df.columns:
            should |= ~held_df[col].values.astype(bool)
            has_label = True
    return should if has_label else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_predicates(
    result: "ForwardChainResult",
    held_df: pd.DataFrame,
    prov_map: Optional[Dict[str, "ProvenanceRecord"]] = None,
    label_cols: Optional[Dict[str, str]] = None,
    derived_order: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-assumption and per-derived-node detection metrics.

    Parameters
    ----------
    result       : output of run_forward_chain()
    held_df      : held-out DataFrame from data.generate.*_dataset()
    prov_map     : provenance map for precise per-node ground truth
    label_cols   : {assumption_id: label_column_name}; defaults to ideal gas
    derived_order: list of derived node IDs to evaluate; defaults to ideal gas

    Returns
    -------
    Nested dict {node_id: {metric_name: value}}.
    """
    if label_cols is None:
        label_cols = IG_LABEL_COLS
    if derived_order is None:
        derived_order = IG_DERIVED_ORDER

    metrics: Dict[str, Dict[str, float]] = {}
    n = result.n_states

    # ------ Per-assumption metrics ----------------------------------------
    for assumption_id, label_col in label_cols.items():
        if label_col not in held_df.columns:
            continue
        if assumption_id not in result.assumption_scores:
            continue

        valid    = held_df[label_col].values.astype(bool)
        violated = ~valid
        scores   = result.assumption_scores[assumption_id]
        flagged  = result.assumption_flagged[assumption_id]

        m = _binary_metrics(violated, flagged)
        m["auroc"]      = _auroc(violated.astype(int), 1.0 - scores.astype(float))
        m["n_total"]    = n
        m["n_violated"] = int(violated.sum())
        m["n_flagged"]  = int(flagged.sum())
        metrics[assumption_id] = m

    # ------ Derived-node flagging -----------------------------------------
    for node_id in derived_order:
        if node_id not in result.node_flagged:
            continue
        flagged_arr = result.node_flagged[node_id]

        if prov_map is not None:
            should = _should_flag_for_node(node_id, prov_map, held_df, label_cols, n)
        else:
            should = np.zeros(n, dtype=bool)
            for _, col in label_cols.items():
                if col in held_df.columns:
                    should |= ~held_df[col].values.astype(bool)

        if should is None:
            metrics[node_id] = {"skip": True}
            continue

        m = _binary_metrics(should, flagged_arr)
        m["n_total"]       = n
        m["n_should_flag"] = int(should.sum())
        m["n_flagged"]     = int(flagged_arr.sum())
        metrics[node_id]   = m

    return metrics


def print_report(
    metrics: Dict[str, Dict[str, float]],
    result: "ForwardChainResult",
    held_df: pd.DataFrame,
    prov_map: Optional[Dict[str, "ProvenanceRecord"]] = None,
    assumption_order: Optional[List[str]] = None,
    assumption_labels: Optional[Dict[str, str]] = None,
    derived_order: Optional[List[str]] = None,
    node_labels: Optional[Dict[str, str]] = None,
    primary_node: Optional[str] = None,
) -> None:
    """Print a formatted breakdown-detection report to stdout.

    All domain-specific display config (labels, orderings, primary node) is
    passed explicitly so the same function works for both domains.
    """
    if assumption_order  is None: assumption_order  = IG_ASSUMPTION_ORDER
    if assumption_labels is None: assumption_labels = IG_ASSUMPTION_LABELS
    if derived_order     is None: derived_order     = IG_DERIVED_ORDER
    if node_labels       is None: node_labels       = IG_NODE_LABELS
    if primary_node      is None: primary_node      = "D6_ideal_gas_law"

    n   = result.n_states
    thr = result.threshold

    # ---- Assumption-level table -----------------------------------------
    print()
    print(f"  Per-assumption breakdown detection  (N={n}, threshold={thr})")
    print("  " + "-" * 74)
    print(f"  {'Assumption':<44} {'Violat':>6} {'Flaggd':>6} "
          f"{'Recall':>7} {'Precis':>7} {'F1':>6} {'AUROC':>7}")
    print("  " + "-" * 74)

    for assumption_id in assumption_order:
        label = assumption_labels.get(assumption_id, assumption_id)
        m     = metrics.get(assumption_id, {})
        if m and not m.get("skip"):
            row = (f"  {label:<44} "
                   f"{m['n_violated']:>6} "
                   f"{m['n_flagged']:>6} "
                   f"{m['recall']:>7.3f} "
                   f"{m['precision']:>7.3f} "
                   f"{m['f1']:>6.3f} "
                   f"{m['auroc']:>7.3f}")
        else:
            row = (f"  {label:<44} "
                   f"{'--':>6} {'--':>6} {'n/a':>7} {'n/a':>7} {'n/a':>6} {'n/a':>7}")
        print(row)
    print("  " + "-" * 74)

    # ---- Derived-node flagging table ------------------------------------
    print()
    print("  Provenance propagation -- derived node flagging")
    print("  " + "-" * 62)
    print(f"  {'Node':<44} {'Should':>6} {'Flaggd':>6} {'Recall':>7}")
    print("  " + "-" * 62)

    for node_id in derived_order:
        label = node_labels.get(node_id, node_id)
        m     = metrics.get(node_id, {})
        if not m or m.get("skip"):
            print(f"  {label:<44} {'--':>6} {'--':>6} {'n/a':>7}  (no criterion in ancestors)")
        else:
            print(f"  {label:<44} "
                  f"{m['n_should_flag']:>6} "
                  f"{m['n_flagged']:>6} "
                  f"{m['recall']:>7.3f}")
    print("  " + "-" * 62)

    # ---- Key result sentence --------------------------------------------
    primary_m = metrics.get(primary_node, {})
    labeled_ms = {
        aid: metrics[aid]
        for aid in assumption_order
        if aid in metrics and not metrics[aid].get("skip")
    }
    print()
    if labeled_ms and primary_m and not primary_m.get("skip"):
        recall_parts = "   ".join(
            f"{aid.split('_')[0]} recall = {m['recall']:.3f}"
            for aid, m in labeled_ms.items()
        )
        print(f"  Key result:  {recall_parts}")
        print(f"    {primary_node} flagging recall = {primary_m['recall']:.3f}")
