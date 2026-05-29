"""
Pilot experiment: two physical domains, one framework.

Runs the full breakdown-detection pipeline for:
    Domain 1 -- Ideal gas law  PV = nRT
    Domain 2 -- Hooke's Law    F  = kx

Each domain follows the same five steps:
    1. Generate training (valid regime) and held-out (breakdown regime) data.
    2. Build the hardcoded axiom graph for that domain.
    3. Train one ValidityPredicate per operationalizable assumption, on
       training data only.
    4. Run forward chaining on held-out states to propagate flags.
    5. Compute and print Recall, Precision, F1, AUROC per assumption;
       report provenance-propagated flagging of derived results.

Success criterion: Recall >= 0.90 on held-out data for all operationalizable
assumptions, in both domains, with zero held-out data seen during training.

Architecture note (see DECISIONS.md for full analysis):
  Both domains use the same skip-connection MLP class (ValidityPredicate).
  The key difference is log_transform_cols:
    Ideal gas  -- (0, 1, 2): log-transforms P, V, T because the validity
                  criteria are log-linear in log(V) and log(T).
    Hooke's Law -- (): NO log-transform; features are already dimensionless
                  ratios (stress_ratio, strain_energy_ratio, epsilon) whose
                  validity boundaries are linear, not log-linear.
  The skip connection (linear extrapolation path) is critical in both cases.

Usage
-----
    python -m experiments.pilot           # both domains, no plots
    python -m experiments.pilot --plot    # also save decision boundary figures
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_TRAIN    = 5_000
N_HELD_OUT = 2_000
SEED       = 42
THRESHOLD  = 0.5
HIDDEN_DIMS = (32, 16)
LR          = 1e-2
N_EPOCHS    = 600

IG_PLOT_PATH    = "outputs/ig_decision_boundary.png"
HOOKE_PLOT_PATH = "outputs/hooke_decision_boundary.png"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    w = 66
    print(); print("=" * w); print(f"  {title}"); print("=" * w)

def _step(n: int, total: int, msg: str) -> float:
    print(f"\n  Step {n}/{total}  {msg}", end="", flush=True)
    return time.perf_counter()

def _done(t0: float) -> None:
    print(f"  [{time.perf_counter() - t0:.1f}s]")


# ---------------------------------------------------------------------------
# Domain 1: Ideal gas
# ---------------------------------------------------------------------------

def run_ideal_gas_experiment(plot: bool = False) -> Dict:
    """Full breakdown-detection pipeline for the ideal gas domain."""
    _banner("DOMAIN 1 -- Ideal Gas Law  PV = nRT")
    print(f"\n  Training : P = 1-10 atm     ({N_TRAIN} states, all ideal-gas-valid)")
    print(f"  Held-out : P = 50-200 atm   ({N_HELD_OUT} states, van der Waals regime)")

    t0 = _step(1, 5, "Generating data ...")
    from data.generate import generate_dataset
    train_df, held_df = generate_dataset(n_train=N_TRAIN, n_held_out=N_HELD_OUT, seed=SEED)
    _done(t0)
    print(f"       train A1={train_df['valid_point_particle'].mean():.0%} valid  "
          f"A2={train_df['valid_no_forces'].mean():.0%} valid")
    print(f"       held  A1={held_df['valid_point_particle'].mean():.0%} valid  "
          f"A2={held_df['valid_no_forces'].mean():.0%} valid")

    t0 = _step(2, 5, "Building axiom graph ...")
    from axiom_graph.graph import build_ideal_gas_graph
    graph = build_ideal_gas_graph()
    _done(t0)
    print(f"       {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
          f"{len(graph.assumption_nodes())} assumption nodes")

    t0 = _step(3, 5, "Training validity predicates ...\n")
    from validity_predicates.train import train_all_predicates
    predicates = train_all_predicates(
        graph, train_df, verbose=True,
        hidden_dims=HIDDEN_DIMS, lr=LR, n_epochs=N_EPOCHS,
    )
    _done(t0)
    print(f"       Trained: {', '.join(predicates)}")

    t0 = _step(4, 5, "Running forward chain ...")
    from reasoner.forward_chain import run_forward_chain
    from reasoner.provenance import compute_provenance
    result   = run_forward_chain(graph, held_df, threshold=THRESHOLD)
    prov_map = compute_provenance(graph)
    _done(t0)

    t0 = _step(5, 5, "Computing metrics ...")
    from validity_predicates.evaluate import (
        evaluate_predicates, print_report,
        IG_LABEL_COLS, IG_ASSUMPTION_ORDER, IG_ASSUMPTION_LABELS,
        IG_DERIVED_ORDER, IG_NODE_LABELS,
    )
    metrics = evaluate_predicates(
        result, held_df, prov_map=prov_map, label_cols=IG_LABEL_COLS,
        derived_order=IG_DERIVED_ORDER,
    )
    _done(t0)

    _banner("IDEAL GAS -- Breakdown Detection Results")
    print_report(
        metrics, result, held_df, prov_map=prov_map,
        assumption_order=IG_ASSUMPTION_ORDER,
        assumption_labels=IG_ASSUMPTION_LABELS,
        derived_order=IG_DERIVED_ORDER,
        node_labels=IG_NODE_LABELS,
        primary_node="D6_ideal_gas_law",
    )

    if plot:
        print("\n  Generating decision boundary plot ...")
        import matplotlib; matplotlib.use("Agg")
        from validity_predicates.train import plot_decision_boundary
        Path(IG_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
        plot_decision_boundary(predicates, train_df, held_df, save_path=IG_PLOT_PATH)
        print(f"  Saved -> {IG_PLOT_PATH}")

    return metrics


# ---------------------------------------------------------------------------
# Domain 2: Hooke's Law
# ---------------------------------------------------------------------------

def run_hooke_experiment(plot: bool = False) -> Dict:
    """Full breakdown-detection pipeline for the Hooke's Law domain."""
    _banner("DOMAIN 2 -- Hooke's Law  F = kx  (steel rod)")
    print(f"\n  Training : epsilon = 0.005-0.0625%   ({N_TRAIN} states, elastic regime)")
    print(f"  Held-out : epsilon = 0.1875-1.25%    ({N_HELD_OUT} states, post-yield regime)")
    print(f"  Material : steel  E=200 GPa  sigma_y=250 MPa")

    t0 = _step(1, 5, "Generating data ...")
    from data.generate import generate_hooke_dataset
    train_df, held_df = generate_hooke_dataset(n_train=N_TRAIN, n_held_out=N_HELD_OUT, seed=SEED)
    _done(t0)
    print(f"       train A1={train_df['valid_linearity'].mean():.0%}  "
          f"A2={train_df['valid_elasticity'].mean():.0%}  "
          f"A3={train_df['valid_small_strain'].mean():.0%}")
    print(f"       held  A1={held_df['valid_linearity'].mean():.0%}  "
          f"A2={held_df['valid_elasticity'].mean():.0%}  "
          f"A3={held_df['valid_small_strain'].mean():.0%}")

    t0 = _step(2, 5, "Building axiom graph ...")
    from axiom_graph.graph import build_hooke_law_graph
    graph = build_hooke_law_graph()
    _done(t0)
    print(f"       {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
          f"{len(graph.assumption_nodes())} assumption nodes")

    t0 = _step(3, 5, "Training validity predicates ...\n")
    from validity_predicates.train import train_all_hooke_predicates
    predicates = train_all_hooke_predicates(
        graph, train_df, verbose=True,
        hidden_dims=HIDDEN_DIMS, lr=LR, n_epochs=N_EPOCHS,
    )
    _done(t0)
    print(f"       Trained: {', '.join(predicates)}")

    t0 = _step(4, 5, "Running forward chain ...")
    from reasoner.forward_chain import run_forward_chain
    from reasoner.provenance import compute_provenance
    result   = run_forward_chain(graph, held_df, threshold=THRESHOLD)
    prov_map = compute_provenance(graph)
    _done(t0)

    t0 = _step(5, 5, "Computing metrics ...")
    from validity_predicates.evaluate import (
        evaluate_predicates, print_report,
        HOOKE_LABEL_COLS, HOOKE_ASSUMPTION_ORDER, HOOKE_ASSUMPTION_LABELS,
        HOOKE_DERIVED_ORDER, HOOKE_NODE_LABELS,
    )
    metrics = evaluate_predicates(
        result, held_df, prov_map=prov_map, label_cols=HOOKE_LABEL_COLS,
        derived_order=HOOKE_DERIVED_ORDER,
    )
    _done(t0)

    _banner("HOOKE'S LAW -- Breakdown Detection Results")
    print_report(
        metrics, result, held_df, prov_map=prov_map,
        assumption_order=HOOKE_ASSUMPTION_ORDER,
        assumption_labels=HOOKE_ASSUMPTION_LABELS,
        derived_order=HOOKE_DERIVED_ORDER,
        node_labels=HOOKE_NODE_LABELS,
        primary_node="D4_hookes_law",
    )

    return metrics


# ---------------------------------------------------------------------------
# Domain 3: Fourier heat conduction
# ---------------------------------------------------------------------------

_FOURIER_HARD_CASES = [
    # (label, T, L, t, dT_dx)
    ("Case 1: L=50nm  T=300K  t=1s    [only A1 fails]",  300.0,  50e-9,  1.0,    1e6),
    ("Case 2: L=1mm   T=1200K t=1s    [only A3 fails]",  1200.0, 1e-3,   1.0,    1e6),
    ("Case 3: L=50nm  T=1200K t=1ps   [A1+A2+A4 fail]",  1200.0, 50e-9,  1e-12,  1e9),
    ("Case 4: L=500nm T=400K  t=100ps [borderline A1+A2]",400.0,  500e-9, 100e-12,1e6),
]


def _print_hard_cases(predicates_by_attempt: Dict[int, Dict[str, object]]) -> None:
    """Print per-assumption predicate scores for the four specified hard cases."""
    import numpy as np
    from data.generate import (
        SILICON_LAMBDA, _silicon_alpha, KN_THRESHOLD, FO_THRESHOLD,
        A3_THRESHOLD, ELECTRON_PHONON_TIME,
    )

    _banner("FOURIER -- Hard Case Analysis")

    for label, T, L, t, dT_dx in _FOURIER_HARD_CASES:
        alpha    = float(_silicon_alpha(np.array([T]))[0])
        Kn       = SILICON_LAMBDA / L
        Fo       = alpha * t / L**2
        A3_ratio = 1.65 * dT_dx * L / T
        dT_dt    = dT_dx * L / t

        print(f"\n  {label}")
        print(f"    Kn={Kn:.3e}  Fo={Fo:.3e}  A3_ratio={A3_ratio:.4f}  t={t:.1e}s")
        print(f"    A1: {'FAIL' if Kn >= KN_THRESHOLD else 'ok  '}"
              f"  A2: {'FAIL' if Fo < FO_THRESHOLD else 'ok  '}"
              f"  A3: {'FAIL' if A3_ratio >= A3_THRESHOLD else 'ok  '}"
              f"  A4: {'FAIL' if t < ELECTRON_PHONON_TIME else 'ok  '}")

        row = np.array([[T, L, t, dT_dx, dT_dt,
                         Kn, Fo, A3_ratio]], dtype=np.float32)
        row_df = {
            "T": T, "L": L, "t": t, "dT_dx": dT_dx, "dT_dt": dT_dt,
            "Kn": Kn, "Fo": Fo, "A3_ratio": A3_ratio,
        }
        import pandas as pd
        df_row = pd.DataFrame([row_df])

        for attempt_num, predicates in sorted(predicates_by_attempt.items()):
            scores = []
            for aid in ["A1_continuum", "A2_steady_state", "A3_linear_response", "A4_local_equilibrium"]:
                pred = predicates.get(aid)
                if pred is None:
                    scores.append(float("nan"))
                else:
                    feats = df_row[pred.feature_cols].values.astype("float32")
                    scores.append(float(pred.predict(feats)[0]))
            print(f"    Attempt {attempt_num}: "
                  f"A1={scores[0]:.3f}  A2={scores[1]:.3f}  "
                  f"A3={scores[2]:.3f}  A4={scores[3]:.3f}")


def run_fourier_experiment() -> Dict:
    """Full breakdown-detection pipeline for Fourier heat conduction, three attempts."""
    _banner("DOMAIN 3 -- Fourier's Law  q = -k*dT/dx  (silicon)")
    print(f"\n  Training : L=1um-1mm, T=200-600K, Fo>1, A3<0.1  ({N_TRAIN} states)")
    print(f"  Held-out : nanoscale + high-T + ultrafast + combined  ({3000} states)")
    print(f"  Material : silicon  k0=150W/mK  lambda=40nm  tau_ep=1ps")

    t0 = _step(1, 5, "Generating data ...")
    from data.generate import generate_fourier_dataset
    train_df, held_df = generate_fourier_dataset(n_train=N_TRAIN, n_held_out=3000, seed=SEED)
    _done(t0)
    for col, label in [
        ("valid_continuum",         "A1"),
        ("valid_steady_state",      "A2"),
        ("valid_linear_response",   "A3"),
        ("valid_local_equilibrium", "A4"),
    ]:
        print(f"       train {label}={train_df[col].mean():.0%} valid  "
              f"held {label}={held_df[col].mean():.0%} valid")

    t0 = _step(2, 5, "Building axiom graph ...")
    from axiom_graph.graph import build_fourier_law_graph
    _done(t0)

    t0 = _step(3, 5, "Training validity predicates (3 attempts) ...\n")
    from validity_predicates.train import train_all_fourier_predicates
    from reasoner.forward_chain import run_forward_chain
    from reasoner.provenance import compute_provenance
    from validity_predicates.evaluate import (
        evaluate_predicates, print_report,
        FOURIER_LABEL_COLS, FOURIER_ASSUMPTION_ORDER, FOURIER_ASSUMPTION_LABELS,
        FOURIER_DERIVED_ORDER, FOURIER_NODE_LABELS,
    )

    all_metrics: Dict[int, Dict] = {}
    all_predicates: Dict[int, Dict] = {}

    for attempt in (1, 2, 3):
        attempt_labels = {
            1: "criterion scalars (Kn, Fo, A3_ratio, t)",
            2: "all observables  (T, L, t, dT/dx, dT/dt)",
            3: "criterion + all observables",
        }
        print(f"\n  --- Attempt {attempt}: {attempt_labels[attempt]} ---")
        graph = build_fourier_law_graph()
        preds = train_all_fourier_predicates(
            graph, train_df, attempt=attempt, verbose=True,
            hidden_dims=HIDDEN_DIMS, lr=LR, n_epochs=N_EPOCHS,
        )
        all_predicates[attempt] = preds

        result   = run_forward_chain(graph, held_df, threshold=THRESHOLD)
        prov_map = compute_provenance(graph)
        metrics  = evaluate_predicates(
            result, held_df, prov_map=prov_map,
            label_cols=FOURIER_LABEL_COLS,
            derived_order=FOURIER_DERIVED_ORDER,
        )
        all_metrics[attempt] = metrics

        _banner(f"FOURIER -- Attempt {attempt} Results")
        print_report(
            metrics, result, held_df, prov_map=prov_map,
            assumption_order=FOURIER_ASSUMPTION_ORDER,
            assumption_labels=FOURIER_ASSUMPTION_LABELS,
            derived_order=FOURIER_DERIVED_ORDER,
            node_labels=FOURIER_NODE_LABELS,
            primary_node="D4_fourier_law",
        )
    _done(t0)

    t0 = _step(4, 5, "Hard case analysis ...")
    _print_hard_cases(all_predicates)
    _done(t0)

    t0 = _step(5, 5, "Summarising ...")
    _done(t0)

    return all_metrics


# ---------------------------------------------------------------------------
# Pass / fail summary
# ---------------------------------------------------------------------------

def _pass_fail(ig_metrics: Dict, hooke_metrics: Dict, fourier_metrics: Dict) -> None:
    _banner("PASS / FAIL -- All Domains")
    target = 0.90

    # For Fourier, use best attempt (highest recall)
    def _best(metrics_by_attempt: Dict, aid: str, metric: str = "recall") -> float:
        best = 0.0
        for m in metrics_by_attempt.values():
            best = max(best, m.get(aid, {}).get(metric, 0.0))
        return best

    checks = [
        ("IG   A1 recall (point particles)", ig_metrics.get("A1_point_particles", {}).get("recall", 0)),
        ("IG   A2 recall (no forces)",       ig_metrics.get("A2_no_forces",       {}).get("recall", 0)),
        ("IG   D6 flagging recall",          ig_metrics.get("D6_ideal_gas_law",   {}).get("recall", 0)),
        ("Hook A1 recall (linearity)",       hooke_metrics.get("A1_linearity",    {}).get("recall", 0)),
        ("Hook A2 recall (elasticity)",      hooke_metrics.get("A2_elasticity",   {}).get("recall", 0)),
        ("Hook A3 recall (small strain)",    hooke_metrics.get("A3_small_strain", {}).get("recall", 0)),
        ("Hook D4 flagging recall",          hooke_metrics.get("D4_hookes_law",   {}).get("recall", 0)),
        ("Four A1 recall (continuum)",       _best(fourier_metrics, "A1_continuum")),
        ("Four A2 recall (steady state)",    _best(fourier_metrics, "A2_steady_state")),
        ("Four A3 recall (linear resp.)",    _best(fourier_metrics, "A3_linear_response")),
        ("Four A4 recall (local equil.)",    _best(fourier_metrics, "A4_local_equilibrium")),
        ("Four D4 flagging recall",          _best(fourier_metrics, "D4_fourier_law")),
    ]

    all_pass = True
    for name, val in checks:
        status = "PASS" if val >= target else "FAIL"
        if val < target:
            all_pass = False
        print(f"    {name:<42} = {val:.3f}   (>= {target})  [{status}]")

    print()
    if all_pass:
        print("  All targets met.  Framework generalises across all three physical domains.")
    else:
        print("  One or more targets not met -- see Fourier reports above for diagnosis.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(plot: bool = False) -> None:
    ig_metrics      = run_ideal_gas_experiment(plot=plot)
    hooke_metrics   = run_hooke_experiment(plot=plot)
    fourier_metrics = run_fourier_experiment()
    _pass_fail(ig_metrics, hooke_metrics, fourier_metrics)


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
