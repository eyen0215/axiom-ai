# Axiom-AI: Validity-Predicate-Based Theory Breakdown Detection

A neurosymbolic system that detects when a physical theory breaks down in regimes never seen during training — by tracking the validity of explicit physical assumptions rather than flagging anomalies in raw data.

---

## Core Idea

Standard anomaly detectors ask: *"does this data look unusual?"*  
This system asks: *"are the assumptions that underpin this equation still physically valid?"*

Each physical assumption (e.g. "molecules have negligible volume") gets its own **validity predicate** — a small skip-connection MLP trained only on data where that assumption holds. At inference time, the predicate extrapolates: when physical conditions move outside the learned validity region, the predicate fires. Because every derived result tracks which assumptions it depends on (provenance), flagging propagates automatically downstream through the axiom graph.

```text
Physical state
     │
     ▼
┌─────────────────────────────────────────────┐
│  Validity Predicates  (one per assumption)  │
│  A1: score = 0.03 → FLAGGED                 │
│  A2: score = 0.12 → FLAGGED                 │
│  A3: score = 0.91 → ok                      │
│  A4: score = 0.88 → ok                      │
└─────────────────────────────────────────────┘
     │  provenance propagation
     ▼
┌─────────────────────────────────────────────┐
│  Derived results flagged by union rule:     │
│  D2, D4, D5, D6 → SUSPECT                  │
│  D1, D3         → clean                    │
└─────────────────────────────────────────────┘
```

---

## Key Results

Three physical domains tested — all trained exclusively on valid-regime data, evaluated on held-out breakdown regimes.

### Pilot 1 — Ideal Gas (PV = nRT)

| Assumption | Criterion | Recall | AUROC |
| --- | --- | --- | --- |
| A1 Point particles | (V/n)/b > 10 | **1.000** | 1.000 |
| A2 No forces | a·n/(VRT) < 0.10 | **1.000** | 1.000 |

Training: P = 1–10 atm. Held-out: P = 50–200 atm (van der Waals regime). The learned decision boundary aligns with the analytical van der Waals boundary without ever seeing high-pressure data.

### Pilot 2 — Hooke's Law (σ = Eε)

| Assumption | Criterion | Recall | AUROC |
| --- | --- | --- | --- |
| A1 Linearity | stress\_ratio < 0.90 | **1.000** | 1.000 |
| A2 Elasticity | strain\_energy\_ratio < 0.80 | **1.000** | 1.000 |
| A3 Small strain | ε < 0.85 ε\_y | **1.000** | 1.000 |

Training: ε < 0.5 ε\_y (elastic regime). Held-out: ε > 1.5 ε\_y (post-yield). Key finding: feeding all three features to every predicate causes recall to collapse to 0.0 (collinearity sign-flip). Fix: per-assumption feature isolation.

### Pilot 3 — Fourier Heat Conduction (q = −k∇T, Silicon)

| Assumption | Attempt 1 | Attempt 2 | Attempt 3 |
| --- | --- | --- | --- |
| A1 Continuum (Kn < 0.1) | 0.165 | **0.937** | 0.792 |
| A2 Steady-state (Fo > 1) | 0.190 | **0.873** | 0.285 |
| A3 Linear response | 0.273 | **1.000** | **1.000** |
| A4 Local equilibrium (t > 1 ps) | 0.000 | 0.000 | 0.000 |

A4 fails due to a structural calibration bias: training t ∈ [1 ns, 1 s] puts the mean log-criterion at ≈ 22, the combined skip+MLP output is ≈ 12 at the boundary (t = 1 ps) — 12.9× too large to cross the 0.5 threshold.

---

## Architecture

### Validity Predicate (skip-connection MLP)

```text
input features (optionally log-transformed + normalized)
        │
   ┌────┴──────────────────────────────┐
   │  Skip path: Linear(n→1)          │  ← learns linear extrapolation trend
   │  MLP  path: Linear-ReLU-Linear   │  ← learns nonlinear in-distribution residual
   └────┬──────────────────────────────┘
        │  sum
        ▼
     raw logit
        │  sigmoid
        ▼
     score ∈ (0, 1)   >0.5 = valid,  <0.5 = flagged
```

**Why the skip connection is critical:** All training data is valid, so all regression targets are positive. Without the skip, the MLP collapses to a large positive constant (≈ training mean) and never fires on out-of-distribution inputs. The skip's linear extrapolation into the held-out regime is what carries the flag signal.

**Training target:** `log(criterion / threshold)` — positive in training, zero at boundary, negative for violations. Regression on this log-criterion (rather than soft or binary labels) gives non-zero gradients throughout the training range.

**Regularization:** `weight_decay = 0` on skip (free to learn trend), `weight_decay = 5.0` on MLP (kept near zero outside training hull so skip dominates in extrapolation).

### Provenance Propagation

A node is flagged if **any** ancestor assumption fires. Implemented as a static OR over per-assumption flag arrays — no re-traversal of the graph per state.

---

## Failure Modes Discovered

### Collinearity sign-flip (Hooke's Law)

When training features are algebraically related (stress\_ratio ≡ ε/ε\_y, strain\_energy\_ratio ≡ stress\_ratio²), the skip regression has infinitely many solutions. Gradient descent randomly assigns signs; a positive weight on strain\_energy\_ratio produces logits of +628 to +1058 in the held-out regime → recall = 0.000.

**Fix:** Per-assumption feature isolation — each predicate sees only the one feature in its own validity criterion.

### Calibration bias (Fourier A4)

When the training regime is far from the validity boundary in log-criterion space (A4: training t ∈ [1 ns, 1 s], boundary at t = 1 ps, mean log-criterion ≈ 22), the combined skip+MLP output saturates well above 0 everywhere. The predicate cannot fire even though AUROC = 0.999 (it correctly ranks states but threshold is never crossed).

**Diagnosable in advance:** if `mean(log-criterion)` on training data is large (≫ 0), calibration bias is guaranteed. Fix: shrink training regime to bring training-boundary gap below ≈ 5×, or use a per-assumption calibrated threshold.

---

## Project Structure

```text
project/
├── CLAUDE.md                     # project instructions
├── DECISIONS.md                  # architectural decisions log
├── data/
│   └── generate.py               # synthetic data for all three domains
├── axiom_graph/
│   ├── nodes.py                  # AssumptionNode, DerivedNode
│   ├── edges.py                  # Entailment edges
│   └── graph.py                  # AxiomGraph + domain-specific builders
├── validity_predicates/
│   ├── predicate.py              # ValidityPredicate (skip-connection MLP)
│   ├── train.py                  # training loops for all three domains
│   └── evaluate.py               # recall / AUROC evaluation
├── reasoner/
│   ├── forward_chain.py          # run_forward_chain() → ForwardChainResult
│   └── provenance.py             # compute_provenance() → static assumption footprints
├── experiments/
│   ├── pilot.py                  # main 3-domain experiment script
│   ├── plot_pilot1_boundary.py   # Pilot 1 decision boundary figure
│   ├── plot_pilot1_provenance.py # Pilot 1 provenance DAG figure
│   ├── plot_pilot2_boundary.py   # Pilot 2 validity score vs strain
│   ├── plot_pilot2_collinearity.py # Pilot 2 collinearity failure figure
│   ├── plot_pilot3_recall.py     # Pilot 3 grouped recall bar chart
│   ├── plot_pilot3_a4_failure.py # Pilot 3 A4 calibration bias figure
│   └── plot_oneclass_comparison.py # Schematic: anomaly detection vs validity predicate
├── figures/                      # generated PNG outputs (300 dpi)
└── tests/
    └── test_provenance.py
```

---

## Running the Experiments

```bash
# Install dependencies
pip install numpy torch matplotlib pytest sympy

# Run the full 3-domain pilot experiment
python experiments/pilot.py

# Generate individual figures
python experiments/plot_pilot1_boundary.py
python experiments/plot_pilot1_provenance.py
python experiments/plot_pilot2_boundary.py
python experiments/plot_pilot2_collinearity.py
python experiments/plot_pilot3_recall.py
python experiments/plot_pilot3_a4_failure.py
python experiments/plot_oneclass_comparison.py

# Run tests
pytest tests/
```

> **Windows note:** prepend `$env:KMP_DUPLICATE_LIB_OK="TRUE";` to each command if you see an OpenMP DLL conflict.

---

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for full rationale. Summary:

1. **Skip connection is critical** — guarantees linear extrapolation into held-out regimes; plain MLP collapses to training-mean constant.
2. **Log-transform is domain-specific** — ideal gas features (P, V, T, n) need log-transform; Hooke's engineered ratios do not.
3. **Regression on log-criterion, not classification** — avoids zero-gradient problem from saturated sigmoid labels.
4. **Provenance union is sufficient** — flag OR over ancestor assumptions; no re-traversal needed.
5. **Feature isolation prevents collinearity sign-flip** — required when domain features are algebraically dependent.
6. **Per-criterion feature isolation is domain-specific** — Fourier observables are independently sampled; Hooke's features are not.
7. **Calibration bias is diagnosable from training data** — large mean(log-criterion) in training predicts A4-style failure.

---

## Related Work

- **AI-Hilbert** (Cornelio et al., 2023, *Nature Communications*) — finds axioms inconsistent with observed data. We flag axioms that break in unseen regimes; that's the gap.
- **de Kleer & Brown** (1984–87) — ATMS, the conceptual precursor. Discrete/propositional, no learned components. We extend with continuous learned validity.
- **Neural Theorem Provers** (Rocktäschel & Riedel, 2017) — differentiable reasoning over symbolic structures; architectural family reference.
