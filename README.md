# PRISM: Prospective Regime Inference via Symbolic Modeling

A neurosymbolic system that detects when a physical theory breaks down in regimes never seen during training — by learning from the theory's own residual rather than from labeled breakdown events.

---

## Core Idea

Standard anomaly detectors ask: *"does this data look unusual?"*  
This system asks: *"is the theory's residual growing, and where?"*

Each physical theory (ideal gas, Hooke's law, Fourier conduction) gets a **residual-based validity predicate** — a skip-connection MLP trained on raw observables to predict the log theory residual. The model never sees breakdown criteria, correction parameters, or high-pressure data. It must learn from low-residual training states and extrapolate to flag high-residual held-out regimes.

The longer-term goal is a three-stage loop:

1. **Stage 1 — Detect:** Learn where the base theory's residual is growing, from raw observables, no criterion given.  
2. **Stage 2 — Discover:** In the flagged region, use sparse regression to discover the correction term (e.g., van der Waals a, b terms). The Stage 1 flag is used to target a small additional sample for the regression, tested against baselines for sample efficiency.  
3. **Stage 3 — Recurse:** Apply Stage 1 to the corrected model to find its own new validity boundary.

Current status: Stage 1 complete; Stage 2 attempted once (failure documented).

---

## Architecture

### Residual Predicate (skip-connection MLP)

```text
raw observables [P, V, T, n]
        │  log-transform + normalize
        ▼
   ┌────┴───────────────────────────────┐
   │  Skip path: Linear(4 → 1)         │  ← learns linear extrapolation trend
   │  MLP  path: Linear-ReLU-...-Linear│  ← learns nonlinear in-distribution fit
   └────┬───────────────────────────────┘
        │  sum
        ▼
   log(residual)   [regression target, no sigmoid]
```

**Why the skip connection is critical:** All training data is in the valid regime, so the MLP collapses to a constant. The skip's linear extrapolation beyond the training hull is what carries the breakdown signal.

**Training target:** `log(|theory residual|)` — positive/small in training, grows as theory breaks down. MSE regression; no breakdown criterion baked in.

**Regularization:** `weight_decay = 0` on skip (free to learn trend), `weight_decay = 5.0` on MLP (kept near zero OOD so skip dominates in extrapolation).

**Calibration:** detection threshold = training mean + 3σ of log(residual). Anything above fires the predicate.

---

## Results

### Stage 1 — Ideal Gas Residual Predicate (Pilot 1 Reformulation)

Trained on (P, V, T, n) from the CO2 van der Waals EOS at P = 1–10 atm. Target: log(|PV/nRT − 1|). Not given: a, b, or any breakdown criterion.

Evaluated on P = 50–200 atm (never seen in training):

| Metric | Value | Note |
| ------ | ----- | ---- |
| AUROC | **0.999** | vs 0.921 for pressure-only linear baseline |
| Recall | **1.000** | all 948/1000 broken states detected |
| Precision | 0.948 | at calibrated 3σ threshold |
| Pearson r | 0.963 | correct trend extrapolated |
| R² | −4.06 | systematic +1.25 log-unit upward bias (characterized) |

The model learned that higher P / smaller V / lower T predicts larger residual — physically correct, from raw observables, with no hints.

### Stage 1 — Pilots 1–3 (original, criterion-based predicates)

The original predicate formulation used `log(criterion/threshold)` as the training target, which encodes the known validity boundary directly. This gives perfect results by construction (Recall = 1.000, AUROC = 1.000 across all valid assumptions in all three domains) but is circular: the model evaluates a formula, not discovers a relationship.

These results are kept as architecture validation — they confirm the skip+MLP design extrapolates correctly when the boundary is known.

| Domain | Assumptions | Recall | AUROC |
| ------ | ----------- | ------ | ----- |
| Ideal gas (PV = nRT) | A1 point particles, A2 no forces | 1.000 | 1.000 |
| Hooke's law (σ = Eε) | A1 linearity, A2 elasticity, A3 small strain | 1.000 | 1.000 |
| Fourier conduction (q = −k∇T) | A1 continuum, A2 steady-state, A3 linear response | 0.873–1.000 | — |
| Fourier conduction | A4 local equilibrium (t > 1 ps) | 0.000 | 0.999 |

Fourier A4 fails because the calibration bias is too large: training t ∈ [1 ns, 1 s] puts the skip output far above zero at the boundary (t = 1 ps). AUROC = 0.999 confirms correct ranking; threshold is never crossed.

### Stage 2 — Correction-Term Discovery (Attempt 1, failed)

STLSQ sparse regression on an 11-term library to recover the vdW correction r ≈ (b − a/RT)·(n/V) + b²·(n/V)² from data. Three conditions compared (A: low-P only, B: full range, C: low-P + 50 Stage-1-targeted points).

| Condition | Hi-P pts | Terms selected | Test R² |
| --------- | -------- | -------------- | ------- |
| A (P = 1–10) | 0 | n/V, (n/V)², (n/V)³, P/T, 1/T, n/T | −10,282,988 |
| B (P = 1–200) | 1000 | n/V, (n/V)², P/T, 1/T, n/T | +0.970 |
| C (P = 1–10 + 50 targeted) | 50 | n/V, (n/V)², P/T, 1/T, n/T | −15.025 |

**Root cause:** In vdW-generated data, V is smooth in P and T, so n/V ≈ P/(RT). Pearson r(n/V, P/T) = 0.9999 in training. OLS cannot distinguish them; STLSQ retains a spurious linear combination that fits in-distribution but catastrophically extrapolates. Condition number of the column-normalized 11-term matrix: 1615. Removing P/T and P*V from the library drops the condition number to 212.

The Stage-1 targeting idea is not falsified — the failure is in library collinearity, not in the targeting mechanism.

Full diagnosis: [STAGE2_ATTEMPT1_SUMMARY.md](STAGE2_ATTEMPT1_SUMMARY.md).

---

## Failure Modes

### Collinearity sign-flip (Hooke's, Criterion-based)

When training features are algebraically related (stress_ratio ≡ ε/ε_y exactly), the skip regression has infinitely many solutions. A random positive weight on a correlated feature produces logits of +628 to +1058 OOD → recall = 0.  
**Fix:** Per-assumption feature isolation — each predicate sees only its own criterion feature.

### Calibration bias (Fourier A4)

When the training-to-boundary gap in log-criterion space is large (1000× for A4), the skip output never crosses zero at the boundary.  
**Diagnosable in advance:** if mean(log-criterion) ≫ 0 on training data, the predicate cannot fire. Fix: shrink training regime, or use a calibrated per-assumption threshold.

### Library collinearity (Stage 2)

When training data is generated by a smooth EOS, physically distinct library terms become numerically indistinguishable. OLS distributes coefficient mass arbitrarily; STLSQ cannot recover the sparse true solution regardless of threshold or sample size.  
**Proposed fix:** dimensionless library terms that are physically orthogonal by construction.

---

## Project Structure

```text
project/
├── CLAUDE.md                            # project instructions (start here)
├── STATUS.md                            # current state, open problem, next step
├── DECISIONS.md                         # architectural decisions (7 entries)
├── RESULTS_LOG.md                       # dated experiment records
├── STAGE2_ATTEMPT1_SUMMARY.md           # Stage 2 collinearity failure analysis
├── data/
│   ├── generate.py                      # ideal gas, Hooke's, Fourier data generators
│   └── generate_vdw_residual.py         # PR-EOS ground truth, vdW residual target
├── axiom_graph/                         # provenance tracking (existing, validated)
├── reasoner/                            # forward chain + provenance (existing, validated)
├── validity_predicates/
│   ├── predicate.py                     # original criterion-based ValidityPredicate
│   ├── residual_predicate.py            # ResidualPredicate (Stage 1, no criterion)
│   ├── train_residual.py                # training loop for residual predicate
│   └── evaluate_residual.py            # evaluation + calibrated AUROC
├── symbolic_regression/
│   ├── library.py                       # 10-term candidate feature library
│   └── sparse_regression.py            # STLSQ implementation
├── experiments/
│   ├── pilot.py                         # 3-domain pilot (criterion-based)
│   ├── stage2_comparison.py            # A/B/C condition comparison
│   ├── train_vdw_predicate.py          # Stage 1 on PR-EOS data
│   └── plot_pilot*.py                  # figures for Pilots 1–3
└── tests/
    └── test_provenance.py
```

---

## Running Things

```bash
# Stage 1 — residual predicate (ideal gas)
python validity_predicates/train_residual.py
python validity_predicates/evaluate_residual.py

# Stage 2 — sparse correction-term recovery
python experiments/stage2_comparison.py

# Stage 1 on PR-EOS ground truth (CO2, near-critical held-out)
python experiments/train_vdw_predicate.py

# Original 3-domain pilot (criterion-based, for architecture reference)
python experiments/pilot.py
```

> **Windows:** prepend `$env:KMP_DUPLICATE_LIB_OK="TRUE";` if you see an OpenMP DLL conflict.

---

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for full rationale. Summary:

1. **Skip connection is critical** — guarantees linear extrapolation OOD; plain MLP collapses to training-mean constant.
2. **Log-transform is domain-specific** — ideal gas raw features need log-transform; Hooke's engineered ratios do not.
3. **Regression on log(residual), not classification** — avoids zero-gradient problem from saturated sigmoid labels.
4. **Provenance union is sufficient** — flag propagation is OR over ancestor assumptions; no re-traversal needed.
5. **Feature isolation prevents collinearity sign-flip** — required when domain features are algebraically dependent.
6. **Per-criterion isolation is not universal** — Fourier observables are independently sampled; Hooke's are not.
7. **Calibration bias is diagnosable from training data** — large mean(log-criterion) predicts A4-style failure.

---

## Related Work

- **AI-Hilbert** (Cornelio et al., 2023, *Nature Communications*) — finds axioms inconsistent with observed data. We flag axioms that break in unseen regimes from the residual alone; that is the gap.
- **de Kleer & Brown** (1984–87) — ATMS, conceptual precursor. Discrete/propositional, no learned components. We extend with continuous learned validity.
- **SINDy** (Brunton et al., 2016) — sparse regression for equation discovery from data. Stage 2 uses STLSQ from this framework for correction-term discovery.
- **Neural Theorem Provers** (Rocktäschel & Riedel, 2017) — differentiable reasoning over symbolic structures; architectural family reference.
