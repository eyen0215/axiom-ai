# Axiom-Based Neurosymbolic Scientific Discovery

## Current status
- [x] Project scaffolded
- [x] data/generate.py complete
- [x] axiom_graph/ 
- [x] reasoner/provenance.py
- [x] validity_predicates/
- [x] experiments/pilot.py — complete (3-domain: ideal gas, Hooke's, Fourier)

Update this section manually at the end of each session.

## What this project is
A system that detects when a physical theory breaks down in regimes 
never seen during training, by tracking the validity of explicit 
physical assumptions rather than flagging anomalies in data.

The core insight: instead of asking "does this data look unusual," 
ask "are the assumptions that underpin this equation still physically 
valid under these conditions?"

## The novel contribution
Validity predicates — small MLPs attached to each physical assumption 
— learn what conditions make that assumption safe, trained only on 
data where assumptions hold. At inference time they extrapolate: when 
physical conditions move outside the learned validity region, the 
predicate fires. Because every derived result tracks which assumptions 
it depends on (provenance), flagging propagates automatically 
downstream.

## Scoped MVP (do this first)
Do NOT try to build the full differentiable end-to-end system yet.
The publishable core result is:
1. Hard-code the ideal gas axiom graph and PV=nRT derivation
2. Train validity predicates on low-pressure data only
3. Show they fire correctly at the van der Waals boundary 
   without ever seeing high-pressure training data

## Target domain: ideal gas → van der Waals
- Train on: P = 1–10 atm, large V, high T (ideal gas regime)
- Withhold: high-pressure data (van der Waals regime)
- Key assumption to flag: "molecules have no volume and don't 
  interact" — operationalized as free volume per molecule
- Success metric: breakdown detection recall on held-out 
  high-pressure test set

## Key assumptions to represent (ideal gas)
1. Point particles (molecules have no volume)
2. No intermolecular forces
3. Elastic collisions only
4. Thermal equilibrium

Each gets its own validity predicate MLP.

## Folder structure

    project/
    ├── CLAUDE.md
    ├── DECISIONS.md
    ├── axiom_graph/
    │   ├── nodes.py
    │   ├── edges.py
    │   └── graph.py
    ├── validity_predicates/
    │   ├── predicate.py
    │   ├── train.py
    │   └── evaluate.py
    ├── reasoner/
    │   ├── forward_chain.py
    │   └── provenance.py
    ├── data/
    │   ├── generate.py
    │   └── regimes.py
    ├── experiments/
    │   └── pilot.py
    └── tests/
        └── test_provenance.py

## Key literature
- AI-Hilbert (2024, Nature Comms) — closest prior work. Finds axioms 
  inconsistent with observed data. We flag axioms that break in 
  unseen regimes. That's the gap.
- de Kleer & Brown (1984–87) — ATMS, the conceptual precursor. 
  Discrete/propositional, no learned components. We extend with 
  continuous learned validity.
- Neural Theorem Provers (Rocktäschel & Riedel 2017) — architectural 
  family for differentiable reasoning over symbolic structures.

## Build order
1. data/generate.py — synthetic PV=nRT data with regime labels
2. axiom_graph/ — hardcoded for ideal gas
3. reasoner/provenance.py — unit test exhaustively before touching neural components
4. validity_predicates/predicate.py + train.py
5. experiments/pilot.py — the held-out test

## Do not build yet
- Learned search (MCTS) over derivations
- Gumbel-softmax / straight-through estimators
- Cross-domain transfer
- GNN over axiom graph

## Rules
- Always read CLAUDE.md and DECISIONS.md at the start of every session
- Always run tests before saying a task is done
- Never modify more than one module at a time unless explicitly told to
- If you are unsure about an architectural decision, stop and ask rather than guess
- Keep functions small and single-purpose
- Every function needs a docstring
- If you make a significant architectural decision, add it to DECISIONS.md

## Mistakes to avoid
(Update this as the project progresses)
- Do not try to make the system end-to-end differentiable yet
- Do not add complexity to the axiom graph before provenance tracking is fully tested

## Code style
- Python only
- Type hints on all functions
- No external dependencies beyond numpy, torch, matplotlib, pytest, sympy
- Prefer clarity over cleverness — this is a research codebase, it needs to be readable