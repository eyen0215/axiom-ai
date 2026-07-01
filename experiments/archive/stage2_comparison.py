"""
Stage 2 — Targeted Correction-Term Discovery: Condition A / B / C comparison.

Three STLSQ sparse-regression fits on the same candidate library, differing
only in which data is used for fitting:

  A  Training only          P = 1-10 atm      N = 2000   (0 high-P samples)
  B  Training + full HO     P = 1-200 atm     N = 3000   (1000 high-P samples)
  C  Training + 50 targeted P = 1-10 + 50-80  N = 2050   (50 high-P samples)

Evaluation: fresh full-range test set (P = 1-200 atm, N = 500, seed = 99)
            never seen by any condition during fitting.

True van der Waals correction (CO2, a=3.592, b=0.04267, R=0.08206):
  r = PV/nRT - 1 = b*rho/(1-b*rho) - a*rho/(RT)     rho = n/V
  Leading terms:
    coeff of n/V    = b - a/(RT)   varies -0.103 (T=300K) to -0.045 (T=500K)
    coeff of (n/V)^2 = b^2        = 0.001821  (T-independent)

Usage:
    python experiments/stage2_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate import generate_ideal_gas_residual_data
from symbolic_regression.library import build_library
from symbolic_regression.sparse_regression import stlsq

R       = 0.08206
A_CO2   = 3.592
B_CO2   = 0.04267
THRESHOLD = 0.001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signed_residual(data: dict) -> np.ndarray:
    return (data["P"] * data["V"]) / (data["n"] * R * data["T"]) - 1.0


def make_theta(data: dict):
    lib   = build_library(data["P"], data["V"], data["T"], data["n"])
    names = list(lib.keys())
    return np.column_stack(list(lib.values())), names


def fit_condition(data: dict, label: str) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    r     = signed_residual(data)
    Theta, names = make_theta(data)
    result = stlsq(Theta, r, names, threshold=THRESHOLD)
    return result, names, Theta, r


def test_r2(result: dict, names: list[str], test_data: dict) -> float:
    r_test = signed_residual(test_data)
    Theta_test, _ = make_theta(test_data)
    if not result:
        return float("nan")
    cols   = [names.index(k) for k in result]
    coefs  = np.array([result[k] for k in result])
    y_pred = Theta_test[:, cols] @ coefs
    ss_res = ((r_test - y_pred) ** 2).sum()
    ss_tot = ((r_test - r_test.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

T_FILTER: tuple[float, float] | None = (390.0, 410.0)  # set None to use all T


def filter_T(data: dict, T_range: tuple[float, float]) -> dict:
    """Keep only rows where T_range[0] <= T <= T_range[1]."""
    mask = (data["T"] >= T_range[0]) & (data["T"] <= T_range[1])
    return {k: v[mask] for k, v in data.items()}


def main():
    # ---- Load / generate data ----------------------------------------------
    train_raw = dict(np.load("data/ideal_gas_residual_train.npz"))
    test  = dict(np.load("data/ideal_gas_residual_test.npz"))

    if T_FILTER is not None:
        train = filter_T(train_raw, T_FILTER)
        print(f"T filter [{T_FILTER[0]}, {T_FILTER[1]}] K: "
              f"{len(train_raw['P'])} -> {len(train['P'])} training points")
    else:
        train = train_raw

    targeted = generate_ideal_gas_residual_data(
        n_samples=50, P_range=(50, 80), seed=42
    )

    # Fresh full-range evaluation set (never used in any fit)
    fresh_test = generate_ideal_gas_residual_data(
        n_samples=500, P_range=(1, 200), seed=99
    )

    def concat(d1, d2):
        return {k: np.concatenate([d1[k], d2[k]]) for k in d1}

    cond_A_data = train
    cond_B_data = concat(train, test)
    cond_C_data = concat(train, targeted)

    # ---- Fit each condition ------------------------------------------------
    res_A, names, _, _ = fit_condition(cond_A_data, "A")
    res_B, _,     _, _ = fit_condition(cond_B_data, "B")
    res_C, _,     _, _ = fit_condition(cond_C_data, "C")

    r2_A = test_r2(res_A, names, fresh_test)
    r2_B = test_r2(res_B, names, fresh_test)
    r2_C = test_r2(res_C, names, fresh_test)

    # ---- Print STAGE2_SPEC comparison table --------------------------------
    n_hiP = {"A": 0, "B": len(test["P"]), "C": len(targeted["P"])}
    n_tot  = {"A": len(train["P"]), "B": len(train["P"]) + len(test["P"]),
              "C": len(train["P"]) + len(targeted["P"])}

    true_coefs = {
        "n/V":     "b-a/RT  (-0.103 to -0.045)",
        "(n/V)^2": f"b^2 = {B_CO2**2:.6f}",
    }

    results  = {"A": res_A, "B": res_B, "C": res_C}
    r2s      = {"A": r2_A,  "B": r2_B,  "C": r2_C}

    print("=" * 80)
    print("Stage 2 — Targeted Correction-Term Discovery (Ideal Gas, CO2 vdW)")
    print("=" * 80)
    print(f"\nLibrary threshold : {THRESHOLD}")
    print(f"Evaluation set    : N=500, P=1-200 atm, seed=99 (not used in any fit)\n")

    print(f"{'Term':<12} {'True coef':<28} {'Cond A':>11} {'Cond B':>11} {'Cond C':>11}")
    print(f"{'':12} {'':28} {'(P=1-10)':>11} {'(P=1-200)':>11} {'(1-10+50)':>11}")
    print("-" * 75)
    for term in names:
        true = true_coefs.get(term, "0  (distractor)")
        vals = []
        for cond in ("A", "B", "C"):
            v = results[cond].get(term)
            vals.append(f"{v:>+.4f}" if v is not None else f"{'—':>11}")
        print(f"{term:<12} {true:<28} {vals[0]:>11} {vals[1]:>11} {vals[2]:>11}")

    print("-" * 75)
    print(f"{'Test R²':<12} {'(fresh P=1-200)':<28} {r2_A:>+11.4f} {r2_B:>+11.4f} {r2_C:>+11.4f}")
    print(f"{'N total':<12} {'':28} {n_tot['A']:>11} {n_tot['B']:>11} {n_tot['C']:>11}")
    print(f"{'N hi-P pts':<12} {'':28} {n_hiP['A']:>11} {n_hiP['B']:>11} {n_hiP['C']:>11}")
    print("=" * 80)

    print("\nSelected terms summary:")
    for cond, res in results.items():
        terms_str = ", ".join(f"{t}:{v:+.4f}" for t, v in res.items())
        print(f"  Cond {cond} ({len(res)} terms): {terms_str}")

    # ---- Print DECISIONS.md entry ------------------------------------------
    print()
    print("=" * 80)
    print("DECISIONS.md ENTRY (append this):")
    print("=" * 80)

    # Gather selected-terms strings for the table
    def terms_str(res):
        if not res:
            return "none"
        return ", ".join(f"{t}({v:+.4f})" for t, v in res.items())

    entry = f"""
---

## 2026-06-12 — Stage 2: Targeted Correction-Term Discovery (Ideal Gas)

### Setup

Sparse regression (STLSQ, threshold={THRESHOLD}) on 11-term candidate library
[{", ".join(names)}]
to recover correction terms for r = PV/nRT - 1, comparing three data conditions
(A: low-P only, B: full range, C: low-P + 50 Stage-1-targeted points).

Evaluation: fresh full-range test set (N=500, P=1-200 atm, seed=99).

True van der Waals correction (CO2, a={A_CO2}, b={B_CO2}, R={R}):
  Leading terms: (b - a/RT)*(n/V) + b^2*(n/V)^2
  coeff n/V     = b - a/(RT)  varies {B_CO2 - A_CO2/(R*300):.4f} (T=300K) to {B_CO2 - A_CO2/(R*500):.4f} (T=500K)
  coeff (n/V)^2 = b^2        = {B_CO2**2:.6f}  (T-independent)

### Results

| Condition | Hi-P samples | Terms recovered | Test R² |
|---|---|---|---|
| A (P=1-10)        | {n_hiP['A']:4d} | {terms_str(res_A)} | {r2_A:+.4f} |
| B (P=1-200)       | {n_hiP['B']:4d} | {terms_str(res_B)} | {r2_B:+.4f} |
| C (P=1-10 + 50)   | {n_hiP['C']:4d} | {terms_str(res_C)} | {r2_C:+.4f} |

### Interpretation: Failure — multicollinearity prevents term isolation

All three conditions fail to recover just the two true terms (n/V, (n/V)^2).
Instead, all select 5-6 terms including spurious distractors P/T, 1/T, and n/T.

Root cause: in the vdW-generated data, V is determined by the cubic EOS, which
is smooth in P and T. This means n/V ~ P/(RT) holds approximately throughout,
making n/V and P/T nearly collinear across all pressure regimes. OLS distributes
coefficient mass across them arbitrarily; STLSQ's thresholding retains the
spurious combination.

Condition C does improve over A: n/V coefficient moves from -3.78 (A) toward
-0.44 (C) vs -0.15 (B), and (n/V)^2 flips to the correct positive sign.
But the 50 targeted points are insufficient to break the collinearity.

The Stage-1-targeting idea is not falsified by this result — the failure is in
the sparse regression step (collinear library on smooth EOS data), not in the
targeting. But Stage 2 as designed cannot demonstrate sample efficiency when
the library terms are not identifiable from data at any sample size.

### What to do next (options, not yet implemented)

1. Column-normalize the library matrix before STLSQ (threshold applies to
   standardised coefs; rescale back). This may separate n/V from P/T if the
   variation in their collinearity differs across pressure ranges.
2. Add n/(V*T) explicitly to the library to absorb the T-dependent part of
   the n/V coefficient; then n/V should carry only the b term.
3. Fix T (single temperature slice) to eliminate the T-dependent coefficient
   and make n/V and (n/V)^2 the only relevant terms — cleaner but less general.
4. Use a physics-aware library that includes only dimensionless combinations.
"""
    print(entry)
    return entry


if __name__ == "__main__":
    main()
