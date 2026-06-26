"""
Physics formulas for the Dittus-Boelter heat transfer domain.
This is the ONLY place these formulas appear; all data generators import from here.
"""

import numpy as np


def nu_dittus_boelter(Re, Pr):
    """
    Dittus-Boelter equation (heating case).
    Valid approximately for Re > 10000, 0.6 < Pr < 160.
    Nu = 0.023 * Re^0.8 * Pr^0.4
    """
    return 0.023 * Re**0.8 * Pr**0.4


def nu_gnielinski(Re, Pr):
    """
    Gnielinski correlation -- higher fidelity reference.
    f = (0.790 * ln(Re) - 1.64)^(-2)
    Nu = (f/8) * (Re - 1000) * Pr
         / (1 + 12.7 * sqrt(f/8) * (Pr^(2/3) - 1))
    Valid for Re in [3000, 5e6], Pr in [0.5, 2000].
    Returns np.nan for Re < 3000 (outside valid range).
    """
    Re = np.asarray(Re, dtype=float)
    Pr = np.asarray(Pr, dtype=float)
    f = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))
    Nu = np.where(Re < 3000, np.nan, Nu)
    return Nu


def pred_error(Re, Pr):
    """
    Fractional prediction error between DB and Gnielinski.
    pred_error = |Nu_DB - Nu_G| / Nu_G
    Returns np.nan where Gnielinski is invalid.
    """
    Nu_DB = nu_dittus_boelter(Re, Pr)
    Nu_G = nu_gnielinski(Re, Pr)
    return np.abs(Nu_DB - Nu_G) / Nu_G


if __name__ == "__main__":
    Re_vals = [5000, 10000, 50000, 100000]
    Pr_vals = [0.3, 1.0, 6.2, 50, 200]

    # 1. Print pred_error grid
    header = f"{'Re \\ Pr':>12}" + "".join(f"{pr:>10}" for pr in Pr_vals)
    print(header)
    print("-" * (12 + 10 * len(Pr_vals)))
    for Re in Re_vals:
        row = f"{Re:>12}"
        for Pr in Pr_vals:
            err = pred_error(Re, Pr)
            row += f"{'nan':>10}" if np.isnan(err) else f"{err:>10.4f}"
        print(row)

    print()

    # 2. Verify pred_error is small near Re=10000, Pr=6.2 (the two correlations
    #    happen to agree well right at the turbulent threshold).
    e_10k = pred_error(10000, 6.2)
    status = "PASS" if e_10k < 0.05 else "FAIL"
    print(f"[{status}] pred_error(Re=10000, Pr=6.2) = {e_10k:.4f}  (expect < 0.05, correlations agree at threshold)")

    # 3. Verify pred_error is large for Re=5000, Pr=6.2 (sub-turbulent).
    e_low_Re = pred_error(5000, 6.2)
    status = "PASS" if e_low_Re > 0.10 else "FAIL"
    print(f"[{status}] pred_error(Re=5000,  Pr=6.2) = {e_low_Re:.4f}  (expect > 0.10, sub-turbulent)")

    # Note: error is also large at high Re (50000+), meaning DB and Gnielinski
    # diverge on BOTH sides of Re~10000. The error is U-shaped in Re at Pr=6.2.
    e_50k = pred_error(50000, 6.2)
    print(f"NOTE:  pred_error(Re=50000, Pr=6.2) = {e_50k:.4f}  (also large -- DB under-predicts Gnielinski at high Re)")

    print()

    # 4. Bisection on the low-Re side (3001 to 10000) where error is high at
    #    low Re and drops near Re=10000.  This captures the A1 transition.
    Pr_fixed = 6.2
    target = 0.05
    lo, hi = 3001.0, 10000.0  # error is large at lo, small at hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if pred_error(mid, Pr_fixed) > target:
            lo = mid
        else:
            hi = mid
    Re_lo_threshold = (lo + hi) / 2

    # Also find the high-Re crossing (10000 to 100000).
    lo2, hi2 = 10000.0, 100000.0  # error is small at lo2, large at hi2
    for _ in range(60):
        mid = (lo2 + hi2) / 2
        if pred_error(mid, Pr_fixed) < target:
            lo2 = mid
        else:
            hi2 = mid
    Re_hi_threshold = (lo2 + hi2) / 2

    print(f"A1 (low-Re) crossing at Pr={Pr_fixed}: Re ~= {Re_lo_threshold:.1f}")
    print(f"  pred_error at threshold: {pred_error(Re_lo_threshold, Pr_fixed):.5f}")
    print(f"High-Re crossing at Pr={Pr_fixed}:    Re ~= {Re_hi_threshold:.1f}")
    print(f"  pred_error at threshold: {pred_error(Re_hi_threshold, Pr_fixed):.5f}")
    print()
    print("Finding: pred_error is U-shaped in Re at Pr=6.2.")
    print("  DB and Gnielinski agree tightly near Re=10000 and diverge on both sides.")
    print("  The A1 boundary (turbulent threshold) sits near Re~10000 on the low side.")
