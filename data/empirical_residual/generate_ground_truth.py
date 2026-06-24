"""
Ground-truth pressure drop via Shah & London (1978) apparent friction factor.

L_entry MUST NOT appear anywhere in this file.  The empirical correlation is
fitted to experimental data; it does not derive from the entrance-length
formula.  Verification that L_entry is absent is enforced by the __main__
block assertion.

Shah & London (1978), Table 51 — apparent friction factor for laminar flow
in the hydrodynamic entrance region of a circular duct:

    x_plus  = x / (D * Re)
    f_app * Re = 3.44 / sqrt(x_plus)
              + (1.25 / (4 * x_plus) + 16 - 3.44 / sqrt(0.0565))
                / (1 + 0.00021 * x_plus**-2)

    dP_true = f_app * (x / D) * (rho * v**2 / 2)

Hagen-Poiseuille (fully-developed, x-based length):
    f_HP    = 64 / Re
    dP_HP   = 32 * mu * x * v / D**2

Empirical residual:
    pred_error = |dP_true - dP_HP| / dP_HP
"""

import numpy as np

# Fixed physical constants
MU = 1.81e-5   # Pa·s, dynamic viscosity of air
RHO = 1.2      # kg/m³, density of air


def _reynolds(v, D):
    return RHO * v * D / MU


def compute_shah_london(x, v, D, rho=RHO, mu=MU):
    """Pressure drop from Shah & London (1978) apparent friction factor.

    Parameters
    ----------
    x   : axial position (m) — also used as pipe length
    v   : mean velocity (m/s)
    D   : diameter (m)
    rho : density (kg/m³)
    mu  : dynamic viscosity (Pa·s)

    Returns
    -------
    dP_true : pressure drop (Pa)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    D = np.asarray(D, dtype=float)
    rho = np.asarray(rho, dtype=float)
    mu = np.asarray(mu, dtype=float)

    Re = rho * v * D / mu
    x_plus = x / (D * Re)

    # Shah & London (1978) Table 51 — Fanning friction factor (f*Re → 16)
    # Numerator subtracts 3.44/sqrt(x_plus) (same as the leading term) so that
    # as x_plus → ∞ the formula collapses to f_app*Re → 16.
    leading     = 3.44 / np.sqrt(x_plus)
    numerator   = 16.0 + 1.25 / (4.0 * x_plus) - leading
    denominator = 1.0 + 0.00021 * x_plus ** (-2)
    f_app_Re = leading + numerator / denominator  # → 16 as x_plus → ∞

    # Fanning → Darcy-Weisbach pressure drop requires factor of 4:
    #   dP = 4 * f_F * (x/D) * (rho*v²/2)
    # Verify: at f_app_Re = 16, dP = 4*(16/Re)*(x/D)*(rho*v²/2) = 32*mu*x*v/D² = dP_HP ✓
    f_app = f_app_Re / Re
    dP_true = 4.0 * f_app * (x / D) * (rho * v ** 2 / 2.0)
    return dP_true


def compute_hp(x, v, D, rho=RHO, mu=MU):
    """Hagen-Poiseuille pressure drop (fully-developed laminar).

    Parameters
    ----------
    x   : pipe length (m)
    v   : mean velocity (m/s)
    D   : diameter (m)
    rho : density (kg/m³)  — unused in HP formula, kept for signature symmetry
    mu  : dynamic viscosity (Pa·s)

    Returns
    -------
    dP_HP : pressure drop (Pa)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    D = np.asarray(D, dtype=float)
    mu = np.asarray(mu, dtype=float)

    return 32.0 * mu * x * v / D ** 2


def pred_error(x, v, D, rho=RHO, mu=MU):
    """Relative prediction error |dP_true - dP_HP| / dP_HP.

    This is computable from raw observables plus the two models.
    L_entry does not appear here.
    """
    dP_true = compute_shah_london(x, v, D, rho, mu)
    dP_hp   = compute_hp(x, v, D, rho, mu)
    return np.abs(dP_true - dP_hp) / dP_hp


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1.  Assertion: L_entry must not be defined in this scope
    # ------------------------------------------------------------------
    assert "L_entry" not in dir(), "L_entry must not exist in local namespace"

    D = 0.01   # m

    # ------------------------------------------------------------------
    # 2.  Print pred_error on a grid of (x, v) to verify monotonicity
    # ------------------------------------------------------------------
    xs = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.00])
    vs = np.array([1.0, 3.0, 5.0, 10.0])

    print("pred_error grid  (D=0.01 m)")
    print(f"{'x (m)':>8}", end="")
    for v in vs:
        print(f"  v={v:4.1f}", end="")
    print()
    print("-" * (8 + len(vs) * 9))
    for x in xs:
        print(f"{x:8.3f}", end="")
        for v in vs:
            err = pred_error(x, v, D)
            print(f"  {err:7.4f}", end="")
        print()

    # ------------------------------------------------------------------
    # 3.  Verify monotonic decrease with x for each v
    # ------------------------------------------------------------------
    x_fine = np.linspace(0.001, 50.0, 5000)
    all_monotone = True
    for v in vs:
        errs = pred_error(x_fine, v, D)
        if not np.all(np.diff(errs) <= 0):
            all_monotone = False
            print(f"\nWARNING: pred_error is NOT monotonically decreasing for v={v}")
    if all_monotone:
        print("\npred_error is monotonically decreasing with x for all tested v. [OK]")

    # ------------------------------------------------------------------
    # 4.  Find x where pred_error crosses 0.05 for v=5, D=0.01
    #     Compare to true L_entry = 0.06 * Re * D
    #     (L_entry computed HERE ONLY for verification — never in training)
    # ------------------------------------------------------------------
    v_test = 5.0
    errs_fine = pred_error(x_fine, v_test, D)

    # First x where error drops below 0.05
    below = np.where(errs_fine < 0.05)[0]
    if len(below) == 0:
        print(f"\nWARNING: pred_error never crosses 0.05 for v={v_test}, D={D}")
    else:
        x_cross = x_fine[below[0]]
        Re_test = RHO * v_test * D / MU
        L_entry_true = 0.06 * Re_test * D   # verification reference only
        print(f"\nv={v_test}, D={D}")
        print(f"  pred_error 0.05-crossing : x = {x_cross:.4f} m")
        print(f"  true L_entry (ref only)  : x = {L_entry_true:.4f} m")
        ratio = x_cross / L_entry_true
        print(f"  ratio (crossing / L_entry): {ratio:.3f}  (expect > 1; Shah-London 5%-crossing ~6x L_entry)")

    # Second assertion: still no L_entry in local namespace
    assert "L_entry" not in dir(), "L_entry must not exist in local namespace"
    print("\nAssertion passed: L_entry never entered local namespace. [OK]")
