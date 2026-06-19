"""Generate training and test data for the coupled-spring injection-mode track.

System: two-mass coupled spring (non-decomposable by construction).
    wall --[k1]-- m1 --[k_coupling]-- m2 --[k2]-- wall
Parameters: m1=m2=1 kg, k1=k2=1 N/m, k_coupling=0.5 N/m.

Three datasets:
    train_valid.npz      -- undamped, linear, |x|<0.5  (500 traj)
    test_scenario_D.npz  -- viscous damping c in [0.05, 0.5]  (100 traj)
    test_scenario_E.npz  -- cubic-hardening springs beta=0.3, |x| in [1,3]  (100 traj)

Each .npz stores:
    states:  (N_traj, 200, 4) -- [x1, v1, x2, v2] at t = 0, dt, ..., 199*dt
    energy:  (N_traj, 200)    -- linear-energy formula E(t) at every timestep

Note: energy uses the LINEAR formula (0.5*(v1^2+v2^2+x1^2+x2^2) + 0.25*(x1-x2)^2)
for ALL scenarios. For the valid regime this is (approximately) conserved; for
Scenario D it dissipates; for Scenario E it drifts because the true PE contains
quartic terms the formula ignores. This makes energy drift the breakdown signal.

Spec note: dt=0.05 s (user prompt), beta=0.3 (user prompt).
The CLAUDE_INJECTION_MODE.md spec has dt=0.01 and beta=0.5; user values take
precedence. Total trajectory time = 200 * 0.05 = 10.0 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

M1 = 1.0
M2 = 1.0
K1 = 1.0
K2 = 1.0
KC = 0.5   # coupling spring stiffness

BETA = 0.3  # N/m^3, cubic-hardening coefficient (Scenario E)

DT   = 0.05  # s, timestep
T    = 200   # steps per trajectory

N_TRAIN   = 500
N_TEST_D  = 100
N_TEST_E  = 100

OUT_DIR = Path(__file__).parent
RNG = np.random.default_rng(42)

# Integration tolerances: tight to minimise numerical drift in valid regime.
RTOL = 1e-9
ATOL = 1e-9

# ---------------------------------------------------------------------------
# ODE right-hand sides
# ---------------------------------------------------------------------------

def _ode_valid(t, y):
    x1, v1, x2, v2 = y
    a1 = (-(K1 + KC) * x1 + KC * x2) / M1
    a2 = (-(K2 + KC) * x2 + KC * x1) / M2
    return [v1, a1, v2, a2]


def _ode_damped(t, y, c):
    x1, v1, x2, v2 = y
    a1 = (-(K1 + KC) * x1 + KC * x2 - c * v1) / M1
    a2 = (-(K2 + KC) * x2 + KC * x1 - c * v2) / M2
    return [v1, a1, v2, a2]


def _ode_nonlinear(t, y):
    x1, v1, x2, v2 = y
    F1 = -(K1 * x1 + BETA * x1**3) - KC * (x1 - x2)
    F2 = -(K2 * x2 + BETA * x2**3) - KC * (x2 - x1)
    return [v1, F1 / M1, v2, F2 / M2]


# ---------------------------------------------------------------------------
# Energy — linear formula, used for ALL scenarios
# ---------------------------------------------------------------------------

def linear_energy(states: np.ndarray) -> np.ndarray:
    """
    Compute mechanical energy using the linear (valid-regime) formula.
    states: shape (..., 4) with columns [x1, v1, x2, v2].
    Returns: shape (...,).

    Scenario D: this quantity decreases (damping dissipates energy).
    Scenario E: this quantity drifts (quartic PE term missing from formula).
    Both produce a nonzero energy drift, which is the breakdown signal.
    """
    x1, v1, x2, v2 = (states[..., i] for i in range(4))
    KE = 0.5 * M1 * v1**2 + 0.5 * M2 * v2**2
    PE = 0.5 * K1 * x1**2 + 0.5 * K2 * x2**2 + 0.5 * KC * (x1 - x2)**2
    return KE + PE


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

_T_EVAL = np.arange(T) * DT  # t = 0, dt, 2*dt, ..., (T-1)*dt — shape (200,)


def integrate(ode_fn, y0: list[float]) -> np.ndarray:
    """
    Integrate ode_fn from y0 at the fixed t_eval grid.
    Returns states shape (T, 4) in float64.
    """
    sol = solve_ivp(
        ode_fn,
        t_span=(0.0, _T_EVAL[-1]),
        y0=y0,
        method='RK45',
        t_eval=_T_EVAL,
        rtol=RTOL,
        atol=ATOL,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y.T  # (T, 4)


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def generate_valid() -> dict:
    """Valid regime: undamped linear springs, small displacement."""
    states_all = np.empty((N_TRAIN, T, 4), dtype=np.float32)
    energy_all = np.empty((N_TRAIN, T),    dtype=np.float64)

    for i in range(N_TRAIN):
        y0 = [RNG.uniform(-0.5, 0.5) for _ in range(4)]
        s  = integrate(_ode_valid, y0)
        states_all[i] = s.astype(np.float32)
        energy_all[i] = linear_energy(s)
        if (i + 1) % 100 == 0:
            print(f"  valid  {i+1}/{N_TRAIN}")

    return {"states": states_all, "energy": energy_all.astype(np.float32)}


def generate_scenario_d() -> dict:
    """Scenario D: viscous damping, c drawn per trajectory from [0.05, 0.5]."""
    states_all = np.empty((N_TEST_D, T, 4), dtype=np.float32)
    energy_all = np.empty((N_TEST_D, T),    dtype=np.float64)
    c_values   = RNG.uniform(0.05, 0.5, N_TEST_D)

    for i in range(N_TEST_D):
        y0 = [RNG.uniform(-0.5, 0.5) for _ in range(4)]
        c  = float(c_values[i])
        s  = integrate(lambda t, y, c=c: _ode_damped(t, y, c), y0)
        states_all[i] = s.astype(np.float32)
        energy_all[i] = linear_energy(s)
        if (i + 1) % 50 == 0:
            print(f"  scenario_D  {i+1}/{N_TEST_D}")

    return {
        "states":   states_all,
        "energy":   energy_all.astype(np.float32),
        "c_values": c_values.astype(np.float32),
    }


def generate_scenario_e() -> dict:
    """Scenario E: cubic-hardening springs, large displacement (|x| in [1, 3])."""
    states_all = np.empty((N_TEST_E, T, 4), dtype=np.float32)
    energy_all = np.empty((N_TEST_E, T),    dtype=np.float64)

    for i in range(N_TEST_E):
        # Large-displacement ICs with random signs to explore all quadrants.
        s1 = float(RNG.choice([-1, 1]))
        s2 = float(RNG.choice([-1, 1]))
        y0 = [
            s1 * float(RNG.uniform(1.0, 3.0)),  # x1
            float(RNG.uniform(-0.5, 0.5)),        # v1
            s2 * float(RNG.uniform(1.0, 3.0)),  # x2
            float(RNG.uniform(-0.5, 0.5)),        # v2
        ]
        s = integrate(_ode_nonlinear, y0)
        states_all[i] = s.astype(np.float32)
        energy_all[i] = linear_energy(s)
        if (i + 1) % 50 == 0:
            print(f"  scenario_E  {i+1}/{N_TEST_E}")

    return {"states": states_all, "energy": energy_all.astype(np.float32)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _drift_stats(energy: np.ndarray) -> dict:
    """Compute per-trajectory max and mean energy drift from t=0."""
    E0        = energy[:, 0:1]              # (N, 1)
    drift     = np.abs(energy - E0)         # (N, T)
    max_drift = drift.max(axis=1)           # (N,)
    mean_drift = drift.mean(axis=1)         # (N,)
    return {
        "max_drift_mean": float(max_drift.mean()),
        "max_drift_std":  float(max_drift.std()),
        "max_drift_p95":  float(np.percentile(max_drift, 95)),
        "max_drift_max":  float(max_drift.max()),
        "mean_drift_mean": float(mean_drift.mean()),
        "E0_mean":         float(E0.mean()),
        "E0_std":          float(E0.std()),
    }


def print_stats(label: str, data: dict) -> None:
    e  = data["energy"]
    st = _drift_stats(e)
    print(f"\n  {label}")
    print(f"    Initial energy: mean={st['E0_mean']:.4f}  std={st['E0_std']:.4f}")
    print(f"    Max drift / traj:")
    print(f"      mean = {st['max_drift_mean']:.4e}")
    print(f"      std  = {st['max_drift_std']:.4e}")
    print(f"      p95  = {st['max_drift_p95']:.4e}")
    print(f"      max  = {st['max_drift_max']:.4e}")
    print(f"    Mean drift / traj:  {st['mean_drift_mean']:.4e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    W = 65
    print("=" * W)
    print("COUPLED SPRING DATA GENERATION")
    print("=" * W)
    print(f"  m1={M1}, m2={M2}, k1={K1}, k2={K2}, k_coupling={KC}")
    print(f"  dt={DT}s, T={T} steps, span={T*DT:.1f}s")
    print(f"  beta={BETA} (Scenario E), rtol={RTOL}, atol={ATOL}")
    print(f"  Output: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print(f"\n{'='*W}")
    print("1. Valid training regime  (undamped, linear, |x|<0.5)")
    print(f"{'='*W}")
    valid = generate_valid()
    path  = OUT_DIR / "train_valid.npz"
    np.savez(path, **valid)
    print(f"\n  Saved {path.name}:")
    print(f"    states shape: {valid['states'].shape}")
    print(f"    energy shape: {valid['energy'].shape}")
    print(f"    file size:    {path.stat().st_size // 1024} KB")
    print_stats(
        "Energy drift (NOISE FLOOR — pure numerical integration error):",
        valid,
    )

    # ------------------------------------------------------------------
    print(f"\n{'='*W}")
    print("2. Scenario D — damped  (c in [0.05, 0.5])")
    print(f"{'='*W}")
    scen_d = generate_scenario_d()
    path   = OUT_DIR / "test_scenario_D.npz"
    np.savez(path, **scen_d)
    print(f"\n  Saved {path.name}:")
    print(f"    states shape:  {scen_d['states'].shape}")
    print(f"    energy shape:  {scen_d['energy'].shape}")
    print(f"    c_values:      range=[{scen_d['c_values'].min():.3f}, {scen_d['c_values'].max():.3f}]")
    print_stats("Energy drift (expected: large, energy dissipates):", scen_d)

    # ------------------------------------------------------------------
    print(f"\n{'='*W}")
    print(f"3. Scenario E — nonlinear springs  (beta={BETA}, |x| in [1, 3])")
    print(f"{'='*W}")
    scen_e = generate_scenario_e()
    path   = OUT_DIR / "test_scenario_E.npz"
    np.savez(path, **scen_e)
    print(f"\n  Saved {path.name}:")
    print(f"    states shape: {scen_e['states'].shape}")
    print(f"    energy shape: {scen_e['energy'].shape}")
    print_stats(
        "Energy drift (expected: large, linear E formula misses quartic PE):",
        scen_e,
    )

    # ------------------------------------------------------------------
    print(f"\n{'='*W}")
    print("VERIFICATION")
    print(f"{'='*W}")
    for fname in ["train_valid.npz", "test_scenario_D.npz", "test_scenario_E.npz"]:
        p = OUT_DIR / fname
        d = np.load(p)
        keys  = sorted(d.keys())
        size  = p.stat().st_size // 1024
        shapes = {k: d[k].shape for k in keys}
        ok    = all(
            d[k].shape[0] in (N_TRAIN, N_TEST_D, N_TEST_E)
            and d[k].shape[1:] == (T, 4) if k == "states" else True
            for k in ["states"]
        )
        status = "OK" if ok else "SHAPE MISMATCH"
        print(f"  {fname:<30}  {size:>5} KB  keys={keys}  {status}")

    # ------------------------------------------------------------------
    # Noise-floor summary and detection guidance
    valid_drift = _drift_stats(valid["energy"])
    d_drift     = _drift_stats(scen_d["energy"])
    e_drift     = _drift_stats(scen_e["energy"])

    print(f"\n{'='*W}")
    print("ENERGY DRIFT SUMMARY")
    print(f"{'='*W}")
    print(f"\n  {'Dataset':<24}  {'max_drift mean':>14}  {'max_drift p95':>14}")
    print(f"  {'-'*56}")
    print(f"  {'train_valid (noise floor)':<24}  {valid_drift['max_drift_mean']:>14.4e}  "
          f"{valid_drift['max_drift_p95']:>14.4e}")
    print(f"  {'test_scenario_D':<24}  {d_drift['max_drift_mean']:>14.4e}  "
          f"{d_drift['max_drift_p95']:>14.4e}")
    print(f"  {'test_scenario_E':<24}  {e_drift['max_drift_mean']:>14.4e}  "
          f"{e_drift['max_drift_p95']:>14.4e}")
    print()
    threshold = valid_drift['max_drift_mean'] + 3.0 * valid_drift['max_drift_std']
    print(f"  Suggested detection threshold (mean + 3*std on valid):")
    print(f"    threshold = {threshold:.4e}")
    print()
    print(f"  Scenario D max_drift / threshold = "
          f"{d_drift['max_drift_mean'] / (threshold + 1e-30):.1f}x")
    print(f"  Scenario E max_drift / threshold = "
          f"{e_drift['max_drift_mean'] / (threshold + 1e-30):.1f}x")
    print(f"\n{'='*W}")


if __name__ == "__main__":
    main()
