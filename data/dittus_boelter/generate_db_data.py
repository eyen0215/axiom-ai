"""
Data generator for Dittus-Boelter predicates A1, A2, A3.

Training files: valid regime, is_valid=True, log_criterion as soft label.
Test files: one assumption broken, two held valid.

Feature sets (only raw physical variables, never Re/Pr/Nu/L/D):
  A1: [v, D, rho, mu]
  A2: [mu, cp, k]
  A3: [L, D]

C_entrance note: spec body says 6.0 but the inline example
(pred_error > 0.6/10 = 0.06 at L/D=10, threshold at L/D~12)
is only consistent with C_entrance=0.6, which is used here.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import pred_error as phys_pred_error

SAVE_DIR = os.path.dirname(__file__)

# ── Physical constants ────────────────────────────────────────────────────────
RHO = 1000.0    # kg/m^3
MU  = 8.9e-4    # Pa*s
CP  = 4182.0    # J/kg/K
K   = 0.6       # W/m/K
D   = 0.02      # m

C_ENTRANCE = 0.6   # entrance correction factor; 0.6 not 6.0 — see module docstring

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_Re(v, rho=RHO, d=D, mu=MU):
    return rho * v * d / mu

def get_Pr(mu_val, cp_val, k_val):
    return mu_val * cp_val / k_val

def log_crit(err):
    """log(0.05 / (err + eps)), clipped to [-20, 20]."""
    return np.clip(np.log(0.05 / (np.asarray(err, float) + 1e-10)), -20.0, 20.0)

def pred_error_A3(L, d=D):
    """Fractional error from entrance effects: (D/L)*C_entrance."""
    return (d / np.asarray(L, float)) * C_ENTRANCE

def log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FILES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_train_A1(n=5000, seed=42):
    """
    Valid regime: Re in [12000, 80000], Pr=6.2 (fixed).
    Features: [v, D, rho, mu].
    """
    rng = np.random.default_rng(seed)
    Re = log_uniform(rng, 12000, 80000, n)
    v  = Re * MU / (RHO * D)

    X = np.column_stack([v, np.full(n, D), np.full(n, RHO), np.full(n, MU)])

    Pr  = MU * CP / K   # 6.2, fixed
    err = phys_pred_error(Re, Pr)
    lc  = log_crit(err)

    path = os.path.join(SAVE_DIR, "train_A1.npz")
    np.savez(path, X=X, is_valid=np.ones(n, bool), log_criterion=lc,
             Re=Re, Pr=np.full(n, Pr))

    print("=== train_A1.npz ===")
    print(f"  n={n}, X.shape={X.shape}, features=[v, D, rho, mu]")
    print(f"  Re range: [{Re.min():.0f}, {Re.max():.0f}]")
    print(f"  mean pred_error: {err.mean():.4f}")
    print(f"  mean log_criterion: {lc.mean():.3f}")
    print()


def generate_train_A2(n=5000, seed=43):
    """
    Valid regime: Pr in [2.0, 80.0], Re=50000 (fixed).
    Features: [mu, cp, k].
    Pr is U-shaped — both low and high Pr are invalid.
    Training range is well inside [0.6, 160].
    """
    rng = np.random.default_rng(seed)
    Pr  = log_uniform(rng, 2.0, 80.0, n)
    mu  = Pr * K / CP    # mu = Pr * k / cp

    X = np.column_stack([mu, np.full(n, CP), np.full(n, K)])

    Re  = 50000.0
    err = phys_pred_error(Re, Pr)
    lc  = log_crit(err)

    # Boundary-check diagnostics requested by caller
    err_Pr2  = float(phys_pred_error(Re, 2.0))
    err_Pr80 = float(phys_pred_error(Re, 80.0))

    path = os.path.join(SAVE_DIR, "train_A2.npz")
    np.savez(path, X=X, is_valid=np.ones(n, bool), log_criterion=lc,
             Re=np.full(n, Re), Pr=Pr)

    print("=== train_A2.npz ===")
    print(f"  n={n}, X.shape={X.shape}, features=[mu, cp, k]")
    print(f"  Pr range: [{Pr.min():.2f}, {Pr.max():.2f}]")
    print(f"  mean pred_error: {err.mean():.4f}")
    print(f"  mean log_criterion: {lc.mean():.3f}")
    print(f"  pred_error at Pr=2.0:  {err_Pr2:.4f}  (expect small -- inside valid range)")
    print(f"  pred_error at Pr=80.0: {err_Pr80:.4f}  (expect small -- inside valid range)")
    print()


def generate_train_A3(n=5000, seed=44):
    """
    Valid regime: L/D in [15, 100], i.e. L in [0.3, 2.0] m.
    Features: [L, D].
    pred_error_A3 = (D/L) * C_entrance.
    """
    rng = np.random.default_rng(seed)
    L = log_uniform(rng, 0.3, 2.0, n)   # D=0.02, so L/D in [15, 100]

    X = np.column_stack([L, np.full(n, D)])

    err = pred_error_A3(L)
    lc  = log_crit(err)

    LD = L / D
    path = os.path.join(SAVE_DIR, "train_A3.npz")
    np.savez(path, X=X, is_valid=np.ones(n, bool), log_criterion=lc, LD=LD)

    print("=== train_A3.npz ===")
    print(f"  n={n}, X.shape={X.shape}, features=[L, D]")
    print(f"  L/D range: [{LD.min():.1f}, {LD.max():.1f}]")
    print(f"  mean pred_error_A3: {err.mean():.4f}")
    print(f"  mean log_criterion: {lc.mean():.3f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FILES  (all three feature sets stored; each predicate reads its own)
# ═══════════════════════════════════════════════════════════════════════════════

def _baseline_A2_features(n):
    """A2 features for water-like fluid (Pr=6.2): [mu, cp, k] all fixed."""
    return np.column_stack([np.full(n, MU), np.full(n, CP), np.full(n, K)])

def _baseline_A3_features(n, LD=50.0):
    """A3 features for well-developed flow: [L, D], L/D=50."""
    return np.column_stack([np.full(n, LD * D), np.full(n, D)])

def _baseline_A1_features(n, Re_fixed=50000.0):
    """A1 features for well-turbulent flow: [v, D, rho, mu], Re=50000."""
    v = Re_fixed * MU / (RHO * D)
    return np.column_stack([np.full(n, v), np.full(n, D), np.full(n, RHO), np.full(n, MU)])


def generate_test_A1break(n=1000, seed=50):
    """
    A1 breaks: Re in [3000, 8000] (laminar/transitional).
    A2, A3 remain valid (Pr=6.2, L/D=50).
    """
    rng = np.random.default_rng(seed)
    Re  = log_uniform(rng, 3000, 8000, n)
    v   = Re * MU / (RHO * D)

    X_A1 = np.column_stack([v, np.full(n, D), np.full(n, RHO), np.full(n, MU)])
    X_A2 = _baseline_A2_features(n)
    X_A3 = _baseline_A3_features(n)

    path = os.path.join(SAVE_DIR, "test_A1break.npz")
    np.savez(path, X_A1=X_A1, X_A2=X_A2, X_A3=X_A3,
             a1_breaks=np.ones(n, bool),
             a2_breaks=np.zeros(n, bool),
             a3_breaks=np.zeros(n, bool),
             Re=Re)

    print("=== test_A1break.npz ===")
    print(f"  n={n}, X_A1={X_A1.shape}, X_A2={X_A2.shape}, X_A3={X_A3.shape}")
    print(f"  Re range: [{Re.min():.0f}, {Re.max():.0f}]")
    print(f"  a1_breaks=True, a2_breaks=False, a3_breaks=False")
    print()


def generate_test_A2break_low(n=500, seed=51):
    """
    A2 breaks (low Pr): Pr in [0.01, 0.5] (liquid-metal regime).
    Vary mu to achieve target Pr; adjust v to keep Re=50000.
    A1 valid (Re=50000), A3 valid (L/D=50).
    """
    rng = np.random.default_rng(seed)
    Re_fixed = 50000.0

    Pr  = log_uniform(rng, 0.01, 0.5, n)
    mu  = Pr * K / CP
    v   = Re_fixed * mu / (RHO * D)   # maintain Re=50000

    X_A1 = np.column_stack([v, np.full(n, D), np.full(n, RHO), mu])
    X_A2 = np.column_stack([mu, np.full(n, CP), np.full(n, K)])
    X_A3 = _baseline_A3_features(n)

    path = os.path.join(SAVE_DIR, "test_A2break_low.npz")
    np.savez(path, X_A1=X_A1, X_A2=X_A2, X_A3=X_A3,
             a1_breaks=np.zeros(n, bool),
             a2_breaks=np.ones(n, bool),
             a3_breaks=np.zeros(n, bool),
             Re=np.full(n, Re_fixed), Pr=Pr)

    frac_low = (Pr < 0.6).mean()
    frac_high = (Pr > 160).mean()
    print("=== test_A2break_low.npz ===")
    print(f"  n={n}, X_A1={X_A1.shape}, X_A2={X_A2.shape}, X_A3={X_A3.shape}")
    print(f"  Pr range: [{Pr.min():.4f}, {Pr.max():.4f}]")
    print(f"  Fraction Pr<0.6: {frac_low*100:.1f}%   Fraction Pr>160: {frac_high*100:.1f}%")
    print(f"  a1_breaks=False, a2_breaks=True, a3_breaks=False")
    print()


def generate_test_A2break_high(n=500, seed=52):
    """
    A2 breaks (high Pr): Pr in [200, 500] (viscous-oil regime).
    Vary mu to achieve target Pr; adjust v to keep Re=50000.
    Note: v becomes very large (~70-180 m/s) due to high viscosity.
    A1 valid (Re=50000), A3 valid (L/D=50).
    """
    rng = np.random.default_rng(seed)
    Re_fixed = 50000.0

    Pr  = log_uniform(rng, 200, 500, n)
    mu  = Pr * K / CP
    v   = Re_fixed * mu / (RHO * D)   # maintain Re=50000; v can be 70-180 m/s

    X_A1 = np.column_stack([v, np.full(n, D), np.full(n, RHO), mu])
    X_A2 = np.column_stack([mu, np.full(n, CP), np.full(n, K)])
    X_A3 = _baseline_A3_features(n)

    path = os.path.join(SAVE_DIR, "test_A2break_high.npz")
    np.savez(path, X_A1=X_A1, X_A2=X_A2, X_A3=X_A3,
             a1_breaks=np.zeros(n, bool),
             a2_breaks=np.ones(n, bool),
             a3_breaks=np.zeros(n, bool),
             Re=np.full(n, Re_fixed), Pr=Pr)

    frac_low = (Pr < 0.6).mean()
    frac_high = (Pr > 160).mean()
    print("=== test_A2break_high.npz ===")
    print(f"  n={n}, X_A1={X_A1.shape}, X_A2={X_A2.shape}, X_A3={X_A3.shape}")
    print(f"  Pr range: [{Pr.min():.1f}, {Pr.max():.1f}]")
    print(f"  Fraction Pr<0.6: {frac_low*100:.1f}%   Fraction Pr>160: {frac_high*100:.1f}%")
    print(f"  v range: [{v.min():.1f}, {v.max():.1f}] m/s  (large due to high viscosity)")
    print(f"  a1_breaks=False, a2_breaks=True, a3_breaks=False")
    print()


def generate_test_A3break(n=1000, seed=53):
    """
    A3 breaks: L/D in [1, 8] (short pipe, entrance effects dominate).
    A1 valid (Re=50000, Pr=6.2), A2 valid (Pr=6.2).
    """
    rng = np.random.default_rng(seed)
    LD  = log_uniform(rng, 1, 8, n)
    L   = LD * D

    X_A1 = _baseline_A1_features(n, Re_fixed=50000.0)
    X_A2 = _baseline_A2_features(n)
    X_A3 = np.column_stack([L, np.full(n, D)])

    path = os.path.join(SAVE_DIR, "test_A3break.npz")
    np.savez(path, X_A1=X_A1, X_A2=X_A2, X_A3=X_A3,
             a1_breaks=np.zeros(n, bool),
             a2_breaks=np.zeros(n, bool),
             a3_breaks=np.ones(n, bool),
             LD=LD)

    print("=== test_A3break.npz ===")
    print(f"  n={n}, X_A1={X_A1.shape}, X_A2={X_A2.shape}, X_A3={X_A3.shape}")
    print(f"  L/D range: [{LD.min():.2f}, {LD.max():.2f}]")
    print(f"  a1_breaks=False, a2_breaks=False, a3_breaks=True")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Dittus-Boelter training and test data...\n")

    generate_train_A1()
    generate_train_A2()
    generate_train_A3()

    generate_test_A1break()
    generate_test_A2break_low()
    generate_test_A2break_high()
    generate_test_A3break()

    # Verify all 7 files exist
    files = [
        "train_A1.npz", "train_A2.npz", "train_A3.npz",
        "test_A1break.npz",
        "test_A2break_low.npz", "test_A2break_high.npz",
        "test_A3break.npz",
    ]
    print("File verification:")
    all_ok = True
    for f in files:
        p = os.path.join(SAVE_DIR, f)
        exists = os.path.exists(p)
        size   = os.path.getsize(p) // 1024 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f}  ({size} KB)")
        all_ok = all_ok and exists

    print()
    if all_ok:
        print("All 7 files generated successfully.")
    else:
        print("ERROR: some files are missing.")
        sys.exit(1)
