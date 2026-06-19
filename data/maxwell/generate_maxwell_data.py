"""Generate Maxwell's equations training and test data for validity predicate learning."""

import os
import numpy as np

# Physical constants (glass, lossy dielectric)
epsilon_0 = 8.854e-12          # F/m
epsilon_r = 2.25               # relative permittivity
epsilon   = epsilon_0 * epsilon_r   # = 1.992e-11 F/m
sigma_eff = 1e-3               # S/m, effective conductivity
E_sat     = 1e8                # V/m, nonlinear onset field

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN = 5000
N_TEST  = 1000

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Criterion functions
# ---------------------------------------------------------------------------

def log_criterion_a1(resid):
    """Positive when |D/(epsilonE)-1| < 0.05 (A1 linear_media holds)."""
    return np.clip(np.log(0.05 / (resid + 1e-10)), -20.0, 20.0)


def log_criterion_a2(ratio):
    """Positive when omega*epsilon/sigma_eff < 0.01 (A2 quasi_static holds)."""
    return np.clip(np.log(0.01 / (ratio + 1e-10)), -20.0, 20.0)


# ---------------------------------------------------------------------------
# Feature computation  (same formulas used in training and test generation)
# ---------------------------------------------------------------------------

def compute_a1_resid(E_field):
    """A1 feature (valid-regime / training): |D/(epsilon*E) - 1| with 0.1% Gaussian noise.

    Used for training data (E << E_sat) and for test scenarios where A1 holds
    (E still in [1e2, 1e6] V/m).  Gives resid = |noise| ~ 8e-4, independent of
    E_field.  Do NOT use this for Scenario A where E >> E_sat — use
    compute_a1_resid_physical() instead.
    """
    noise = rng.standard_normal(len(E_field))
    D = epsilon * E_field * (1.0 + 0.001 * noise)
    return np.abs(D / (epsilon * E_field + 1e-30) - 1.0)


def compute_a1_resid_physical(E_field):
    """A1 feature (breakdown regime): |D/(epsilon*E) - 1| from tanh saturation model.

    D = epsilon * E_sat * tanh(E / E_sat) captures polarisation saturation.
    At E >> E_sat: D ~ epsilon*E_sat, so D/(epsilon*E) ~ E_sat/E << 1,
    resid ~ 1 - E_sat/E >> 0.05.  Used only for test Scenario A.
    """
    D = epsilon * E_sat * np.tanh(E_field / E_sat)
    return np.abs(D / (epsilon * E_field + 1e-30) - 1.0)


def compute_a2_ratio(frequency):
    """A2 feature: displacement-to-conduction current ratio omega*epsilon/sigma_eff.

    Quasi-static limit holds when ratio << 1 (conduction dominates).
    Threshold: ratio = 0.01  =>  f_boundary ~ 79.9 kHz.
    """
    omega = 2.0 * np.pi * frequency
    return omega * epsilon / sigma_eff


# ---------------------------------------------------------------------------
# log_criterion functions for raw-feature training
# ---------------------------------------------------------------------------

def log_criterion_a1_from_field(E_field):
    """Positive when E_field < E_sat (linear regime holds)."""
    return np.clip(np.log(E_sat / (E_field + 1e-30)), -20.0, 20.0)


def log_criterion_a2_from_freq(frequency):
    """Positive when frequency < f_boundary ~ 79.9 kHz."""
    f_boundary = 0.01 * sigma_eff / (2.0 * np.pi * epsilon)
    return np.clip(np.log(f_boundary / (frequency + 1e-30)), -20.0, 20.0)


# ---------------------------------------------------------------------------
# Training data generators
# ---------------------------------------------------------------------------

def generate_train_a1():
    """5000 samples, valid A1 regime.

    Feature: E_field (V/m), log-uniform in [1e2, 1e7] — independent state variable.
    log_criterion: log(E_sat / E_field); positive when E_field < E_sat.
    log_transform_cols=(0,) applied internally by ValidityPredicate.
    """
    E_field  = np.exp(rng.uniform(np.log(1e2), np.log(1e7), N_TRAIN))
    features = E_field[:, np.newaxis]         # (N, 1) — raw E_field
    lc       = log_criterion_a1_from_field(E_field)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


def generate_train_a2():
    """5000 samples, valid A2 regime (low frequency).

    Feature: frequency (Hz), log-uniform in [1, 1e4] — independent state variable.
    log_criterion: log(f_boundary / frequency); positive when below ~79.9 kHz.
    log_transform_cols=(0,) applied internally by ValidityPredicate.
    """
    f        = np.exp(rng.uniform(np.log(1.0), np.log(1e4), N_TRAIN))
    features = f[:, np.newaxis]               # (N, 1) — raw frequency
    lc       = log_criterion_a2_from_freq(f)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


# ---------------------------------------------------------------------------
# Test scenario generators
# ---------------------------------------------------------------------------

def generate_test_a():
    """Scenario A: A1 fires (high E-field), A2 silent (low frequency).

    A1_features = E_field (V/m); predicate trained on E_field with log_transform_cols=(0,).
    A2_features = frequency (Hz); predicate trained on frequency with log_transform_cols=(0,).
    """
    E_field = np.exp(rng.uniform(np.log(5e8), np.log(1e10), N_TEST))
    f       = np.exp(rng.uniform(np.log(1.0),  np.log(100.0), N_TEST))
    A1_feat = E_field[:, np.newaxis]
    A2_feat = f[:, np.newaxis]
    lc = log_criterion_a1_from_field(E_field)
    return {
        "features":      A1_feat,
        "log_criterion": lc,
        "is_valid":      np.zeros(N_TEST, dtype=bool),
        "A1_features":   A1_feat,
        "A2_features":   A2_feat,
        "a1_breaks":     np.bool_(True),
        "a2_breaks":     np.bool_(False),
    }


def generate_test_b():
    """Scenario B: A2 fires (high frequency), A1 silent (low E-field).

    KEY RESULT: A2 in D1 footprint only => D1 SUSPECT, D2/D3/D4 TRUSTED.
    Mirrors LE Scenario C cross-domain.
    """
    f       = np.exp(rng.uniform(np.log(1e8), np.log(1e10), N_TEST))
    E_field = np.exp(rng.uniform(np.log(1e2), np.log(1e6),  N_TEST))
    A1_feat = E_field[:, np.newaxis]
    A2_feat = f[:, np.newaxis]
    return {
        "features":      A2_feat,
        "log_criterion": log_criterion_a2_from_freq(f),
        "is_valid":      np.zeros(N_TEST, dtype=bool),
        "A1_features":   A1_feat,
        "A2_features":   A2_feat,
        "a1_breaks":     np.bool_(False),
        "a2_breaks":     np.bool_(True),
    }


def generate_test_c():
    """Scenario C: valid holdout (neither fires) — FPR check.

    E_field in [1e2, 1e6] V/m (below E_sat=1e8) and frequency in [1, 1e4] Hz
    (below f_boundary~79.9 kHz) so all samples are in the valid regime.
    """
    E_field = np.exp(rng.uniform(np.log(1e2), np.log(1e6), N_TEST))
    f       = np.exp(rng.uniform(np.log(1.0),  np.log(1e4), N_TEST))
    A1_feat = E_field[:, np.newaxis]
    A2_feat = f[:, np.newaxis]
    return {
        "features":      A1_feat,
        "log_criterion": log_criterion_a1_from_field(E_field),
        "is_valid":      np.ones(N_TEST, dtype=bool),
        "A1_features":   A1_feat,
        "A2_features":   A2_feat,
        "a1_breaks":     np.bool_(False),
        "a2_breaks":     np.bool_(False),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Training files ===")
    train_specs = [
        ("train_A1", generate_train_a1),
        ("train_A2", generate_train_a2),
    ]

    for name, gen_fn in train_specs:
        features, lc, is_valid = gen_fn()
        mean_lc = float(np.mean(lc))
        std_lc  = float(np.std(lc))

        shift = 0.0
        if mean_lc > 10.0:
            shift = mean_lc
            lc = lc - shift
            print(f"  {name}: WARNING mean(log_criterion)={mean_lc:.3f} > 10 -- "
                  f"recentering applied (shift={shift:.3f})")

        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez(path, features=features, log_criterion=lc, is_valid=is_valid,
                 shift=np.float64(shift))

        print(f"  {name}: n={len(features)}, shape={features.shape}, "
              f"log_criterion mean={mean_lc:.3f}  std={std_lc:.3f}"
              + (f"  [shift={shift:.3f}]" if shift != 0.0 else ""))

        neg_frac = float(np.mean(lc < 0))
        if neg_frac > 0.0:
            print(f"    NOTE: {neg_frac*100:.1f}% of samples have log_criterion < 0 "
                  f"(broken-regime samples in training set)")

    print("\n=== Test files ===")
    test_specs = [
        ("test_scenario_A", generate_test_a),
        ("test_scenario_B", generate_test_b),
        ("test_scenario_C", generate_test_c),
    ]

    for name, gen_fn in test_specs:
        data = gen_fn()
        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez(path, **data)

        n       = len(data["A1_features"])
        lc_mean = float(np.mean(data["log_criterion"]))
        a1_feat_mean = float(np.mean(data["A1_features"]))
        a2_feat_mean = float(np.mean(data["A2_features"]))
        print(f"  {name}: n={n}, "
              f"A1_feat(E_field) mean={a1_feat_mean:.4e}, "
              f"A2_feat(freq) mean={a2_feat_mean:.4e}, "
              f"a1_breaks={data['a1_breaks']}, "
              f"a2_breaks={data['a2_breaks']}, "
              f"mean_lc={lc_mean:.3f}")

    print("\n=== File verification ===")
    expected = [
        "train_A1.npz", "train_A2.npz",
        "test_scenario_A.npz", "test_scenario_B.npz", "test_scenario_C.npz",
    ]
    all_ok = True
    for fname in expected:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"  {fname}  ({size_kb:.0f} KB)  OK")
        else:
            print(f"  {fname}  MISSING")
            all_ok = False

    if all_ok:
        print("\nAll 5 files created successfully.")
    else:
        print("\nERROR: some files are missing.")
