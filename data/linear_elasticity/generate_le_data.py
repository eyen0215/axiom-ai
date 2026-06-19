"""Generate linear elasticity training and test data for validity predicate learning."""

import os
import numpy as np

# Physical constants (steel)
E = 200e9           # Pa, Young's modulus
nu = 0.3            # Poisson's ratio (unused in generation, kept for reference)
rho = 7800          # kg/m^3, density
sigma_yield = 250e6 # Pa, yield stress
H = 1e9             # Pa, hardening modulus (elasto-plastic)
L = 0.01            # m, specimen length scale

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN = 5000
N_TEST = 1000

rng = np.random.default_rng(42)


# --- criterion functions ---

def log_criterion_a1(epsilon_eq):
    """Positive in valid regime (ε_eq < 0.01), zero at boundary."""
    return np.log(0.01 / epsilon_eq)


def log_criterion_a2(epsilon_eq, sigma_vm):
    """Positive when σ_vm tracks E·ε within 10%."""
    residual = np.abs(sigma_vm - E * epsilon_eq) / (E * epsilon_eq + 1e-10)
    return np.clip(np.log(0.10 / (residual + 1e-10)), -20.0, 20.0)


def log_criterion_a5(f):
    """Positive when inertia-to-stress ratio < 0.01."""
    ir = rho * (2 * np.pi * f) ** 2 * L**2 / E
    return np.clip(np.log(0.01 / (ir + 1e-10)), -20.0, 20.0)


# --- training data generators ---

def generate_train_a1():
    epsilon_eq = rng.uniform(0.0001, 0.008, N_TRAIN)
    features = epsilon_eq[:, np.newaxis]          # (N, 1)
    lc = log_criterion_a1(epsilon_eq)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


def generate_train_a2():
    """Train on eps_eq only (independent state variable).

    Boundary: eps_yield = sigma_yield / E = 0.00125.
    Training range [0.0001, 0.001] is strictly sub-yield.
    log_criterion = clip(log(eps_yield / eps_eq), -20, 20): positive below yield.
    """
    epsilon_yield = sigma_yield / E   # 0.00125
    epsilon_eq = rng.uniform(0.0001, 0.001, N_TRAIN)
    features = epsilon_eq[:, np.newaxis]   # (N, 1) — single independent feature
    lc = np.clip(np.log(epsilon_yield / epsilon_eq), -20.0, 20.0)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


def generate_train_a5():
    f = np.exp(rng.uniform(np.log(0.1), np.log(100.0), N_TRAIN))
    features = f[:, np.newaxis]                   # (N, 1)
    lc = log_criterion_a5(f)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


# --- test scenario generators ---

def generate_test_scenario_a():
    """A1 breaks (large strain); A2 and A5 hold (still linear, low freq)."""
    epsilon_eq = rng.uniform(0.05, 0.20, N_TEST)
    f = np.exp(rng.uniform(np.log(0.1), np.log(100.0), N_TEST))
    sigma_vm = E * epsilon_eq   # linear response — A2 holds

    return {
        "A1_features": epsilon_eq[:, np.newaxis],
        "A2_features": np.column_stack([epsilon_eq, sigma_vm]),
        "A5_features": f[:, np.newaxis],
        "a1_breaks": np.bool_(True),
        "a2_breaks": np.bool_(False),
        "a5_breaks": np.bool_(False),
    }


def generate_test_scenario_b():
    """A2 breaks (elasto-plastic); A1 and A5 hold (small strain, low freq)."""
    epsilon_eq = rng.uniform(0.001, 0.008, N_TEST)
    f = np.exp(rng.uniform(np.log(0.1), np.log(100.0), N_TEST))

    epsilon_yield = sigma_yield / E   # ≈ 0.00125
    # Piecewise: elastic below yield, linear-hardening above
    sigma_vm = np.where(
        epsilon_eq <= epsilon_yield,
        E * epsilon_eq,
        sigma_yield + H * (epsilon_eq - epsilon_yield),
    )

    return {
        "A1_features": epsilon_eq[:, np.newaxis],
        "A2_features": np.column_stack([epsilon_eq, sigma_vm]),
        "A5_features": f[:, np.newaxis],
        "a1_breaks": np.bool_(False),
        "a2_breaks": np.bool_(True),
        "a5_breaks": np.bool_(False),
    }


def generate_test_scenario_c():
    """A5 breaks (high frequency); A1 and A2 hold (small strain, sub-yield, linear).

    eps_eq capped at 0.001 (below eps_yield=0.00125) so the rebuilt le_A2
    predicate [eps_eq only] does not falsely fire on this valid-A2 scenario.
    """
    epsilon_eq = rng.uniform(0.0001, 0.001, N_TEST)
    f = np.exp(rng.uniform(np.log(1e4), np.log(1e6), N_TEST))
    sigma_vm = E * epsilon_eq   # linear — A2 holds

    return {
        "A1_features": epsilon_eq[:, np.newaxis],
        "A2_features": np.column_stack([epsilon_eq, sigma_vm]),
        "A5_features": f[:, np.newaxis],
        "a1_breaks": np.bool_(False),
        "a2_breaks": np.bool_(False),
        "a5_breaks": np.bool_(True),
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Training files
    train_specs = [
        ("train_A1", generate_train_a1),
        ("train_A2", generate_train_a2),
        ("train_A5", generate_train_a5),
    ]

    print("=== Training files ===")
    for name, gen_fn in train_specs:
        features, lc, is_valid = gen_fn()
        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez(path, features=features, log_criterion=lc, is_valid=is_valid)
        mean_lc = float(np.mean(lc))
        std_lc = float(np.std(lc))
        print(f"  {name}: n={len(features)}, shape={features.shape}, "
              f"log_criterion mean={mean_lc:.3f}  std={std_lc:.3f}")
        if mean_lc > 10.0:
            print(f"    WARNING: mean(log_criterion) > 10 — calibration bias likely")

    # Test files
    test_specs = [
        ("test_scenario_A", generate_test_scenario_a),
        ("test_scenario_B", generate_test_scenario_b),
        ("test_scenario_C", generate_test_scenario_c),
    ]

    print("\n=== Test files ===")
    for name, gen_fn in test_specs:
        data = gen_fn()
        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez(path, **data)
        n = len(data["A1_features"])
        print(f"  {name}: n={n}, shape={data['A1_features'].shape[1:]}  "
              f"a1_breaks={data['a1_breaks']}  "
              f"a2_breaks={data['a2_breaks']}  "
              f"a5_breaks={data['a5_breaks']}")
