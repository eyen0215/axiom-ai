"""
Generate pipe flow training and test datasets for three axiom predicates.

A1 — laminar_flow:     features [v, D, rho, mu]       breaks when Re > 2300
A2 — fully_developed:  features [x, v, D, rho, mu]    breaks when x < L_entry
A3 — incompressible:   features [v]                    breaks when v > 102.9 m/s

log_criterion convention: positive => assumption healthy, negative => breakdown.
"""

import numpy as np
import os

RHO = 1.2
MU = 1.81e-5
C_SOUND = 343.0
RE_CRITICAL = 2300.0
MA_CRITICAL = 0.3
V_MA_LIMIT = MA_CRITICAL * C_SOUND   # 102.9 m/s
ENTRY_COEFF = 0.06


def _clip(x):
    return np.clip(x, -20.0, 20.0)


# ---------------------------------------------------------------------------
# Training sets (all in the valid regime for the respective predicate)
# ---------------------------------------------------------------------------

def generate_train_A1(n=5000, seed=42):
    """Laminar only: reject Re >= 2300 and resample. ~0.37% acceptance rate."""
    rng = np.random.default_rng(seed)
    features = np.empty((n, 4))
    log_crit = np.empty(n)
    collected = 0

    while collected < n:
        bsize = max(500_000, 4 * (n - collected))
        v = rng.uniform(0.1, 50.0, bsize)
        D = rng.uniform(0.01, 0.5, bsize)
        Re = RHO * v * D / MU
        ok = Re < RE_CRITICAL
        v_ok, D_ok, Re_ok = v[ok], D[ok], Re[ok]
        take = min(len(v_ok), n - collected)
        if take == 0:
            continue
        features[collected:collected + take, 0] = v_ok[:take]
        features[collected:collected + take, 1] = D_ok[:take]
        features[collected:collected + take, 2] = RHO
        features[collected:collected + take, 3] = MU
        log_crit[collected:collected + take] = _clip(
            np.log(RE_CRITICAL / (Re_ok[:take] + 1e-10))
        )
        collected += take

    return features, log_crit, np.ones(n, dtype=bool)


def generate_train_A2(n=5000, seed=43):
    """Fully developed: x sampled close above entrance length, Re < 2300.

    Re < 2300 constraint (same as A1) removes the x-scale mismatch that caused
    the trivially-easy 0.9995 AUROC in Attempt 1.  Tighter x margin
    [1.05*L_entry, 1.05*L_entry+10] instead of [1.5*L_entry, 1.5*L_entry+20]
    puts training mass near the true boundary so the skip path learns the
    correct extrapolation slope.
    """
    rng = np.random.default_rng(seed)
    features = np.empty((n, 5))
    log_crit = np.empty(n)
    collected = 0

    while collected < n:
        bsize = max(500_000, 4 * (n - collected))
        v = rng.uniform(0.1, 50.0, bsize)
        D = rng.uniform(0.01, 0.5, bsize)
        Re = RHO * v * D / MU
        ok = Re < RE_CRITICAL
        v_ok, D_ok, Re_ok = v[ok], D[ok], Re[ok]
        if len(v_ok) == 0:
            continue
        take = min(len(v_ok), n - collected)
        v_t = v_ok[:take]
        D_t = D_ok[:take]
        Re_t = Re_ok[:take]
        L_entry_t = ENTRY_COEFF * Re_t * D_t
        x_t = rng.uniform(0.0, 10.0, take) + L_entry_t * 1.05
        features[collected:collected + take, 0] = x_t
        features[collected:collected + take, 1] = v_t
        features[collected:collected + take, 2] = D_t
        features[collected:collected + take, 3] = RHO
        features[collected:collected + take, 4] = MU
        log_crit[collected:collected + take] = _clip(np.log(x_t / (L_entry_t + 1e-10)))
        collected += take

    return features, log_crit, np.ones(n, dtype=bool)


def generate_train_A3(n=5000, seed=44):
    """Incompressible: v in [0.1, 100.0], all below 102.9 m/s threshold."""
    rng = np.random.default_rng(seed)
    # Sampling to [100.0] means all pass v <= 102.9; rejection loop handles edge case
    collected_v = []
    while len(collected_v) < n:
        batch = rng.uniform(0.1, 100.0, n)
        collected_v.append(batch[batch <= V_MA_LIMIT])
    v = np.concatenate(collected_v)[:n]

    features = v.reshape(-1, 1)
    log_crit = _clip(np.log(V_MA_LIMIT / (v + 1e-10)))
    return features, log_crit, np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def generate_test_A1break(n=1000, seed=45):
    """Re > 2300 (A1 breaks). x past L_entry (A2 holds). v < 102.9 (A3 holds)."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(20.0, 50.0, n)     # v < 102.9 ensures A3 holds
    D = rng.uniform(0.1, 0.5, n)
    Re = RHO * v * D / MU
    L_entry = ENTRY_COEFF * Re * D
    x = rng.uniform(1.5, 2.5, n) * L_entry    # comfortably past entrance length

    print(f"  A1break achieved Re range: [{Re.min():.1f}, {Re.max():.1f}]")

    A1_features = np.column_stack([v, D, np.full(n, RHO), np.full(n, MU)])
    A2_features = np.column_stack([x, v, D, np.full(n, RHO), np.full(n, MU)])
    A3_features = v.reshape(-1, 1)
    return A1_features, A2_features, A3_features


def generate_test_A2break(n=1000, seed=46):
    """x < L_entry (A2 breaks). Re < 2300 (A1 holds). v < 102.9 (A3 holds)."""
    rng = np.random.default_rng(seed)
    A1_features = np.empty((n, 4))
    A2_features = np.empty((n, 5))
    A3_features = np.empty((n, 1))
    collected = 0

    while collected < n:
        bsize = max(500_000, 4 * (n - collected))
        v = rng.uniform(0.1, 50.0, bsize)
        D = rng.uniform(0.01, 0.5, bsize)
        Re = RHO * v * D / MU

        ok = Re < RE_CRITICAL
        v, D, Re = v[ok], D[ok], Re[ok]
        if len(v) == 0:
            continue

        L_entry = ENTRY_COEFF * Re * D
        # Need x in [0.1, L_entry*0.9]; only feasible when L_entry*0.9 > 0.1
        usable = L_entry * 0.9 > 0.1
        v, D, Re, L_entry = v[usable], D[usable], Re[usable], L_entry[usable]
        if len(v) == 0:
            continue

        x = rng.uniform(0.0, 1.0, len(v)) * (L_entry * 0.9 - 0.1) + 0.1
        take = min(len(v), n - collected)
        s = slice(collected, collected + take)
        A1_features[s] = np.column_stack([v[:take], D[:take], np.full(take, RHO), np.full(take, MU)])
        A2_features[s] = np.column_stack([x[:take], v[:take], D[:take], np.full(take, RHO), np.full(take, MU)])
        A3_features[collected:collected + take, 0] = v[:take]
        collected += take

    return A1_features, A2_features, A3_features


def generate_test_A3break(n=1000, seed=47):
    """v > 102.9 (A3 breaks). A1 may also break — reported, not forced independent."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(110.0, 200.0, n)
    D = rng.uniform(0.01, 0.1, n)
    Re = RHO * v * D / MU
    L_entry = ENTRY_COEFF * Re * D
    x = rng.uniform(1.5, 2.5, n) * L_entry

    a1_breaks = Re > RE_CRITICAL
    print(f"  A3break achieved Re range: [{Re.min():.1f}, {Re.max():.1f}]")
    print(f"  A1 also breaks for {a1_breaks.sum()}/{n} samples ({100.0 * a1_breaks.mean():.1f}%)")

    A1_features = np.column_stack([v, D, np.full(n, RHO), np.full(n, MU)])
    A2_features = np.column_stack([x, v, D, np.full(n, RHO), np.full(n, MU)])
    A3_features = v.reshape(-1, 1)
    return A1_features, A2_features, A3_features, a1_breaks


# ---------------------------------------------------------------------------
# Dense grid for A2 boundary figure
# ---------------------------------------------------------------------------

def generate_grid_A2_boundary():
    """50×50 (x, v) grid at fixed D=0.01 for decision boundary visualization.

    D=0.01 gives boundary slope x = 0.06*(rho*D/mu)*D * v = 0.3978*v.
    Grid ranges chosen so the boundary bisects the grid (~50/50 valid/invalid):
      v in [0.5, 3.4]  ->  Re in [331, 2253], all < 2300 (within training dist.)
      x in [0.05, 1.5] ->  boundary enters at v=0.5, x=0.199 (10% of x-range)
                            and exits at v=3.4, x=1.352 (90% of x-range)
    """
    D_fixed = 0.01
    x_grid = np.linspace(0.05, 1.5, 50)
    v_grid = np.linspace(0.5, 3.4, 50)

    XX, VV = np.meshgrid(x_grid, v_grid, indexing='ij')   # shape (50, 50)
    x_flat = XX.ravel()    # len 2500; reshape(50,50)[i,j] -> x_grid[i], v_grid[j]
    v_flat = VV.ravel()

    Re_flat = RHO * v_flat * D_fixed / MU
    L_entry_flat = ENTRY_COEFF * Re_flat * D_fixed
    true_label = x_flat > L_entry_flat   # True = fully developed = valid

    valid_frac   = true_label.mean()
    invalid_frac = 1.0 - valid_frac
    print(f"  grid_A2_boundary: valid_frac={valid_frac:.3f}  "
          f"invalid_frac={invalid_frac:.3f}  "
          f"(target: each side in [10%, 90%])")
    if valid_frac > 0.90 or valid_frac < 0.10:
        raise ValueError(
            f"Grid is miscalibrated: valid_frac={valid_frac:.3f}. "
            "Adjust x_grid/v_grid ranges before proceeding."
        )

    A2_features = np.column_stack([
        x_flat, v_flat,
        np.full(len(x_flat), D_fixed),
        np.full(len(x_flat), RHO),
        np.full(len(x_flat), MU),
    ])
    return x_grid, v_grid, A2_features, true_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating train_A1.npz ...")
    feats, lc, iv = generate_train_A1()
    np.savez(os.path.join(out_dir, "train_A1.npz"),
             features=feats, log_criterion=lc, is_valid=iv)
    print(f"  count={len(feats)}, feature shape={feats.shape}, mean log_criterion={lc.mean():.4f}")

    print("Generating train_A2.npz ...")
    feats, lc, iv = generate_train_A2()
    np.savez(os.path.join(out_dir, "train_A2.npz"),
             features=feats, log_criterion=lc, is_valid=iv)
    print(f"  count={len(feats)}, feature shape={feats.shape}, mean log_criterion={lc.mean():.4f}")

    print("Generating train_A3.npz ...")
    feats, lc, iv = generate_train_A3()
    np.savez(os.path.join(out_dir, "train_A3.npz"),
             features=feats, log_criterion=lc, is_valid=iv)
    print(f"  count={len(feats)}, feature shape={feats.shape}, mean log_criterion={lc.mean():.4f}")

    print("Generating test_scenario_A1break.npz ...")
    A1f, A2f, A3f = generate_test_A1break()
    np.savez(os.path.join(out_dir, "test_scenario_A1break.npz"),
             A1_features=A1f, A2_features=A2f, A3_features=A3f,
             a1_breaks=np.bool_(True), a2_breaks=np.bool_(False), a3_breaks=np.bool_(False))
    print(f"  saved {len(A1f)} samples")

    print("Generating test_scenario_A2break.npz ...")
    A1f, A2f, A3f = generate_test_A2break()
    np.savez(os.path.join(out_dir, "test_scenario_A2break.npz"),
             A1_features=A1f, A2_features=A2f, A3_features=A3f,
             a1_breaks=np.bool_(False), a2_breaks=np.bool_(True), a3_breaks=np.bool_(False))
    print(f"  saved {len(A1f)} samples")

    print("Generating test_scenario_A3break.npz ...")
    A1f, A2f, A3f, a1_brk = generate_test_A3break()
    np.savez(os.path.join(out_dir, "test_scenario_A3break.npz"),
             A1_features=A1f, A2_features=A2f, A3_features=A3f,
             a1_breaks=a1_brk, a2_breaks=np.bool_(False), a3_breaks=np.bool_(True))
    print(f"  saved {len(A1f)} samples")

    print("Generating grid_A2_boundary.npz ...")
    xg, vg, grid_feats, true_lbl = generate_grid_A2_boundary()
    np.savez(os.path.join(out_dir, "grid_A2_boundary.npz"),
             A2_features=grid_feats, x_grid=xg, v_grid=vg, true_label=true_lbl)
    print(f"  A2_features shape={grid_feats.shape}, valid fraction={true_lbl.mean():.3f}")

    print("\nAll 7 files generated successfully.")
