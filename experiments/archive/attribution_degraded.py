"""
Test per-assumption attribution accuracy with degraded (far-from-boundary) A2 predicates.

The key question: does attribution stay correct even when the A2 predicate was trained
far from the boundary (M=10, M=20)?

Three scenarios, 300 samples each:
  X: only A1 breaks  (Re > 2300, x > L_entry, v < 102.9)
  Y: only A2 breaks  (x < L_entry, Re < 2300, v < 102.9)
  Z: only A3 breaks  (v > 102.9, tiny D so Re < 2300, x > L_entry)

For M in [1.05, 10, 20]:
  A1 predicate: pipe_A1.pt  (original, not varied)
  A2 predicate: pipe_A2_M{M}.pt  (trained at M × L_entry from boundary)
  A3 predicate: pipe_A3.pt  (original, not varied)

A predicate is judged to have "fired" if more than 50% of the scenario's
300 samples score below the 0.5 threshold.

Attribution is correct for a scenario when:
  expected predicate fires (majority) AND both others are silent (majority).
Attribution accuracy = number of correctly attributed scenarios out of 3.

Honesty:
  Scenario X v/D ranges ([2,15] × [0.01,0.05]) match the A2 training distribution
  so OOD behavior of A2 doesn't confound the X/Z results.
  Scenario Z uses v > 102.9 and tiny D (sub-mm), which is outside the
  A2 training distribution for v — this tests genuine OOD extrapolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

SAVE_DIR = Path("validity_predicates/saved")

N    = 300
SEED = 42
RHO  = 1.2
MU   = 1.81e-5
ENTRY_COEFF = 0.06
RE_CRIT     = 2300.0
V_MA_LIM    = 0.3 * 343.0   # 102.9 m/s
FIRE_THR    = 0.5

Ms_TEST = [1.05, 10, 20]


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenario_X(n: int, seed: int) -> dict:
    """A1 breaks (Re > 2300), A2 holds (x > L_entry), A3 holds (v < 102.9).

    Uses v/D within A2 degradation-curve training distribution [2,15]×[0.01,0.05]
    so A2's OOD behaviour doesn't contaminate the result.
    """
    rng = np.random.default_rng(seed)
    v_out = np.empty(n)
    D_out = np.empty(n)
    done  = 0
    while done < n:
        bs   = max(100_000, 4 * (n - done))
        v    = np.exp(rng.uniform(np.log(2.0),  np.log(15.0),  bs))
        D    = np.exp(rng.uniform(np.log(0.01), np.log(0.05), bs))
        Re   = RHO * v * D / MU
        ok   = Re > RE_CRIT
        take = min(ok.sum(), n - done)
        v_out[done:done + take] = v[ok][:take]
        D_out[done:done + take] = D[ok][:take]
        done += take

    Re_arr = RHO * v_out * D_out / MU
    L_arr  = ENTRY_COEFF * Re_arr * D_out
    x_arr  = rng.uniform(1.5, 3.0, n) * L_arr   # past entrance → A2 holds
    rho_arr = np.full(n, RHO)
    mu_arr  = np.full(n, MU)

    assert np.all(Re_arr > RE_CRIT),  "X: A1 must break for all samples"
    assert np.all(x_arr > L_arr),     "X: A2 must hold for all samples"
    assert np.all(v_out < V_MA_LIM),  "X: A3 must hold for all samples"

    return {
        "A1_feats": np.column_stack([v_out, D_out, rho_arr, mu_arr]),
        "A2_feats": np.column_stack([x_arr, v_out, D_out, rho_arr, mu_arr]),
        "A3_feats": v_out.reshape(-1, 1),
        "Re": Re_arr, "L_entry": L_arr, "v": v_out,
    }


def generate_scenario_Y(n: int, seed: int) -> dict:
    """A2 breaks (x < L_entry), A1 holds (Re < 2300), A3 holds (v < 102.9)."""
    rng = np.random.default_rng(seed)
    v_out = np.empty(n)
    D_out = np.empty(n)
    done  = 0
    while done < n:
        bs   = max(100_000, 4 * (n - done))
        v    = np.exp(rng.uniform(np.log(0.5),   np.log(15.0),  bs))
        D    = np.exp(rng.uniform(np.log(0.005), np.log(0.05), bs))
        Re   = RHO * v * D / MU
        ok   = Re < RE_CRIT
        take = min(ok.sum(), n - done)
        v_out[done:done + take] = v[ok][:take]
        D_out[done:done + take] = D[ok][:take]
        done += take

    Re_arr  = RHO * v_out * D_out / MU
    L_arr   = ENTRY_COEFF * Re_arr * D_out
    x_arr   = rng.uniform(0.05, 0.9, n) * L_arr   # in entrance region → A2 breaks
    rho_arr = np.full(n, RHO)
    mu_arr  = np.full(n, MU)

    assert np.all(Re_arr < RE_CRIT), "Y: A1 must hold for all samples"
    assert np.all(x_arr  < L_arr),  "Y: A2 must break for all samples"
    assert np.all(v_out  < V_MA_LIM),"Y: A3 must hold for all samples"

    return {
        "A1_feats": np.column_stack([v_out, D_out, rho_arr, mu_arr]),
        "A2_feats": np.column_stack([x_arr, v_out, D_out, rho_arr, mu_arr]),
        "A3_feats": v_out.reshape(-1, 1),
        "Re": Re_arr, "L_entry": L_arr, "v": v_out,
    }


def generate_scenario_Z(n: int, seed: int) -> dict:
    """A3 breaks (v > 102.9), A1 holds (Re < 2300 via tiny D), A2 holds (x > L_entry).

    D < RE_CRIT * MU / (RHO * v) ensures Re < 2300 at the given v.
    This puts D in the sub-mm range (OOD for A2's v dimension as well).
    """
    rng  = np.random.default_rng(seed)
    v_out = np.empty(n)
    D_out = np.empty(n)
    done  = 0
    while done < n:
        bs   = max(100_000, 4 * (n - done))
        v    = rng.uniform(103.0, 150.0, bs)
        # D_max so that Re = RE_CRIT; sample below that to keep Re < 2300
        D_max = RE_CRIT * MU / (RHO * v)
        D     = np.exp(rng.uniform(np.log(D_max * 0.05), np.log(D_max * 0.95)))
        Re    = RHO * v * D / MU
        ok    = (Re < RE_CRIT) & (v > V_MA_LIM)
        take  = min(ok.sum(), n - done)
        v_out[done:done + take] = v[ok][:take]
        D_out[done:done + take] = D[ok][:take]
        done += take

    Re_arr  = RHO * v_out * D_out / MU
    L_arr   = ENTRY_COEFF * Re_arr * D_out
    x_arr   = rng.uniform(1.5, 3.0, n) * L_arr   # past entrance → A2 holds
    rho_arr = np.full(n, RHO)
    mu_arr  = np.full(n, MU)

    assert np.all(v_out  > V_MA_LIM),  "Z: A3 must break for all samples"
    assert np.all(Re_arr < RE_CRIT),   "Z: A1 must hold for all samples"
    assert np.all(x_arr  > L_arr),     "Z: A2 must hold for all samples"

    return {
        "A1_feats": np.column_stack([v_out, D_out, rho_arr, mu_arr]),
        "A2_feats": np.column_stack([x_arr, v_out, D_out, rho_arr, mu_arr]),
        "A3_feats": v_out.reshape(-1, 1),
        "Re": Re_arr, "L_entry": L_arr, "v": v_out,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load(path: Path, n_features: int, log_cols: tuple) -> ValidityPredicate:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    m    = ValidityPredicate(n_features=n_features, log_transform_cols=log_cols)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def load_A1() -> ValidityPredicate:
    return _load(SAVE_DIR / "pipe_A1.pt", n_features=4, log_cols=(0, 1, 2, 3))

def load_A2(M: float) -> ValidityPredicate:
    return _load(SAVE_DIR / f"pipe_A2_M{M:g}.pt", n_features=5, log_cols=(0, 1, 2, 3, 4))

def load_A3() -> ValidityPredicate:
    return _load(SAVE_DIR / "pipe_A3.pt", n_features=1, log_cols=(0,))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def fire_rate(model: ValidityPredicate, feats: np.ndarray) -> float:
    scores = model.predict(feats.astype(np.float32))
    return float((scores < FIRE_THR).mean())


def yn(rate: float) -> str:
    return "YES" if rate > 0.5 else "NO "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPECTED = {
    "X": (True,  False, False),   # A1 breaks, A2 and A3 hold
    "Y": (False, True,  False),   # A2 breaks, A1 and A3 hold
    "Z": (False, False, True),    # A3 breaks, A1 and A2 hold
}
SCENARIO_LABELS = {
    "X": "only A1 breaks (turbulent)",
    "Y": "only A2 breaks (entrance region)",
    "Z": "only A3 breaks (compressible, tiny D)",
}


def main() -> None:
    print("=" * 66)
    print("Attribution accuracy under degraded A2 detection")
    print("=" * 66)
    print()
    print("Scenarios (300 samples each):")
    for sc, lbl in SCENARIO_LABELS.items():
        exp = EXPECTED[sc]
        print(f"  {sc}: {lbl}")
        print(f"     expected: A1={'YES' if exp[0] else 'NO '}, "
              f"A2={'YES' if exp[1] else 'NO '}, A3={'YES' if exp[2] else 'NO '}")
    print()
    print("A1 = pipe_A1.pt (original)  |  A3 = pipe_A3.pt (original)")
    print("A2 = pipe_A2_M{M}.pt — varied by distance from boundary")
    print()

    # ------------------------------------------------------------------
    # Generate scenarios once (same test set for all M)
    # ------------------------------------------------------------------
    scenarios = {
        "X": generate_scenario_X(N, SEED),
        "Y": generate_scenario_Y(N, SEED + 1),
        "Z": generate_scenario_Z(N, SEED + 2),
    }

    print("Scenario statistics:")
    for sc_name, sc in scenarios.items():
        x_over_L = sc["A2_feats"][:, 0] / sc["L_entry"]
        print(f"  {sc_name}: Re=[{sc['Re'].min():.0f},{sc['Re'].max():.0f}]  "
              f"x/L_entry=[{x_over_L.min():.2f},{x_over_L.max():.2f}]  "
              f"v=[{sc['v'].min():.1f},{sc['v'].max():.1f}] m/s")
    print()

    # ------------------------------------------------------------------
    # Evaluate each M
    # ------------------------------------------------------------------
    pred_A1 = load_A1()
    pred_A3 = load_A3()

    all_results: dict[float, tuple[dict, int]] = {}

    for M in Ms_TEST:
        pred_A2 = load_A2(M)

        # Compute fire rates for every scenario × every predicate
        rates: dict[str, tuple[float, float, float]] = {}
        for sc_name, sc in scenarios.items():
            rates[sc_name] = (
                fire_rate(pred_A1, sc["A1_feats"]),
                fire_rate(pred_A2, sc["A2_feats"]),
                fire_rate(pred_A3, sc["A3_feats"]),
            )

        # Attribution accuracy
        n_correct = 0
        for sc_name, (r1, r2, r3) in rates.items():
            exp = EXPECTED[sc_name]
            if ((r1 > 0.5) == exp[0]) and ((r2 > 0.5) == exp[1]) and ((r3 > 0.5) == exp[2]):
                n_correct += 1

        all_results[M] = (rates, n_correct)

        # Per-M block
        print(f"{'='*60}")
        print(f"M = {M:g}  (A2 trained at {M:g}x L_entry from boundary)")
        print(f"{'='*60}")
        for sc_name, (r1, r2, r3) in rates.items():
            exp = EXPECTED[sc_name]
            ok  = ((r1>0.5)==exp[0]) and ((r2>0.5)==exp[1]) and ((r3>0.5)==exp[2])
            tag = "[OK]" if ok else "[XX]"
            print(
                f"  Scenario {sc_name}: "
                f"A1 fired? {yn(r1)} ({r1:.2f}), "
                f"A2 fired? {yn(r2)} ({r2:.2f}), "
                f"A3 fired? {yn(r3)} ({r3:.2f})  {tag}"
            )
        print(f"  Attribution accuracy: {n_correct}/3")
        print()

    # ------------------------------------------------------------------
    # Compact summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY — Attribution by M")
    print("=" * 70)
    print(f"{'M':<6} | {'Sc.X (A1 breaks)':<22} | {'Sc.Y (A2 breaks)':<22} | {'Sc.Z (A3 breaks)':<22} | Acc")
    print("-" * 80)
    for M in Ms_TEST:
        rates, n_correct = all_results[M]
        cols = []
        for sc_name in ("X", "Y", "Z"):
            r1, r2, r3 = rates[sc_name]
            exp = EXPECTED[sc_name]
            ok  = ((r1>0.5)==exp[0]) and ((r2>0.5)==exp[1]) and ((r3>0.5)==exp[2])
            tag = "[OK]" if ok else "[XX]"
            cols.append(f"A1={yn(r1)},A2={yn(r2)},A3={yn(r3)}{tag}")
        print(f"{M:<6g} | {cols[0]:<22} | {cols[1]:<22} | {cols[2]:<22} | {n_correct}/3")
    print("=" * 80)
    print()
    print("Expected fire pattern:")
    print("  Sc.X → A1=YES, A2=NO,  A3=NO  (turbulent; fully-developed; subsonic)")
    print("  Sc.Y → A1=NO,  A2=YES, A3=NO  (laminar; entrance; subsonic)")
    print("  Sc.Z → A1=NO,  A2=NO,  A3=YES (laminar; fully-developed; compressible)")
    print()
    print("Key: [OK] = correct attribution, [XX] = wrong attribution")
    print()

    # Concise verdict
    all_acc = [all_results[M][1] for M in Ms_TEST]
    print("Verdict:")
    for M, acc in zip(Ms_TEST, all_acc):
        q = "maintained" if acc == 3 else ("degraded" if acc >= 2 else "lost")
        print(f"  M={M:g}: {acc}/3 correct — attribution {q}")
    if min(all_acc) == 3:
        print()
        print("  Attribution is PERFECT at all tested M values.")
        print("  The predictor correctly identifies which assumption breaks")
        print("  even when A2 is trained 20x away from the boundary.")
    elif all_results[Ms_TEST[-1]][1] >= 2:
        print()
        print("  Attribution degrades partially at far distances but remains useful.")


if __name__ == "__main__":
    main()
