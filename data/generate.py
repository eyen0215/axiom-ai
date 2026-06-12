"""
Synthetic ideal gas data generator.

Generates (P, V, T, n) state tuples via PV = nRT with per-assumption validity
labels and a hard regime split:
    Training:  P ∈ [P_TRAIN_LOW,  P_TRAIN_HIGH]  — ideal gas assumptions safe
    Held-out:  P ∈ [P_TEST_LOW,   P_TEST_HIGH]   — van der Waals regime

Validity labels are derived analytically from two operationalizable
ideal-gas assumptions:

    Point-particle assumption  —  free volume per mole >> excluded volume b
        valid when  (V/n) / VDW_B  >  FREE_VOL_THRESHOLD

    No-intermolecular-forces   —  thermal energy >> interaction energy
        valid when  (VDW_A * n / V) / (R * T)  <  FORCE_THRESHOLD

Van der Waals parameters are those of N₂ as a representative real gas.

The labels are intentionally constructed so that nearly all training-regime
states are valid and nearly all held-out states are invalid — this is the
signal the validity predicates must learn to extrapolate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R = 0.08206       # L·atm / (mol·K)   universal gas constant

# Van der Waals parameters for N₂
VDW_A = 1.39      # L²·atm / mol²    intermolecular attraction coefficient
VDW_B = 0.0391    # L / mol          excluded volume per mole

# ---------------------------------------------------------------------------
# Regime boundaries
# ---------------------------------------------------------------------------

P_TRAIN_LOW  =   1.0   # atm — lower edge of training band
P_TRAIN_HIGH =  10.0   # atm — upper edge of training band / regime boundary
P_TEST_LOW   =  50.0   # atm — lower edge of held-out band
P_TEST_HIGH  = 200.0   # atm — upper edge of held-out band

T_LOW  = 300.0   # K — temperature sampling range (both regimes)
T_HIGH = 500.0   # K

# ---------------------------------------------------------------------------
# Assumption validity thresholds
# ---------------------------------------------------------------------------

# (V/n) / VDW_B must exceed this for the point-particle assumption to hold.
# At 10x the excluded volume, the finite-size correction is ~10% — marginal.
# At >10x the free volume, the correction is negligible.
FREE_VOL_THRESHOLD = 10.0

# (VDW_A * n / V) / (R * T) must stay below this fraction for the no-forces
# assumption to hold. 0.10 means interaction PE < 10% of thermal KE.
FORCE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_validity_labels(
    P: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytically compute per-assumption validity labels for a batch of states.

    Returns
    -------
    valid_point_particle : bool array
        True where the point-particle (no-volume) assumption holds.
    valid_no_forces : bool array
        True where the no-intermolecular-forces assumption holds.
    """
    free_vol_ratio = (V / n) / VDW_B              # dimensionless
    force_ratio    = (VDW_A * n / V) / (R * T)    # dimensionless

    valid_point_particle = free_vol_ratio > FREE_VOL_THRESHOLD
    valid_no_forces      = force_ratio    < FORCE_THRESHOLD

    return valid_point_particle, valid_no_forces


def generate_regime(
    n_samples: int,
    P_low: float,
    P_high: float,
    T_low: float = T_LOW,
    T_high: float = T_HIGH,
    n_moles: float = 1.0,
    regime_label: str = "train",
    rng: np.random.Generator = None,
    noise_frac: float = 0.005,
) -> pd.DataFrame:
    """
    Sample `n_samples` physical states uniformly in [P_low, P_high] × [T_low, T_high].

    Volume is derived from the ideal gas law (V = nRT/P), making PV=nRT the
    ground truth for the training regime. The same formula is applied in the
    held-out regime — the discrepancy between this V and the true van der Waals
    volume is the signal the predicates must detect, not something pre-encoded
    in the data.

    A small Gaussian noise (noise_frac ≈ 0.5%) is added to every quantity to
    simulate realistic measurement uncertainty.

    Parameters
    ----------
    n_samples    : number of states to generate
    P_low, P_high: pressure range in atm
    T_low, T_high: temperature range in K
    n_moles      : moles of gas (fixed per sample)
    regime_label : 'train' or 'held_out'
    rng          : seeded Generator for reproducibility
    noise_frac   : fractional std of Gaussian noise on each variable

    Returns
    -------
    DataFrame with columns: P, V, T, n, regime, valid_point_particle, valid_no_forces
    """
    if rng is None:
        rng = np.random.default_rng(42)

    P = rng.uniform(P_low, P_high, n_samples)
    T = rng.uniform(T_low, T_high, n_samples)
    n = np.full(n_samples, n_moles, dtype=float)

    V_ideal = (n * R * T) / P  # ideal gas law

    # Add small independent measurement noise to each observable
    P = P * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    V = V_ideal * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    T = T * (1.0 + rng.normal(0.0, noise_frac, n_samples))

    valid_pp, valid_nf = compute_validity_labels(P, V, T, n)

    return pd.DataFrame(
        {
            "P": P,
            "V": V,
            "T": T,
            "n": n,
            "regime": regime_label,
            "valid_point_particle": valid_pp,
            "valid_no_forces": valid_nf,
        }
    )


def generate_dataset(
    n_train: int = 5000,
    n_held_out: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full training / held-out dataset pair.

    The two RNG states are drawn from a single seeded Generator so that the
    split is fully reproducible without correlating the two regimes.

    Returns
    -------
    train_df    : low-pressure states (P_TRAIN_LOW – P_TRAIN_HIGH atm)
    held_out_df : high-pressure states (P_TEST_LOW – P_TEST_HIGH atm)
    """
    rng = np.random.default_rng(seed)
    train_df = generate_regime(
        n_train, P_TRAIN_LOW, P_TRAIN_HIGH, regime_label="train", rng=rng
    )
    held_out_df = generate_regime(
        n_held_out, P_TEST_LOW, P_TEST_HIGH, regime_label="held_out", rng=rng
    )
    return train_df, held_out_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_regime_boundary(
    train_df: pd.DataFrame,
    held_out_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Produce a two-panel figure showing the P–V regime split and assumption validity.

    Left panel  — P–V scatter, training vs held-out, with the regime boundary
                  drawn as a dashed horizontal line at P = P_TRAIN_HIGH.
    Right panel — same scatter colored by whether ALL ideal-gas assumptions are
                  valid (blue) or at least one is violated (red), plus the regime
                  boundary for reference.

    Both axes use log–log scale to show the full dynamic range of the dataset.

    Parameters
    ----------
    train_df    : DataFrame returned by generate_regime(..., regime_label='train')
    held_out_df : DataFrame returned by generate_regime(..., regime_label='held_out')
    save_path   : if given, save the figure to this path (PNG/PDF/SVG)
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Ideal Gas Regime Split and Assumption Validity  |  PV = nRT synthetic data",
        fontsize=13,
    )

    # ---- left panel: regime split ----------------------------------------
    ax_left.scatter(
        train_df["V"], train_df["P"],
        s=5, alpha=0.35, color="steelblue", label=f"Training  P = {P_TRAIN_LOW}–{P_TRAIN_HIGH} atm",
    )
    ax_left.scatter(
        held_out_df["V"], held_out_df["P"],
        s=5, alpha=0.35, color="tomato", label=f"Held-out  P = {P_TEST_LOW}–{P_TEST_HIGH} atm",
    )
    ax_left.axhline(
        P_TRAIN_HIGH, color="black", linewidth=1.8, linestyle="--",
        label=f"Regime boundary  P = {P_TRAIN_HIGH} atm",
    )
    ax_left.set_xlabel("Volume  V  (L)")
    ax_left.set_ylabel("Pressure  P  (atm)")
    ax_left.set_title("Training vs Held-out Regime")
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.legend(fontsize=8, markerscale=3)

    # Annotate regime labels directly on the plot
    ax_left.text(
        0.97, 0.20, "Training\n(ideal gas valid)",
        transform=ax_left.transAxes, ha="right", va="bottom",
        fontsize=8, color="steelblue",
    )
    ax_left.text(
        0.97, 0.80, "Held-out\n(van der Waals regime)",
        transform=ax_left.transAxes, ha="right", va="top",
        fontsize=8, color="tomato",
    )

    # ---- right panel: assumption validity --------------------------------
    all_df = pd.concat([train_df, held_out_df], ignore_index=True)
    both_valid = all_df["valid_point_particle"] & all_df["valid_no_forces"]

    ax_right.scatter(
        all_df.loc[both_valid, "V"],
        all_df.loc[both_valid, "P"],
        s=5, alpha=0.35, color="steelblue",
    )
    ax_right.scatter(
        all_df.loc[~both_valid, "V"],
        all_df.loc[~both_valid, "P"],
        s=5, alpha=0.35, color="tomato",
    )
    ax_right.axhline(
        P_TRAIN_HIGH, color="black", linewidth=1.8, linestyle="--",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=7, label="All assumptions valid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=7, label="≥1 assumption violated"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--",
               label=f"Regime boundary  P = {P_TRAIN_HIGH} atm"),
    ]
    ax_right.legend(handles=legend_handles, fontsize=8)
    ax_right.set_xlabel("Volume  V  (L)")
    ax_right.set_ylabel("Pressure  P  (atm)")
    ax_right.set_title("Assumption Validity in P–V Space")
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ===========================================================================
# HOOKE'S LAW DOMAIN
# ===========================================================================
#
# Synthetic data for F = kx (Hooke's Law / linear elasticity).
# Material: steel rod under uniaxial tension.
#
# The four assumptions and their analytical validity criteria:
#
#   A1 (Linearity)   : stress ratio r = sigma/sigma_y  < STRESS_RATIO_THRESHOLD
#                      Violated past yield when F-x response becomes nonlinear.
#
#   A2 (Elasticity)  : strain energy ratio U/U_yield   < STRAIN_ENERGY_THRESHOLD
#                      Violated when stored energy exceeds the elastic limit.
#                      U = sigma^2 / (2*E), so U/U_yield = r^2.
#
#   A3 (Small strain): strain epsilon = x/L0            < EPSILON_THRESHOLD
#                      Violated when geometric nonlinearity (large displacement)
#                      changes the effective stiffness k = EA/L0.
#
#   A4 (Homogeneity) : no operationalizable criterion from macroscopic
#                      (F, x, A, L0) alone -- skipped (like A3/A4 in ideal gas).
#
# Features stored in the DataFrame: F (N), x (m), A (m^2), L0 (m),
#   sigma (Pa), epsilon (dimensionless), stress_ratio (dimensionless),
#   strain_energy_ratio (dimensionless).
#
# The predicate uses ['stress_ratio', 'strain_energy_ratio', 'epsilon'] directly
# WITHOUT log-transform, because the validity boundaries are LINEAR in these
# normalised features (not log-linear as in the ideal gas case).
# ---------------------------------------------------------------------------

# Material constants -- steel
E_STEEL  = 200e9    # Young's modulus (Pa)
SIGMA_Y  = 250e6    # Yield strength (Pa)
EPSILON_Y = SIGMA_Y / E_STEEL          # Yield strain = 0.00125
U_YIELD   = SIGMA_Y**2 / (2 * E_STEEL) # Yield strain energy density (J/m^3) = 156250

# Regime boundaries -- split cleanly around the yield strain
EPSILON_TRAIN_LOW  = 0.00005            # 0.005% strain (well within elastic)
EPSILON_TRAIN_HIGH = EPSILON_Y * 0.50   # 0.0625%  (50% of yield)
EPSILON_TEST_LOW   = EPSILON_Y * 1.50   # 0.1875%  (past yield)
EPSILON_TEST_HIGH  = EPSILON_Y * 10.0   # 1.25%    (strongly post-yield)

# Geometry sampling ranges
A_LOW  = 1e-4   # 1 cm^2 cross-sectional area
A_HIGH = 1e-2   # 100 cm^2
L0_LOW  = 0.10  # 10 cm rod length
L0_HIGH = 2.00  # 2 m

# Validity thresholds (slightly below the physical yield to give margin)
STRESS_RATIO_THRESHOLD    = 0.90   # A1: sigma/sigma_y must be below this
STRAIN_ENERGY_THRESHOLD   = 0.80   # A2: U/U_yield must be below this
EPSILON_THRESHOLD         = EPSILON_Y * 0.85  # A3: epsilon threshold


def compute_hooke_validity_labels(
    stress_ratio: np.ndarray,
    strain_energy_ratio: np.ndarray,
    epsilon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytically compute per-assumption validity labels for Hooke's Law.

    Returns
    -------
    valid_linearity      : bool array -- A1 satisfied (stress ratio below threshold)
    valid_elasticity     : bool array -- A2 satisfied (strain energy ratio below threshold)
    valid_small_strain   : bool array -- A3 satisfied (strain below threshold)
    """
    return (
        stress_ratio      < STRESS_RATIO_THRESHOLD,
        strain_energy_ratio < STRAIN_ENERGY_THRESHOLD,
        epsilon           < EPSILON_THRESHOLD,
    )


def generate_hooke_regime(
    n_samples: int,
    epsilon_low: float,
    epsilon_high: float,
    regime_label: str = "train",
    rng: np.random.Generator = None,
    noise_frac: float = 0.002,
) -> pd.DataFrame:
    """Sample n_samples Hooke's Law states uniformly in [epsilon_low, epsilon_high].

    Geometry (A, L0) is sampled independently.  Force F and displacement x are
    derived from the linear elastic relation sigma = E*epsilon so that Hooke's
    Law is the ground truth in the training regime.  The same linear formula is
    applied in the held-out regime -- the discrepancy with true post-yield
    behaviour is what the predicates must detect.

    Parameters
    ----------
    n_samples          : number of states to generate
    epsilon_low/high   : strain sampling range
    regime_label       : 'train' or 'held_out'
    rng                : seeded Generator for reproducibility
    noise_frac         : fractional std of Gaussian noise (0.2%)

    Returns
    -------
    DataFrame with columns: F, x, A, L0, sigma, epsilon,
        stress_ratio, strain_energy_ratio, regime,
        valid_linearity, valid_elasticity, valid_small_strain
    """
    if rng is None:
        rng = np.random.default_rng(42)

    epsilon = rng.uniform(epsilon_low, epsilon_high, n_samples)
    A       = rng.uniform(A_LOW,  A_HIGH,  n_samples)
    L0      = rng.uniform(L0_LOW, L0_HIGH, n_samples)

    sigma = E_STEEL * epsilon          # stress (Pa) via Hooke's Law
    F     = sigma * A                  # force (N)
    x     = epsilon * L0               # displacement (m)

    # Small measurement noise
    epsilon = epsilon * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    F       = F       * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    x       = x       * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    A       = A       * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    L0      = L0      * (1.0 + rng.normal(0.0, noise_frac, n_samples))

    # Re-derive engineered features from noisy observables
    epsilon_derived      = np.clip(x / L0, 1e-12, None)
    sigma_derived        = np.clip(F / A,  1e-12, None)
    stress_ratio         = sigma_derived / SIGMA_Y
    strain_energy_ratio  = stress_ratio**2          # = (sigma/sigma_y)^2 = U/U_yield
    strain_energy        = sigma_derived**2 / (2 * E_STEEL)  # J/m^3 (physical U)

    vl, ve, vs = compute_hooke_validity_labels(
        stress_ratio, strain_energy_ratio, epsilon_derived
    )

    return pd.DataFrame({
        "F":                   F,
        "x":                   x,
        "A":                   A,
        "L0":                  L0,
        "sigma":               sigma_derived,
        "epsilon":             epsilon_derived,
        "stress_ratio":        stress_ratio,
        "strain_energy_ratio": strain_energy_ratio,
        "strain_energy":       strain_energy,
        "regime":              regime_label,
        "valid_linearity":     vl,
        "valid_elasticity":    ve,
        "valid_small_strain":  vs,
    })


def generate_hooke_dataset(
    n_train: int = 5000,
    n_held_out: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate training / held-out dataset pair for Hooke's Law.

    Training:  epsilon in [EPSILON_TRAIN_LOW, EPSILON_TRAIN_HIGH]  (elastic regime)
    Held-out:  epsilon in [EPSILON_TEST_LOW,  EPSILON_TEST_HIGH]   (post-yield)

    Returns
    -------
    train_df, held_out_df
    """
    rng = np.random.default_rng(seed)
    train_df = generate_hooke_regime(
        n_train, EPSILON_TRAIN_LOW, EPSILON_TRAIN_HIGH,
        regime_label="train", rng=rng,
    )
    held_out_df = generate_hooke_regime(
        n_held_out, EPSILON_TEST_LOW, EPSILON_TEST_HIGH,
        regime_label="held_out", rng=rng,
    )
    return train_df, held_out_df


# ===========================================================================
# FOURIER HEAT CONDUCTION DOMAIN
# ===========================================================================
#
# Synthetic data for Fourier's Law: q = -k*dT/dx (silicon, 1D conduction).
#
# Four assumptions and their analytical validity criteria:
#
#   A1 (Continuum)          : Kn = lambda/L < KN_THRESHOLD = 0.1
#                             Violated when phonon mean free path is not << L.
#
#   A2 (Steady state)       : Fo = alpha*t/L^2 > FO_THRESHOLD = 1.0
#                             Violated when system hasn't reached thermal equil.
#
#   A3 (Linear response)    : 1.65 * |dT/dx| * L / T < A3_THRESHOLD = 0.1
#                             Violated when k(T) varies significantly across L.
#                             (silicon: k(T) = k0*(T/300)^{-1.65})
#
#   A4 (Local equilibrium)  : t > ELECTRON_PHONON_TIME = 1ps
#                             Violated in ultrafast laser heating (non-equil.).
#
# Training regime  (all assumptions satisfied by construction):
#   L in [1 um, 1 mm]  (log-uniform)
#   T in [200, 600] K
#   t sampled so Fo > 1 and t > 1 ns
#   dT/dx sampled so A3 ratio < 0.1
#
# Held-out regime (three simultaneous failure modes, all labeled):
#   Nanoscale:  L < 100 nm  (Kn > 0.4 >> 0.1 -> A1 violated)
#   High-T:     T > 1000 K  (A3 more easily violated with large gradient)
#   Ultrafast:  t < 10 ps   (A4 violated when t < 1 ps; A2 often violated too)
# ---------------------------------------------------------------------------

# Silicon material constants
SILICON_K0        = 150.0     # W/(m*K) thermal conductivity at 300 K
SILICON_LAMBDA    = 40e-9     # m  phonon mean free path (~40 nm)
SILICON_RHO       = 2329.0    # kg/m^3  density
SILICON_C         = 700.0     # J/(kg*K)  specific heat
SILICON_K_EXP     = -1.65     # k(T) = K0 * (T/300)^K_EXP (empirical fit)

ELECTRON_PHONON_TIME = 1e-12  # s  electron-phonon coupling time (~1 ps Si)

# Validity thresholds
KN_THRESHOLD  = 0.1   # Kn < 0.1  for continuum assumption
FO_THRESHOLD  = 1.0   # Fo > 1    for steady-state assumption
A3_THRESHOLD  = 0.1   # A3_ratio < 0.1  for linear-response assumption

# Training regime bounds
FOURIER_L_TRAIN_LOW  = 1e-6   # 1 um
FOURIER_L_TRAIN_HIGH = 1e-3   # 1 mm
FOURIER_T_TRAIN_LOW  = 200.0  # K
FOURIER_T_TRAIN_HIGH = 600.0  # K
FOURIER_T_MIN        = 1e-9   # 1 ns  (ensures A4 always valid in training)
FOURIER_DTDX_LOW     = 1e3    # K/m
FOURIER_DTDX_HIGH    = 1e9    # K/m

# Held-out regime failure thresholds
FOURIER_L_NANO_MAX   = 100e-9  # 100 nm  (L below this -> A1 fails)
FOURIER_T_HOT_MIN    = 1000.0  # K       (T above this -> hot regime)
FOURIER_T_FAST_MAX   = 10e-12  # 10 ps   (t below this -> ultrafast regime)


def _silicon_k(T: np.ndarray) -> np.ndarray:
    """Effective thermal conductivity of silicon: k(T) = k0*(T/300)^{-1.65}."""
    return SILICON_K0 * (np.clip(T, 1.0, None) / 300.0) ** SILICON_K_EXP


def _silicon_alpha(T: np.ndarray) -> np.ndarray:
    """Thermal diffusivity alpha = k/(rho*c) for silicon."""
    return _silicon_k(T) / (SILICON_RHO * SILICON_C)


def compute_fourier_validity_labels(
    T: np.ndarray,
    L: np.ndarray,
    t: np.ndarray,
    dT_dx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytically compute per-assumption validity labels for Fourier's Law.

    Returns
    -------
    valid_continuum          : A1  Kn = lambda/L < 0.1
    valid_steady_state       : A2  Fo = alpha*t/L^2 > 1
    valid_linear_response    : A3  1.65*|dT_dx|*L/T < 0.1
    valid_local_equilibrium  : A4  t > 1 ps
    """
    Kn       = SILICON_LAMBDA / np.clip(L, 1e-20, None)
    alpha    = _silicon_alpha(T)
    Fo       = alpha * np.clip(t, 0, None) / np.clip(L**2, 1e-40, None)
    A3_ratio = 1.65 * np.abs(dT_dx) * np.clip(L, 0, None) / np.clip(T, 1.0, None)

    return (
        Kn       < KN_THRESHOLD,
        Fo       > FO_THRESHOLD,
        A3_ratio < A3_THRESHOLD,
        t        > ELECTRON_PHONON_TIME,
    )


def generate_fourier_training(
    n_samples: int,
    rng: np.random.Generator,
    noise_frac: float = 0.002,
) -> pd.DataFrame:
    """Sample training states where all four Fourier assumptions are satisfied.

    L is log-uniform in [1 um, 1 mm].  T is uniform in [200, 600] K.
    t is log-uniform in [max(L^2/alpha, 1 ns), 1 s] to guarantee Fo > 1 and A4.
    dT/dx is log-uniform in [1e3, min(1e9, 0.1*T/(1.65*L))] to guarantee A3.
    """
    L = np.exp(rng.uniform(
        np.log(FOURIER_L_TRAIN_LOW), np.log(FOURIER_L_TRAIN_HIGH), n_samples
    ))
    T = rng.uniform(FOURIER_T_TRAIN_LOW, FOURIER_T_TRAIN_HIGH, n_samples)

    alpha   = _silicon_alpha(T)
    t_min   = np.maximum(L**2 / alpha, FOURIER_T_MIN)  # Fo=1 boundary or 1ns
    t_min   = np.minimum(t_min, 0.9)                   # guard: keep below t_max=1s
    log_t   = rng.uniform(np.log(t_min), np.log(np.ones(n_samples)))
    t       = np.exp(log_t)

    dtdx_max = np.minimum(A3_THRESHOLD * T / (1.65 * L), FOURIER_DTDX_HIGH)
    dtdx_max = np.maximum(dtdx_max, FOURIER_DTDX_LOW * 1.01)  # ensure range exists
    log_dtdx = rng.uniform(np.log(FOURIER_DTDX_LOW * np.ones(n_samples)), np.log(dtdx_max))
    dT_dx    = np.exp(log_dtdx)

    # Small measurement noise
    T     = np.clip(T     * (1 + rng.normal(0, noise_frac, n_samples)), 50.0, None)
    L     = np.clip(L     * (1 + rng.normal(0, noise_frac, n_samples)), 1e-11, None)
    t     = np.clip(t     * (1 + rng.normal(0, noise_frac, n_samples)), 1e-15, None)
    dT_dx = np.clip(dT_dx * (1 + rng.normal(0, noise_frac, n_samples)), 1.0, None)

    dT_dt = dT_dx * L / t  # characteristic heating rate
    vc, vs, vl, ve = compute_fourier_validity_labels(T, L, t, dT_dx)

    return pd.DataFrame({
        "T": T, "L": L, "t": t, "dT_dx": dT_dx, "dT_dt": dT_dt,
        "Kn": SILICON_LAMBDA / L,
        "Fo": _silicon_alpha(T) * t / L**2,
        "A3_ratio": 1.65 * dT_dx * L / T,
        "regime": "train",
        "valid_continuum":         vc,
        "valid_steady_state":      vs,
        "valid_linear_response":   vl,
        "valid_local_equilibrium": ve,
    })


def generate_fourier_held_out(
    n_samples: int,
    rng: np.random.Generator,
    noise_frac: float = 0.002,
) -> pd.DataFrame:
    """Sample held-out states covering all three Fourier failure modes.

    Each quarter of the held-out set targets a distinct failure regime so that
    all assumptions and multi-assumption failures are represented:
      - Nanoscale  (L < 100 nm)         A1 violated
      - High-T     (T > 1000 K)         A3 more easily violated
      - Ultrafast  (t < 10 ps)          A4 violated; A2 often violated too
      - Combined   (all three extreme)  A1 + A3 + A4 + A2 violated
    """
    per = n_samples // 4
    remainder = n_samples - 3 * per

    def _regime(n, L_log_range, T_range, t_log_range, dtdx_log_range):
        L     = np.exp(rng.uniform(*L_log_range, n))
        T     = rng.uniform(*T_range, n)
        t     = np.exp(rng.uniform(*t_log_range, n))
        dT_dx = np.exp(rng.uniform(*dtdx_log_range, n))
        T     = np.clip(T     * (1 + rng.normal(0, noise_frac, n)), 50.0, None)
        L     = np.clip(L     * (1 + rng.normal(0, noise_frac, n)), 1e-11, None)
        t     = np.clip(t     * (1 + rng.normal(0, noise_frac, n)), 1e-15, None)
        dT_dx = np.clip(dT_dx * (1 + rng.normal(0, noise_frac, n)), 1.0, None)
        dT_dt = dT_dx * L / t
        vc, vs, vl, ve = compute_fourier_validity_labels(T, L, t, dT_dx)
        return pd.DataFrame({
            "T": T, "L": L, "t": t, "dT_dx": dT_dx, "dT_dt": dT_dt,
            "Kn": SILICON_LAMBDA / L,
            "Fo": _silicon_alpha(T) * t / L**2,
            "A3_ratio": 1.65 * dT_dx * L / T,
            "regime": "held_out",
            "valid_continuum":         vc,
            "valid_steady_state":      vs,
            "valid_linear_response":   vl,
            "valid_local_equilibrium": ve,
        })

    log = np.log
    frames = [
        _regime(per,       (log(1e-9), log(100e-9)),  (200.0, 1500.0),  (log(1e-9), log(1.0)),    (log(1e3), log(1e9))),
        _regime(per,       (log(1e-6), log(1e-3)),    (1000.0, 1500.0), (log(1e-9), log(1.0)),    (log(1e4), log(1e9))),
        _regime(per,       (log(1e-9), log(1e-3)),    (200.0, 1500.0),  (log(1e-15), log(10e-12)),(log(1e3), log(1e9))),
        _regime(remainder, (log(1e-9), log(100e-9)),  (1000.0, 1500.0), (log(1e-15), log(10e-12)),(log(1e6), log(1e9))),
    ]
    return pd.concat(frames, ignore_index=True)


def generate_fourier_dataset(
    n_train: int = 5000,
    n_held_out: int = 3000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate training / held-out dataset pair for Fourier heat conduction.

    Training:  all four Fourier assumptions satisfied by construction.
    Held-out:  three failure regimes (nanoscale, high-T, ultrafast) + combined.
    Each held-out point has labeled per-assumption validity flags.

    Returns
    -------
    train_df, held_out_df
    """
    rng = np.random.default_rng(seed)
    train_df    = generate_fourier_training(n_train, rng)
    held_out_df = generate_fourier_held_out(n_held_out, rng)
    return train_df, held_out_df


# ===========================================================================
# RESIDUAL-BASED IDEAL GAS DOMAIN (Pilot 1 Reformulation)
# ===========================================================================
#
# Generates (P, V, T, n) samples using the van der Waals EOS for CO2.
# The predicate trained on this data never sees a, b, or any derived
# criterion — only raw observables and the ideal-gas residual |PV/nRT − 1|.
# ---------------------------------------------------------------------------

CO2_A_VDW = 3.592    # L^2 atm / mol^2  (CO2 van der Waals attraction)
CO2_B_VDW = 0.04267  # L / mol           (CO2 van der Waals excluded volume)


def generate_ideal_gas_residual_data(
    n_samples: int,
    P_range: tuple,
    T_range: tuple = (300, 500),
    n_range: tuple = (0.5, 2.0),
    seed=None,
) -> dict:
    """Generate (P, V, T, n) samples via the CO2 van der Waals EOS.

    Samples P, T, n uniformly, then solves for the real volume V using the
    van der Waals cubic. This ensures that at high pressure the samples
    reflect real-gas behavior, not ideal-gas behavior.

    Returns the ideal-gas residual |PV/nRT − 1|, which measures how far the
    sample deviates from PV = nRT without requiring knowledge of a, b, or any
    validity threshold.

    The predicate trained on this data is never given CO2_A_VDW, CO2_B_VDW,
    or any breakdown criterion — only raw [P, V, T, n] observables.

    Parameters
    ----------
    n_samples : number of samples to attempt (some may be dropped for NaN)
    P_range   : (P_min, P_max) in atm
    T_range   : (T_min, T_max) in K
    n_range   : (n_min, n_max) in mol
    seed      : RNG seed for reproducibility

    Returns
    -------
    dict with keys P, V, T, n, residual (1-D numpy arrays, same length).
    """
    rng = np.random.default_rng(seed)

    P = rng.uniform(*P_range, size=n_samples)
    T = rng.uniform(*T_range, size=n_samples)
    n = rng.uniform(*n_range, size=n_samples)

    # Solve V from van der Waals EOS for each (P, T, n):
    # P V^3 - (n b P + n R T) V^2 + a n^2 V - a b n^3 = 0
    V = np.empty_like(P)
    for i in range(n_samples):
        coeffs = [
            P[i],
            -(n[i] * CO2_B_VDW * P[i] + n[i] * R * T[i]),
            CO2_A_VDW * n[i] ** 2,
            -CO2_A_VDW * CO2_B_VDW * n[i] ** 3,
        ]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        # Physical root: largest positive real root greater than n*b (excluded volume)
        valid_roots = real_roots[real_roots > n[i] * CO2_B_VDW]
        V[i] = float(np.max(valid_roots)) if len(valid_roots) > 0 else np.nan

    # Drop degenerate samples where root-finding failed
    mask = ~np.isnan(V)
    P, T, n, V = P[mask], T[mask], n[mask], V[mask]

    residual = np.abs((P * V) / (n * R * T) - 1.0)

    return {"P": P, "V": V, "T": T, "n": n, "residual": residual}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df, held_out_df = generate_dataset(n_train=5000, n_held_out=2000)

    for label, df in [("Training", train_df), ("Held-out", held_out_df)]:
        n = len(df)
        pp = df["valid_point_particle"].mean()
        nf = df["valid_no_forces"].mean()
        p_range = f"{df['P'].min():.1f} – {df['P'].max():.1f}"
        print(f"{label} ({n} samples, P = {p_range} atm)")
        print(f"  valid_point_particle : {pp:.1%}")
        print(f"  valid_no_forces      : {nf:.1%}")
        print()

    visualize_regime_boundary(train_df, held_out_df)
