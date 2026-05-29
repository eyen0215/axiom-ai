"""
Training loop for validity predicates.

Soft-label design
-----------------
Training data (P = 1–10 atm) contains only physically valid states, so binary
labels would be 100 % positive — useless for learning a boundary.  Instead we
use continuous soft labels derived from the analytical validity criterion:

    A1 (point particles)
        r  = (V/n) / VDW_B          (free-volume ratio)
        soft_A1 = r / (r + FREE_VOL_THRESHOLD)
                = sigmoid(log(r / FREE_VOL_THRESHOLD))
        → equals 0.5 at the binary threshold; is a sigmoid of log(V)

    A2 (no intermolecular forces)
        q  = VDW_A·n / (V·R·T)      (interaction-energy ratio)
        soft_A2 = FORCE_THRESHOLD / (q + FORCE_THRESHOLD)
                = sigmoid(log(FORCE_THRESHOLD / q))
        → equals 0.5 at the binary threshold; is a sigmoid of −log(V) − log(T)

Because the MLP uses log(P) and log(V) as inputs and finishes with a sigmoid,
it learns to represent these functions as linear-in-logit, and extrapolates
correctly: very small V (high P, held-out regime) → score well below 0.5.

A3 and A4 have no operationalizable criterion from macroscopic (P, V, T, n)
observables; their training is skipped.

Fits into the system: called from experiments/pilot.py; reads DataFrames from
data/generate.py; attaches trained predicates to assumption nodes in the axiom
graph via node.attach_predicate(); plot_decision_boundary() visualises where
the learned boundary sits in P–V space relative to the true criterion boundary.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.generate import (
    R,
    VDW_A,
    VDW_B,
    FREE_VOL_THRESHOLD,
    FORCE_THRESHOLD,
    P_TRAIN_HIGH,
    P_TRAIN_LOW,
    P_TEST_HIGH,
    P_TEST_LOW,
    T_LOW,
    T_HIGH,
)
from validity_predicates.predicate import ValidityPredicate, FEATURE_COLS, _LOG_COLS

# Only A1 and A2 have operationalizable soft labels from (P, V, T, n).
TRAINABLE_ASSUMPTIONS = frozenset({"A1_point_particles", "A2_no_forces"})


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def make_features(df: pd.DataFrame) -> np.ndarray:
    """Return raw (P, V, T, n) feature matrix, shape (N, 4), dtype float32."""
    return df[FEATURE_COLS].values.astype(np.float32)


def _log_transform(X: np.ndarray) -> np.ndarray:
    """Apply log to columns that correspond to _LOG_COLS (P and V)."""
    X = X.copy()
    for col in _LOG_COLS:
        X[:, col] = np.log(np.clip(X[:, col], 1e-9, None))
    return X


# ---------------------------------------------------------------------------
# Soft-label computation
# ---------------------------------------------------------------------------

def compute_log_criterion(
    df: pd.DataFrame,
    assumption_id: str,
) -> Optional[np.ndarray]:
    """Return the log-criterion target for regression training.

    The target is log(criterion / threshold), which is:
        >  0  when the assumption is satisfied   (training regime: all > 1.4)
        == 0  at the binary validity threshold   (score = sigmoid(0) = 0.5)
        <  0  when the assumption is violated    (held-out regime: can reach -2)

    This gives the network a well-conditioned regression problem with real
    variation in the training range ([1.4, 4.8] for both A1 and A2), in
    contrast to soft-label targets which are compressed into [0.86, 0.99] and
    produce near-zero gradients throughout training.

    Because the target is log-linear in log(V) (and log(T) for A2), the MLP
    learns a nearly-linear function that extrapolates correctly: very small V
    (high-pressure held-out states) gives large negative output → score well
    below 0.5 → assumption correctly flagged.

    Returns None for assumptions with no operationalizable criterion.

    Formulas
    --------
    A1: target = log(r / FREE_VOL_THRESHOLD)   where r = (V/n) / VDW_B
        Training range: log(63/10) to log(1050/10) ≈ [1.84, 4.65]

    A2: target = log(FORCE_THRESHOLD / q)   where q = VDW_A·n / (V·R·T)
        Training range: log(0.10/0.023) to log(0.10/0.00083) ≈ [1.47, 4.80]
    """
    V = df["V"].values
    n = df["n"].values
    T = df["T"].values

    if assumption_id == "A1_point_particles":
        r = (V / n) / VDW_B
        return np.log(r / FREE_VOL_THRESHOLD).astype(np.float32)

    if assumption_id == "A2_no_forces":
        q = VDW_A * n / (V * R * T)
        return np.log(FORCE_THRESHOLD / q).astype(np.float32)

    return None  # A3_elastic_collisions, A4_thermal_equilibrium: not operationalizable


# Keep the old name as an alias for plotting (returns soft probability, not logit)
def compute_soft_labels(df: pd.DataFrame, assumption_id: str) -> Optional[np.ndarray]:
    """Return sigmoid(log-criterion) — soft labels in (0,1) for plotting."""
    log_c = compute_log_criterion(df, assumption_id)
    if log_c is None:
        return None
    return (1.0 / (1.0 + np.exp(-log_c))).astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_predicate(
    assumption_id: str,
    train_df: pd.DataFrame,
    *,
    hidden_dims: tuple = (32, 16),
    lr: float = 1e-2,
    n_epochs: int = 600,
    val_frac: float = 0.15,
    patience: int = 40,
    seed: int = 0,
    verbose: bool = False,
) -> Optional[ValidityPredicate]:
    """Train a ValidityPredicate for one assumption on low-pressure training data.

    Parameters
    ----------
    assumption_id : node ID of the assumption to train (e.g. 'A1_point_particles')
    train_df      : low-pressure DataFrame from data.generate.generate_dataset()
    hidden_dims   : MLP hidden layer widths
    lr            : Adam learning rate
    n_epochs      : maximum training epochs
    val_frac      : fraction of training data held for early-stopping validation
    patience      : epochs without val-loss improvement before stopping
    seed          : RNG seed for reproducibility
    verbose       : print progress every 100 epochs

    Returns
    -------
    Trained ValidityPredicate, or None if the assumption has no soft label.
    """
    log_targets = compute_log_criterion(train_df, assumption_id)
    if log_targets is None:
        return None

    X_raw = make_features(train_df)
    y_all = log_targets

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_raw))
    n_val = max(1, int(val_frac * len(X_raw)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_raw, y_tr = X_raw[tr_idx], y_all[tr_idx]
    X_val_raw, y_val = X_raw[val_idx], y_all[val_idx]

    # ------------------------------------------------------------------
    # Fit normalisation on log-transformed training features
    # ------------------------------------------------------------------
    X_tr_log = _log_transform(X_tr_raw)
    feat_mean = X_tr_log.mean(axis=0).astype(np.float32)
    feat_std  = (X_tr_log.std(axis=0) + 1e-8).astype(np.float32)

    predicate = ValidityPredicate(hidden_dims=hidden_dims)
    predicate.set_normalization(feat_mean, feat_std)

    # ------------------------------------------------------------------
    # Optimisation (full-batch; dataset is small)
    # ------------------------------------------------------------------
    # The skip connection is the linear extrapolation path and should be
    # free to grow.  The MLP sees only positive targets (all training states
    # are valid), so without regularisation it collapses to a large positive
    # constant that overwhelms the skip on out-of-distribution inputs.
    # Strong L2 on MLP weights keeps it near zero; the skip handles the trend.
    optimizer = torch.optim.Adam([
        {"params": predicate.skip.parameters(), "weight_decay": 0.0},
        {"params": predicate.mlp.parameters(),  "weight_decay": 5.0},
    ], lr=lr)
    loss_fn   = nn.MSELoss()

    X_tr_t  = torch.from_numpy(X_tr_raw)
    y_tr_t  = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val_raw)
    y_val_t = torch.from_numpy(y_val)

    best_val  = float("inf")
    best_sd   = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        predicate.train()
        optimizer.zero_grad()
        loss = loss_fn(predicate(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

        predicate.eval()
        with torch.no_grad():
            val_loss = loss_fn(predicate(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_sd  = copy.deepcopy(predicate.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [{assumption_id}] early stop @ epoch {epoch}  "
                          f"best val MSE (logit) = {best_val:.6f}")
                break

        if verbose and epoch % 100 == 0:
            print(f"  [{assumption_id}]  epoch {epoch:4d}  "
                  f"train={loss.item():.5f}  val={val_loss:.5f}")

    if best_sd is not None:
        predicate.load_state_dict(best_sd)

    return predicate


def train_all_predicates(
    graph,
    train_df: pd.DataFrame,
    verbose: bool = True,
    **train_kwargs,
) -> Dict[str, ValidityPredicate]:
    """Train one ValidityPredicate per trainable assumption; attach to the graph.

    Iterates over all assumption nodes in `graph`, trains a predicate for each
    operationalizable one, and calls node.attach_predicate() so that the axiom
    graph is ready for inference without additional wiring.

    A3 (elastic collisions) and A4 (thermal equilibrium) are skipped — their
    validity is not distinguishable from macroscopic (P, V, T, n) alone.

    Returns
    -------
    Dict mapping assumption node ID → trained ValidityPredicate.
    """
    predicates: Dict[str, ValidityPredicate] = {}

    for node in graph.assumption_nodes():
        if verbose:
            print(f"Training predicate for  {node.id} ...")
        pred = train_predicate(node.id, train_df, verbose=verbose, **train_kwargs)
        if pred is None:
            if verbose:
                print(f"  -> skipped (no operationalizable criterion)\n")
            continue
        node.attach_predicate(pred)
        predicates[node.id] = pred
        if verbose:
            print(f"  -> done\n")

    return predicates


# ===========================================================================
# HOOKE'S LAW DOMAIN -- training functions
# ===========================================================================

from validity_predicates.predicate import (
    HOOKE_FEATURE_COLS, HOOKE_N_FEATURES, HOOKE_LOG_COLS,
)
from data.generate import (
    STRESS_RATIO_THRESHOLD, STRAIN_ENERGY_THRESHOLD, EPSILON_THRESHOLD,
)

HOOKE_TRAINABLE_ASSUMPTIONS = frozenset({
    "A1_linearity", "A2_elasticity", "A3_small_strain"
})

# Each predicate observes only the feature directly measured by its criterion.
# Using all three features causes a multicollinearity sign-flip in the skip:
# stress_ratio, strain_energy_ratio, and epsilon are perfectly correlated in
# the elastic training regime (all monotone in epsilon), so the linear skip
# may assign a positive weight to strain_energy_ratio that fires the wrong
# way when strain_energy_ratio is 1000x larger in the held-out regime.
HOOKE_ASSUMPTION_FEATURES: Dict[str, list] = {
    "A1_linearity":    ["stress_ratio"],
    "A2_elasticity":   ["strain_energy_ratio"],
    "A3_small_strain": ["epsilon"],
}


def make_hooke_features(df: pd.DataFrame) -> np.ndarray:
    """Return [stress_ratio, strain_energy_ratio, epsilon] matrix, shape (N, 3)."""
    return df[HOOKE_FEATURE_COLS].values.astype(np.float32)


def compute_hooke_log_criterion(
    df: pd.DataFrame,
    assumption_id: str,
) -> Optional[np.ndarray]:
    """Return log-criterion regression target for one Hooke's Law assumption.

    Target = log(threshold / feature_value):
        > 0  when assumption is satisfied  (all training states)
        = 0  at the validity boundary      (sigmoid gives 0.5)
        < 0  when assumption is violated   (held-out, post-yield regime)

    Features used are already dimensionless ratios so NO log-transform is
    applied to the inputs before passing to the network.  The log() here is
    only on the TARGET (regression label), not the feature.

    Formulas
    --------
    A1 (linearity):
        target = log(STRESS_RATIO_THRESHOLD / stress_ratio)
        Training range: [log(0.9/0.50), log(0.9/0.04)] ~ [0.59, 3.11]

    A2 (elasticity):
        target = log(STRAIN_ENERGY_THRESHOLD / strain_energy_ratio)
        where strain_energy_ratio = (sigma/sigma_y)^2
        Training range: [log(0.8/0.25), log(0.8/0.0016)] ~ [1.16, 6.21]

    A3 (small strain):
        target = log(EPSILON_THRESHOLD / epsilon)
        Training range: [log(0.85*ey/0.625*ey), log(0.85*ey/0.05*ey)] ~ [0.31, 2.83]

    Returns None for A4 (homogeneity), which has no macroscopic criterion.
    """
    if assumption_id == "A1_linearity":
        r = df["stress_ratio"].values.clip(1e-9)
        return np.log(STRESS_RATIO_THRESHOLD / r).astype(np.float32)

    if assumption_id == "A2_elasticity":
        u = df["strain_energy_ratio"].values.clip(1e-9)
        return np.log(STRAIN_ENERGY_THRESHOLD / u).astype(np.float32)

    if assumption_id == "A3_small_strain":
        eps = df["epsilon"].values.clip(1e-12)
        return np.log(EPSILON_THRESHOLD / eps).astype(np.float32)

    return None  # A4_homogeneity: not operationalizable


def train_hooke_predicate(
    assumption_id: str,
    train_df: pd.DataFrame,
    *,
    hidden_dims: tuple = (32, 16),
    lr: float = 1e-2,
    n_epochs: int = 600,
    val_frac: float = 0.15,
    patience: int = 80,
    seed: int = 0,
    verbose: bool = False,
) -> Optional[ValidityPredicate]:
    """Train a ValidityPredicate for one Hooke's Law assumption.

    Architecture differences from the ideal gas predicate:
    - Features: [stress_ratio, strain_energy_ratio, epsilon]  (n_features=3)
    - log_transform_cols=()  -- NO log-transform of inputs.
      The validity boundaries are LINEAR in these dimensionless features
      (not log-linear as in the ideal gas case), so the skip connection
      extrapolates correctly without a log-transform.
    - MLP weight_decay=5.0, skip weight_decay=0.0  -- same regularisation
      strategy as ideal gas (prevents MLP constant-offset collapse).

    Returns None for A4_homogeneity (no operationalizable criterion).
    """
    log_targets = compute_hooke_log_criterion(train_df, assumption_id)
    if log_targets is None:
        return None

    feat_cols = HOOKE_ASSUMPTION_FEATURES.get(assumption_id)
    if feat_cols is None:
        return None

    X_raw = train_df[feat_cols].values.astype(np.float32)
    y_all = log_targets

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_raw))
    n_val = max(1, int(val_frac * len(X_raw)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_raw, y_tr = X_raw[tr_idx], y_all[tr_idx]
    X_val_raw, y_val = X_raw[val_idx], y_all[val_idx]

    # Normalise on raw (non-log-transformed) training features
    feat_mean = X_tr_raw.mean(axis=0).astype(np.float32)
    feat_std  = (X_tr_raw.std(axis=0) + 1e-8).astype(np.float32)

    predicate = ValidityPredicate(
        hidden_dims=hidden_dims,
        n_features=len(feat_cols),
        log_transform_cols=HOOKE_LOG_COLS,
        feature_cols=feat_cols,
    )
    predicate.set_normalization(feat_mean, feat_std)

    optimizer = torch.optim.Adam([
        {"params": predicate.skip.parameters(), "weight_decay": 0.0},
        {"params": predicate.mlp.parameters(),  "weight_decay": 5.0},
    ], lr=lr)
    loss_fn = nn.MSELoss()

    X_tr_t  = torch.from_numpy(X_tr_raw)
    y_tr_t  = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val_raw)
    y_val_t = torch.from_numpy(y_val)

    best_val   = float("inf")
    best_sd    = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        predicate.train()
        optimizer.zero_grad()
        loss = loss_fn(predicate(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

        predicate.eval()
        with torch.no_grad():
            val_loss = loss_fn(predicate(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val   = val_loss
            best_sd    = copy.deepcopy(predicate.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [{assumption_id}] early stop @ epoch {epoch}  "
                          f"best val MSE = {best_val:.6f}")
                break

        if verbose and epoch % 100 == 0:
            print(f"  [{assumption_id}]  epoch {epoch:4d}  "
                  f"train={loss.item():.5f}  val={val_loss:.5f}")

    if best_sd is not None:
        predicate.load_state_dict(best_sd)

    return predicate


def train_all_hooke_predicates(
    graph,
    train_df: pd.DataFrame,
    verbose: bool = True,
    **train_kwargs,
) -> Dict[str, ValidityPredicate]:
    """Train one ValidityPredicate per trainable Hooke's Law assumption.

    Trains A1 (linearity), A2 (elasticity), A3 (small strain).
    Skips A4 (homogeneity) -- no macroscopic criterion.
    Attaches each trained predicate to its assumption node in the graph.
    """
    predicates: Dict[str, ValidityPredicate] = {}

    for node in graph.assumption_nodes():
        if verbose:
            print(f"  Training predicate for  {node.id} ...")
        pred = train_hooke_predicate(node.id, train_df, verbose=verbose, **train_kwargs)
        if pred is None:
            if verbose:
                print(f"    -> skipped (no operationalizable criterion)\n")
            continue
        node.attach_predicate(pred)
        predicates[node.id] = pred
        if verbose:
            print(f"    -> done\n")

    return predicates


# ===========================================================================
# FOURIER HEAT CONDUCTION DOMAIN -- three-attempt training
# ===========================================================================
#
# Three feature-set strategies are trained and compared:
#
#   Attempt 1 (criterion)     : each predicate receives only its scalar
#                               criterion (Kn, Fo, A3_ratio, t), log-transformed.
#                               Minimal information -- same strategy as Hooke's.
#
#   Attempt 2 (observables)   : each predicate receives all five observables
#                               (T, L, t, dT_dx, dT_dt), all log-transformed.
#                               Forces the skip to discover the boundary from
#                               raw measurements.  All Fourier criteria are
#                               log-linear in log-observables, so the skip can
#                               represent them exactly -- IF there is no
#                               collinearity in training.
#
#   Attempt 3 (criterion+obs) : criterion scalar + all five observables.
#                               Provides both the direct signal and context.
#
# See DECISIONS.md for analysis of which attempts succeed and why.
# ---------------------------------------------------------------------------

from data.generate import (
    SILICON_LAMBDA, SILICON_K0, SILICON_RHO, SILICON_C, SILICON_K_EXP,
    ELECTRON_PHONON_TIME,
    KN_THRESHOLD, FO_THRESHOLD, A3_THRESHOLD,
    _silicon_alpha,
)

FOURIER_TRAINABLE_ASSUMPTIONS = frozenset({
    "A1_continuum", "A2_steady_state", "A3_linear_response", "A4_local_equilibrium"
})

# Observable column names as stored in the DataFrame
FOURIER_OBS_COLS = ["T", "L", "t", "dT_dx", "dT_dt"]

# Per-assumption criterion scalar columns
FOURIER_CRITERION_COLS: Dict[str, list] = {
    "A1_continuum":          ["Kn"],
    "A2_steady_state":       ["Fo"],
    "A3_linear_response":    ["A3_ratio"],
    "A4_local_equilibrium":  ["t"],
}

# Criterion + all observables
FOURIER_CRITERION_OBS_COLS: Dict[str, list] = {
    aid: FOURIER_CRITERION_COLS[aid] + FOURIER_OBS_COLS
    for aid in FOURIER_CRITERION_COLS
}


def compute_fourier_log_criterion(
    df: pd.DataFrame,
    assumption_id: str,
) -> Optional[np.ndarray]:
    """Return log-criterion regression target for one Fourier assumption.

    Target = log(criterion_value / threshold) or log(value / boundary):
        > 0  when assumption is satisfied (all training states)
        = 0  at the validity boundary
        < 0  when assumption is violated (held-out states)

    Formulas
    --------
    A1: log(KN_THRESHOLD / Kn) = log(0.1 * L / lambda)
    A2: log(Fo / FO_THRESHOLD) = log(alpha*t / L^2)
    A3: log(A3_THRESHOLD / A3_ratio) = log(0.1 * T / (1.65 * |dT_dx| * L))
    A4: log(t / ELECTRON_PHONON_TIME) = log(t / 1e-12)
    """
    if assumption_id == "A1_continuum":
        Kn = SILICON_LAMBDA / df["L"].values.clip(1e-20)
        return np.log(KN_THRESHOLD / Kn).astype(np.float32)

    if assumption_id == "A2_steady_state":
        alpha = _silicon_alpha(df["T"].values)
        Fo    = alpha * df["t"].values / df["L"].values.clip(1e-20)**2
        return np.log(Fo / FO_THRESHOLD + 1e-30).astype(np.float32)

    if assumption_id == "A3_linear_response":
        ratio = 1.65 * df["dT_dx"].values.clip(1e-30) * df["L"].values.clip(1e-20) / df["T"].values.clip(1.0)
        return np.log(A3_THRESHOLD / ratio + 1e-30).astype(np.float32)

    if assumption_id == "A4_local_equilibrium":
        return np.log(df["t"].values.clip(1e-30) / ELECTRON_PHONON_TIME).astype(np.float32)

    return None


def _train_fourier_predicate_core(
    assumption_id: str,
    feat_cols: list,
    log_transform_cols: tuple,
    train_df: pd.DataFrame,
    *,
    hidden_dims: tuple = (32, 16),
    lr: float = 1e-2,
    n_epochs: int = 600,
    val_frac: float = 0.15,
    patience: int = 60,
    seed: int = 0,
    verbose: bool = False,
) -> Optional[ValidityPredicate]:
    """Shared training loop used by all three Fourier attempts.

    Parameters
    ----------
    feat_cols          : which DataFrame columns to use as features
    log_transform_cols : indices within feat_cols to log-transform in forward()
    """
    log_targets = compute_fourier_log_criterion(train_df, assumption_id)
    if log_targets is None:
        return None

    X_raw = train_df[feat_cols].values.astype(np.float32)
    y_all = log_targets

    torch.manual_seed(seed)
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_raw))
    n_val = max(1, int(val_frac * len(X_raw)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_raw, y_tr = X_raw[tr_idx], y_all[tr_idx]
    X_val_raw, y_val = X_raw[val_idx], y_all[val_idx]

    # Normalise on (optionally log-transformed) training features
    X_tr_log = X_tr_raw.copy()
    for col in log_transform_cols:
        X_tr_log[:, col] = np.log(np.clip(X_tr_raw[:, col], 1e-30, None))
    feat_mean = X_tr_log.mean(axis=0).astype(np.float32)
    feat_std  = (X_tr_log.std(axis=0) + 1e-8).astype(np.float32)

    predicate = ValidityPredicate(
        hidden_dims=hidden_dims,
        n_features=len(feat_cols),
        log_transform_cols=log_transform_cols,
        feature_cols=feat_cols,
    )
    predicate.set_normalization(feat_mean, feat_std)

    optimizer = torch.optim.Adam([
        {"params": predicate.skip.parameters(), "weight_decay": 0.0},
        {"params": predicate.mlp.parameters(),  "weight_decay": 5.0},
    ], lr=lr)
    loss_fn = nn.MSELoss()

    X_tr_t  = torch.from_numpy(X_tr_raw)
    y_tr_t  = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val_raw)
    y_val_t = torch.from_numpy(y_val)

    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(1, n_epochs + 1):
        predicate.train()
        optimizer.zero_grad()
        loss_fn(predicate(X_tr_t), y_tr_t).backward()
        optimizer.step()

        predicate.eval()
        with torch.no_grad():
            val_loss = loss_fn(predicate(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val, best_sd, no_improve = val_loss, copy.deepcopy(predicate.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    [{assumption_id}] early stop @ epoch {epoch}  "
                          f"best val MSE = {best_val:.6f}")
                break

        if verbose and epoch % 100 == 0:
            print(f"    [{assumption_id}]  epoch {epoch:4d}  "
                  f"train={loss_fn(predicate(X_tr_t), y_tr_t).item():.5f}  "
                  f"val={val_loss:.5f}")

    if best_sd is not None:
        predicate.load_state_dict(best_sd)
    return predicate


def train_all_fourier_predicates(
    graph,
    train_df: pd.DataFrame,
    attempt: int = 1,
    verbose: bool = True,
    **train_kwargs,
) -> Dict[str, ValidityPredicate]:
    """Train one ValidityPredicate per Fourier assumption, using the given attempt.

    Parameters
    ----------
    attempt : 1 = per-criterion scalar  (Kn, Fo, A3_ratio, t)
              2 = all observables        (T, L, t, dT_dx, dT_dt)
              3 = criterion + observables

    Returns
    -------
    Dict mapping assumption_id -> trained ValidityPredicate.
    """
    # All Fourier criteria are log-linear in log-features, so log-transform all.
    # For attempt 1: 1 feature (the criterion scalar), log-transform col 0.
    # For attempt 2: 5 features (observables), log-transform all.
    # For attempt 3: 6 features (criterion + 5 observables), log-transform all.

    predicates: Dict[str, ValidityPredicate] = {}

    for node in graph.assumption_nodes():
        aid = node.id
        if verbose:
            print(f"  [{aid}]", end=" ", flush=True)

        if attempt == 1:
            feat_cols = FOURIER_CRITERION_COLS.get(aid)
            log_cols  = tuple(range(len(feat_cols))) if feat_cols else ()
        elif attempt == 2:
            feat_cols = FOURIER_OBS_COLS
            log_cols  = tuple(range(len(FOURIER_OBS_COLS)))
        else:  # attempt 3
            feat_cols = FOURIER_CRITERION_OBS_COLS.get(aid, FOURIER_OBS_COLS)
            log_cols  = tuple(range(len(feat_cols)))

        if feat_cols is None:
            if verbose:
                print("skipped (no criterion)")
            continue

        pred = _train_fourier_predicate_core(
            aid, feat_cols, log_cols, train_df,
            verbose=False, **train_kwargs,
        )
        if pred is None:
            if verbose:
                print("skipped (no criterion)")
            continue

        node.attach_predicate(pred)
        predicates[aid] = pred
        if verbose:
            # Diagnostic: report skip weights for the primary feature
            w = pred.skip.weight.data.numpy().ravel()
            print(f"done  skip_w={w}")

    return predicates


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    predicates: Dict[str, ValidityPredicate],
    train_df: pd.DataFrame,
    held_out_df: pd.DataFrame,
    T_ref: float = 400.0,
    n_ref: float = 1.0,
    save_path: Optional[str] = None,
) -> None:
    """Plot the learned decision boundary for each trained predicate.

    Figure layout (one row per assumption):
      Left  — 2D heat-map of predicted validity score in log P–log V space.
               Navy contour = learned decision boundary (score = 0.5).
               Black dashed contour = analytical soft-label boundary (ground truth).
               White dotted line = regime split at P = P_TRAIN_HIGH.
               Scattered points = training (circles) and held-out (squares),
               coloured blue (analytically valid) / red (analytically invalid).
      Right — 1D score-vs-P curve along the ideal-gas isotherm V = nRT/P at
               T = T_ref.  Compares predicted score with analytical soft label;
               shows the regime-split and decision-threshold lines.

    Parameters
    ----------
    predicates   : dict returned by train_all_predicates()
    train_df     : low-pressure training DataFrame
    held_out_df  : held-out high-pressure DataFrame
    T_ref        : reference temperature for the 1-D isotherm slice (K)
    n_ref        : reference moles for the 1-D isotherm slice
    save_path    : if not None, save figure to this path (PNG/PDF/SVG)
    """
    _LABEL_COL = {
        "A1_point_particles": "valid_point_particle",
        "A2_no_forces":       "valid_no_forces",
    }
    _TITLES = {
        "A1_point_particles": "A1 — Point particles (no molecular volume)\n"
                              r"criterion: $(V/n)\,/\,b > 10$",
        "A2_no_forces":       "A2 — No intermolecular forces\n"
                              r"criterion: $a\,n\,/\,(V R T) < 0.10$",
    }

    trainable = [aid for aid in ("A1_point_particles", "A2_no_forces")
                 if aid in predicates]
    if not trainable:
        print("No trained predicates to plot.")
        return

    n_rows = len(trainable)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # ---- 2D grid (log-log P–V space) ----
    P_grid = np.logspace(np.log10(0.4), np.log10(350), 200)
    V_grid = np.logspace(np.log10(0.04), np.log10(100), 160)
    PP, VV = np.meshgrid(P_grid, V_grid)
    X_grid = np.column_stack([
        PP.ravel(),
        VV.ravel(),
        np.full(PP.size, T_ref),
        np.full(PP.size, n_ref),
    ]).astype(np.float32)
    grid_df = pd.DataFrame({"P": PP.ravel(), "V": VV.ravel(),
                             "T": T_ref, "n": n_ref})

    # ---- 1D isotherm (V = nRT/P) ----
    P_line = np.logspace(np.log10(0.4), np.log10(250), 500)
    V_line = (n_ref * R * T_ref) / P_line
    X_line = np.column_stack([
        P_line, V_line,
        np.full(len(P_line), T_ref),
        np.full(len(P_line), n_ref),
    ]).astype(np.float32)
    line_df = pd.DataFrame({"P": P_line, "V": V_line, "T": T_ref, "n": n_ref})

    # colour norm centred on decision threshold
    cnorm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    for row, assumption_id in enumerate(trainable):
        predicate   = predicates[assumption_id]
        lbl_col     = _LABEL_COL[assumption_id]
        ax_2d       = axes[row, 0]
        ax_1d       = axes[row, 1]

        # ------ 2D heatmap -----------------------------------------------
        scores_2d  = predicate.predict(X_grid).reshape(PP.shape)
        analytic_2d = compute_soft_labels(grid_df, assumption_id).reshape(PP.shape)

        im = ax_2d.contourf(PP, VV, scores_2d, levels=60,
                             cmap="RdYlGn", norm=cnorm)
        cb = fig.colorbar(im, ax=ax_2d, label="Predicted validity score")
        cb.ax.axhline(0.5, color="navy", linewidth=1.5)

        # Learned decision boundary (score = 0.5)
        try:
            cs_pred = ax_2d.contour(PP, VV, scores_2d,
                                    levels=[0.5], colors=["navy"],
                                    linewidths=2.5)
            ax_2d.clabel(cs_pred, fmt={0.5: "learned  0.5"}, fontsize=7)
        except Exception:
            pass

        # Analytical boundary (soft label = 0.5)
        try:
            cs_true = ax_2d.contour(PP, VV, analytic_2d,
                                    levels=[0.5], colors=["black"],
                                    linewidths=1.8, linestyles="--")
            ax_2d.clabel(cs_true, fmt={0.5: "analytic  0.5"}, fontsize=7)
        except Exception:
            pass

        # Regime split
        ax_2d.axvline(P_TRAIN_HIGH, color="white", linewidth=1.8, linestyle=":",
                      zorder=5)
        ax_2d.text(P_TRAIN_HIGH * 1.05, V_grid[-1] * 0.6,
                   f"P={P_TRAIN_HIGH:.0f} atm\n(regime split)",
                   color="white", fontsize=7, va="top")

        # Scatter training and held-out points (sampled for clarity)
        rng = np.random.default_rng(42)
        tr_idx = rng.choice(len(train_df),    size=min(600, len(train_df)),    replace=False)
        ho_idx = rng.choice(len(held_out_df), size=min(300, len(held_out_df)), replace=False)

        for sub, idx, marker in [(train_df, tr_idx, "o"), (held_out_df, ho_idx, "s")]:
            sub = sub.iloc[idx]
            valid = sub[lbl_col].values
            ax_2d.scatter(sub.loc[valid,  "P"], sub.loc[valid,  "V"],
                          s=5, color="deepskyblue", alpha=0.6, marker=marker)
            ax_2d.scatter(sub.loc[~valid, "P"], sub.loc[~valid, "V"],
                          s=5, color="red",         alpha=0.6, marker=marker)

        ax_2d.set_xscale("log");  ax_2d.set_yscale("log")
        ax_2d.set_xlabel("Pressure  P  (atm)")
        ax_2d.set_ylabel("Volume  V  (L)")
        ax_2d.set_title(_TITLES[assumption_id]
                        + f"\n2-D predicted score  (T = {T_ref:.0f} K fixed)",
                        fontsize=9)

        # Legend proxies
        from matplotlib.lines import Line2D
        ax_2d.legend(handles=[
            Line2D([0],[0], color="navy",  lw=2.5, label="Learned boundary (score=0.5)"),
            Line2D([0],[0], color="black", lw=1.8, ls="--", label="Analytic boundary"),
            Line2D([0],[0], color="white", lw=1.8, ls=":",  label="Regime split"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="deepskyblue",
                   markersize=5, label="Valid states"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red",
                   markersize=5, label="Invalid states"),
        ], fontsize=7, loc="lower right")

        # ------ 1D isotherm curve ----------------------------------------
        scores_1d   = predicate.predict(X_line)
        analytic_1d = compute_soft_labels(line_df, assumption_id)

        ax_1d.plot(P_line, scores_1d,   color="steelblue", lw=2.0,
                   label="Predicted score (MLP)")
        ax_1d.plot(P_line, analytic_1d, color="black", lw=1.5, ls="--",
                   label="Analytical soft label (ground truth)")
        ax_1d.axhline(0.5, color="red", lw=1.2, ls=":",
                      label="Decision threshold  0.5")
        ax_1d.axvline(P_TRAIN_HIGH, color="gray", lw=1.2, ls="--",
                      label=f"Regime split  P = {P_TRAIN_HIGH:.0f} atm")

        # Shade training and held-out bands
        ax_1d.axvspan(P_TRAIN_LOW, P_TRAIN_HIGH, alpha=0.06, color="steelblue",
                      label="Training regime")
        ax_1d.axvspan(P_TEST_LOW,  P_TEST_HIGH,  alpha=0.06, color="tomato",
                      label="Held-out regime")

        # Sample scatter on the 1D plot
        tr_scores = predicate.predict(make_features(train_df.iloc[tr_idx]))
        ho_scores = predicate.predict(make_features(held_out_df.iloc[ho_idx]))
        ax_1d.scatter(train_df.iloc[tr_idx]["P"], tr_scores,
                      s=7, alpha=0.35, color="steelblue", zorder=4)
        ax_1d.scatter(held_out_df.iloc[ho_idx]["P"], ho_scores,
                      s=7, alpha=0.45, color="tomato",    zorder=4)

        ax_1d.set_xscale("log")
        ax_1d.set_xlabel("Pressure  P  (atm)")
        ax_1d.set_ylabel("Validity score")
        ax_1d.set_ylim(-0.05, 1.05)
        ax_1d.set_title(_TITLES[assumption_id]
                        + f"\n1-D slice along ideal-gas isotherm  (T = {T_ref:.0f} K, n = {n_ref} mol)",
                        fontsize=9)
        ax_1d.legend(fontsize=7, loc="upper right", ncol=2)
        ax_1d.grid(True, alpha=0.3)

    fig.suptitle(
        "Validity Predicate Decision Boundaries\n"
        "Trained on low-pressure regime only  (P = 1–10 atm)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
