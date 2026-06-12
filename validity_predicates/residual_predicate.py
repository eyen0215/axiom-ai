"""
ResidualPredicate: skip-connection MLP mapping raw observables to log(theory residual).

Architecture is identical to ValidityPredicate (skip + MLP), with two differences:
  1. All 4 input features (P, V, T, n) are log-transformed (indices 0-3 vs 0-2).
  2. Output is raw regression value (no sigmoid) — training target is
     log(|PV/nRT - 1|), which has no natural zero-crossing at a known location.

Why no sigmoid: the old predicate applied sigmoid because the target was constructed
to be zero exactly at the known validity boundary. Here, the "boundary" (how large a
residual counts as broken) is calibrated empirically post-training, not baked in.

Fits into the system: trained by validity_predicates/train_residual.py;
evaluated by validity_predicates/evaluate_residual.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple

RESIDUAL_FEATURE_COLS = ["P", "V", "T", "n"]
RESIDUAL_N_FEATURES = 4
RESIDUAL_LOG_COLS = (0, 1, 2, 3)  # all 4 features log-transformed


class ResidualPredicate(nn.Module):
    """Skip-connection MLP predicting log(ideal-gas residual) from raw observables.

    Parameters
    ----------
    hidden_dims        : MLP hidden layer sizes (default: (32, 16))
    n_features         : number of input features (default: 4)
    log_transform_cols : feature indices to log-transform in forward()
    feature_cols       : feature column names (informational)
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 16),
        n_features: int = RESIDUAL_N_FEATURES,
        log_transform_cols: Sequence[int] = RESIDUAL_LOG_COLS,
        feature_cols: Sequence[str] = RESIDUAL_FEATURE_COLS,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self._log_transform_cols = tuple(log_transform_cols)
        self.feature_cols = list(feature_cols)

        # Linear skip — guarantees correct extrapolation direction
        self.skip = nn.Linear(n_features, 1)

        # MLP — residual nonlinear correction, heavily regularised during training
        mlp_dims = [n_features, *hidden_dims, 1]
        mlp_layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store feature normalisation statistics computed from training data."""
        self.feat_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw log-residual prediction (unbounded real, no sigmoid)."""
        x = x.clone().float()
        for col in self._log_transform_cols:
            x[..., col] = torch.log(x[..., col].clamp(min=1e-9))
        x_norm = (x - self.feat_mean) / (self.feat_std + 1e-8)
        return (self.skip(x_norm) + self.mlp(x_norm)).squeeze(-1)

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        """Return raw log-residual predictions as numpy array (no sigmoid)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32))
            return self.forward(x).numpy()

    def __call__(self, features):
        if isinstance(features, np.ndarray):
            return self.predict_raw(features)
        return super().__call__(features)
