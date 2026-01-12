"""Tail risk estimation using Extreme Value Theory."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TailRiskEstimator(nn.Module):
    """
    Tail risk estimation using EVT (Extreme Value Theory).

    Estimates:
    - Value at Risk (VaR) at various confidence levels
    - Expected Shortfall (CVaR)
    - Tail distribution parameters
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        num_horizons: int = 2,
        threshold_quantile: float = 0.95,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_horizons = num_horizons
        self.threshold_quantile = threshold_quantile

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # EVT parameter estimator (Generalized Pareto Distribution)
        # Parameters: xi (shape), sigma (scale)
        self.gpd_params = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2 * num_horizons),  # xi, sigma for each horizon
        )

        # Direct VaR estimation
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # VaR 90%, 95%, 99%
            nn.Softplus(),  # VaR should be positive
        )

        # Expected Shortfall head
        self.es_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # Tail probability head
        self.tail_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_horizons),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
        returns_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Current market state
            returns_history: Optional historical returns for EVT fitting

        Returns:
            Dict with VaR, ES, and tail risk estimates
        """
        # Extract features
        features = self.feature_extractor(state)

        # Estimate GPD parameters
        gpd_raw = self.gpd_params(features)
        xi_sigma = gpd_raw.view(-1, self.num_horizons, 2)

        # Shape parameter (can be negative)
        xi = xi_sigma[:, :, 0]
        # Scale parameter (positive)
        sigma = F.softplus(xi_sigma[:, :, 1]) + 0.001

        # Direct VaR estimates
        var = self.var_head(features)
        var_90 = var[:, 0]
        var_95 = var[:, 1]
        var_99 = var[:, 2]

        # Expected Shortfall
        expected_shortfall = self.es_head(features).squeeze(-1)

        # Tail probability
        tail_probs = self.tail_prob_head(features)

        # If historical returns provided, compute empirical estimates
        if returns_history is not None:
            empirical_var, empirical_es = self._compute_empirical(returns_history)
        else:
            empirical_var = var_95
            empirical_es = expected_shortfall

        return {
            "var_90": var_90,
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall": expected_shortfall,
            "tail_prob": tail_probs,
            "gpd_xi": xi,
            "gpd_sigma": sigma,
            "empirical_var": empirical_var,
            "empirical_es": empirical_es,
        }

    def _compute_empirical(
        self,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute empirical VaR and ES from returns history.

        Args:
            returns: Historical returns (batch, time)

        Returns:
            Tuple of (VaR, ES)
        """
        B, T = returns.shape

        # Sort returns (ascending, so worst losses first)
        sorted_returns, _ = torch.sort(returns, dim=-1)

        # VaR at 95%
        var_idx = int(T * 0.05)
        var_95 = -sorted_returns[:, var_idx]  # Negate because we want loss

        # Expected Shortfall (average of worst 5%)
        worst_returns = sorted_returns[:, :var_idx]
        es = -worst_returns.mean(dim=-1)

        return var_95, es

    def gpd_quantile(
        self,
        xi: torch.Tensor,
        sigma: torch.Tensor,
        p: float,
        threshold: float,
    ) -> torch.Tensor:
        """
        Compute quantile of Generalized Pareto Distribution.

        GPD: F(x) = 1 - (1 + xi*x/sigma)^(-1/xi) for xi != 0
        """
        # Quantile function
        if torch.abs(xi).mean() < 1e-6:
            # Exponential case (xi -> 0)
            return threshold + sigma * (-torch.log(torch.tensor(1 - p)))
        else:
            return threshold + sigma / xi * (torch.pow(torch.tensor(1 - p), -xi) - 1)


class VolatilityRegimeDetector(nn.Module):
    """
    Detects volatility regime for tail risk adjustment.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 64,
        num_regimes: int = 3,  # Low, Medium, High volatility
    ):
        super().__init__()

        self.num_regimes = num_regimes

        self.classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regimes),
        )

        # Regime-specific volatility multipliers
        self.vol_multipliers = nn.Parameter(
            torch.tensor([0.5, 1.0, 2.0])
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify volatility regime.
        """
        logits = self.classifier(state)
        probs = F.softmax(logits, dim=-1)
        regime = probs.argmax(dim=-1)

        # Expected volatility multiplier
        vol_mult = torch.einsum("br,r->b", probs, self.vol_multipliers)

        return {
            "regime": regime,
            "regime_probs": probs,
            "vol_multiplier": vol_mult,
        }


class AdaptiveVaREstimator(nn.Module):
    """
    Adaptive VaR estimator that adjusts based on market conditions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        base_var: float = 0.02,  # 2% base VaR
    ):
        super().__init__()

        self.base_var = base_var

        # VaR scaling network
        self.var_scaler = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Confidence adjustment
        self.confidence_adj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive VaR.
        """
        # Scale factor (how much to multiply base VaR)
        scale = self.var_scaler(state).squeeze(-1)

        # Confidence in estimate
        confidence = self.confidence_adj(state).squeeze(-1)

        # Adaptive VaR
        var = self.base_var * (1 + scale)

        return {
            "var": var,
            "scale": scale,
            "confidence": confidence,
        }
