"""Liquidation cascade detection."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeDetector(nn.Module):
    """
    Detects potential liquidation cascades.

    Monitors:
    - Open interest stress
    - Liquidation concentration
    - Order book thinning
    - Derivatives funding stress
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        oi_change_threshold: float = -0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.oi_change_threshold = oi_change_threshold

        # Feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cascade probability head
        self.cascade_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Cascade severity head
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # Low, Medium, High
        )

        # Time-to-cascade head
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # OI stress indicator
        self.oi_stress = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        derivatives_state: torch.Tensor,
        orderbook_state: Optional[torch.Tensor] = None,
        liquidation_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            derivatives_state: Derivatives market features
            orderbook_state: Order book features (for liquidity analysis)
            liquidation_history: Recent liquidation data

        Returns:
            Dict with cascade probability and related metrics
        """
        # Process features
        features = self.feature_processor(derivatives_state)

        # Cascade probability
        cascade_prob = self.cascade_head(features).squeeze(-1)

        # Cascade severity
        severity_logits = self.severity_head(features)
        severity_probs = F.softmax(severity_logits, dim=-1)

        # Estimated time to cascade (in minutes)
        time_to_cascade = self.time_head(features).squeeze(-1) * 60  # Scale to minutes

        # OI stress
        oi_stress = self.oi_stress(derivatives_state).squeeze(-1)

        return {
            "cascade_probability": cascade_prob,
            "severity_probs": severity_probs,
            "time_to_cascade": time_to_cascade,
            "oi_stress": oi_stress,
        }


class LiquidationPressureModel(nn.Module):
    """
    Models liquidation pressure and its potential impact.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Pressure estimator
        self.pressure_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Price impact estimator
        self.impact_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Impact can be positive or negative
        )

        # Cascade trigger probability
        self.trigger_net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        liquidation_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate liquidation pressure.

        Args:
            liquidation_features: Features about recent/pending liquidations

        Returns:
            Dict with pressure and impact estimates
        """
        # Current pressure
        pressure = self.pressure_net(liquidation_features).squeeze(-1)

        # Expected price impact
        impact = self.impact_net(liquidation_features).squeeze(-1)

        # Cascade trigger probability
        trigger_input = torch.cat([
            liquidation_features,
            pressure.unsqueeze(-1),
            impact.unsqueeze(-1),
        ], dim=-1)
        trigger_prob = self.trigger_net(trigger_input).squeeze(-1)

        return {
            "pressure": pressure,
            "price_impact": impact,
            "trigger_prob": trigger_prob,
        }


class LiquidityStressMonitor(nn.Module):
    """
    Monitors order book liquidity stress that could enable cascades.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 64,
        depth_threshold: float = 0.3,
    ):
        super().__init__()

        self.depth_threshold = depth_threshold

        # Liquidity stress detector
        self.stress_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Stress level head
        self.stress_head = nn.Linear(hidden_dim // 2, 1)

        # Depth thinning detector
        self.thinning_head = nn.Linear(hidden_dim // 2, 1)

        # Spread widening detector
        self.spread_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        orderbook_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze liquidity stress.
        """
        features = self.stress_detector(orderbook_state)

        # Overall stress
        stress = torch.sigmoid(self.stress_head(features)).squeeze(-1)

        # Depth thinning
        thinning = torch.sigmoid(self.thinning_head(features)).squeeze(-1)

        # Spread widening
        spread_stress = torch.sigmoid(self.spread_head(features)).squeeze(-1)

        # Combined liquidity freeze probability
        freeze_prob = (stress + thinning + spread_stress) / 3

        return {
            "liquidity_stress": stress,
            "depth_thinning": thinning,
            "spread_stress": spread_stress,
            "freeze_probability": freeze_prob,
        }


class FundingStressMonitor(nn.Module):
    """
    Monitors funding rate stress that could indicate leverage unwinding.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 32,
        extreme_threshold: float = 0.001,  # 0.1% per 8h
    ):
        super().__init__()

        self.extreme_threshold = extreme_threshold

        # Stress detector
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Output heads
        self.stress_head = nn.Linear(hidden_dim // 2, 1)
        self.unwind_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        funding_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze funding stress.
        """
        features = self.detector(funding_features)

        # Funding stress
        stress = torch.sigmoid(self.stress_head(features)).squeeze(-1)

        # Unwind probability
        unwind_prob = torch.sigmoid(self.unwind_head(features)).squeeze(-1)

        return {
            "funding_stress": stress,
            "unwind_probability": unwind_prob,
        }
