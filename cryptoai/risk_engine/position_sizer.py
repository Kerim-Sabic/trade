"""Position sizing with risk management."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np


class PositionSizer:
    """
    Position sizing with multiple methodologies.

    - Volatility targeting
    - Kelly criterion
    - Risk parity
    - Liquidity-aware sizing
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        max_position_pct: float = 0.2,
        kelly_fraction: float = 0.25,  # Fractional Kelly
        min_position_usd: float = 10.0,
    ):
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.min_position_usd = min_position_usd

    def volatility_target_size(
        self,
        capital: float,
        asset_volatility: float,
        leverage: float = 1.0,
    ) -> float:
        """
        Size position to target portfolio volatility.

        Args:
            capital: Available capital
            asset_volatility: Asset's volatility (annualized)
            leverage: Leverage to use

        Returns:
            Position size in USD
        """
        if asset_volatility <= 0:
            return 0.0

        # Target weight to achieve target portfolio vol
        weight = self.target_volatility / (asset_volatility * leverage)
        weight = min(weight, self.max_position_pct)

        position = capital * weight
        return max(0, position)

    def kelly_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Kelly criterion position sizing.

        Args:
            capital: Available capital
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Position size in USD
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        # Kelly formula
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b

        # Apply fractional Kelly
        kelly = kelly * self.kelly_fraction

        # Cap at max position
        kelly = min(max(0, kelly), self.max_position_pct)

        return capital * kelly

    def risk_parity_weights(
        self,
        volatilities: Dict[str, float],
        correlations: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights.

        Equal risk contribution from each asset.

        Args:
            volatilities: Dict of asset -> volatility
            correlations: Correlation matrix (optional)

        Returns:
            Dict of asset -> weight
        """
        assets = list(volatilities.keys())
        vols = np.array([volatilities[a] for a in assets])

        if correlations is None:
            # Inverse volatility weighting (simplified risk parity)
            inv_vols = 1.0 / (vols + 1e-8)
            weights = inv_vols / inv_vols.sum()
        else:
            # Full risk parity with correlations
            # This is a simplified version - full implementation would use optimization
            cov = np.outer(vols, vols) * correlations
            inv_diag = 1.0 / (np.diag(cov) + 1e-8)
            weights = inv_diag / inv_diag.sum()

        return {asset: w for asset, w in zip(assets, weights)}

    def liquidity_adjusted_size(
        self,
        base_size: float,
        available_liquidity: float,
        impact_coefficient: float = 0.0001,
        max_impact_pct: float = 0.001,  # 10 bps max impact
    ) -> float:
        """
        Adjust position size based on available liquidity.

        Args:
            base_size: Desired position size
            available_liquidity: Available liquidity in order book
            impact_coefficient: Price impact per unit
            max_impact_pct: Maximum acceptable price impact

        Returns:
            Adjusted position size
        """
        if available_liquidity <= 0:
            return 0.0

        # Estimate impact
        estimated_impact = base_size * impact_coefficient / available_liquidity

        if estimated_impact <= max_impact_pct:
            return base_size

        # Reduce size to meet impact constraint
        max_size = max_impact_pct * available_liquidity / impact_coefficient
        return min(base_size, max_size)

    def combined_size(
        self,
        capital: float,
        asset_volatility: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        available_liquidity: float,
        leverage: float = 1.0,
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Combined position sizing using multiple methods.

        Takes the minimum of all methods for safety.
        """
        # Volatility target
        vol_size = self.volatility_target_size(capital, asset_volatility, leverage)

        # Kelly
        kelly_size = self.kelly_size(capital, win_rate, avg_win, avg_loss)

        # Liquidity adjustment
        base_size = min(vol_size, kelly_size)
        liq_size = self.liquidity_adjusted_size(base_size, available_liquidity)

        # Final size (minimum of methods, above minimum threshold)
        final_size = max(self.min_position_usd, min(vol_size, kelly_size, liq_size))

        return final_size


class NeuralPositionSizer(nn.Module):
    """
    Neural network-based position sizer.

    Learns optimal sizing from market conditions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        max_position_pct: float = 0.2,
    ):
        super().__init__()

        self.max_position_pct = max_position_pct

        # Size predictor
        self.sizer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output 0-1 fraction
        )

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
        capital: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict optimal position size.

        Args:
            state: Market state
            capital: Available capital

        Returns:
            Dict with size and confidence
        """
        # Predict size fraction
        size_frac = self.sizer(state).squeeze(-1)

        # Scale to max position
        size_frac = size_frac * self.max_position_pct

        # Get confidence
        conf = self.confidence(state).squeeze(-1)

        # Adjust size by confidence
        adjusted_frac = size_frac * (0.5 + 0.5 * conf)

        # Calculate actual size
        position_size = capital * adjusted_frac

        return {
            "position_size": position_size,
            "size_fraction": adjusted_frac,
            "confidence": conf,
        }
