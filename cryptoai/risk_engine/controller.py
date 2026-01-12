"""Main risk controller for position and exposure management."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np
from loguru import logger


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_pct: float = 0.2  # Max position as % of capital
    max_leverage: float = 5.0
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.05
    max_concentration: float = 0.5  # Max in single asset
    min_liquidity_ratio: float = 0.1  # Min liquidity vs position


@dataclass
class RiskState:
    """Current risk state."""

    current_drawdown: float
    daily_pnl: float
    daily_pnl_pct: float
    total_exposure: float
    max_single_exposure: float
    leverage_used: float
    risk_utilization: float  # 0-1, how much of risk budget used
    is_breached: bool
    breach_reason: Optional[str] = None


class RiskController:
    """
    Main risk controller.

    SURVIVAL FIRST - Capital preservation is the primary goal.
    """

    def __init__(
        self,
        limits: RiskLimits = None,
        target_volatility: float = 0.15,
        lookback_days: int = 20,
    ):
        self.limits = limits or RiskLimits()
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days

        # State tracking
        self._capital_history: List[float] = []
        self._daily_pnl: List[float] = []
        self._max_capital = 0.0
        self._day_start_capital = 0.0
        self._last_day = None
        self._positions: Dict[str, float] = {}
        self._volatility_history: List[float] = []

        # Risk state
        self._current_state = RiskState(
            current_drawdown=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            total_exposure=0.0,
            max_single_exposure=0.0,
            leverage_used=1.0,
            risk_utilization=0.0,
            is_breached=False,
        )

    def update(
        self,
        capital: float,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        timestamp: datetime,
    ) -> RiskState:
        """
        Update risk state with current positions.

        Args:
            capital: Current capital
            positions: Dict of asset -> position value
            current_prices: Dict of asset -> current price
            timestamp: Current time

        Returns:
            Updated risk state
        """
        # Check for new day
        if self._last_day != timestamp.date():
            self._day_start_capital = capital
            self._last_day = timestamp.date()
            self._daily_pnl = []

        # Update capital tracking
        self._capital_history.append(capital)
        if len(self._capital_history) > self.lookback_days * 24 * 60:  # Minute data
            self._capital_history = self._capital_history[-self.lookback_days * 24 * 60:]

        self._max_capital = max(self._max_capital, capital)
        self._positions = positions

        # Calculate metrics
        current_drawdown = (self._max_capital - capital) / self._max_capital if self._max_capital > 0 else 0
        daily_pnl = capital - self._day_start_capital
        daily_pnl_pct = daily_pnl / self._day_start_capital if self._day_start_capital > 0 else 0

        # Calculate exposure
        total_exposure = sum(abs(v) for v in positions.values())
        max_single = max(abs(v) for v in positions.values()) if positions else 0.0
        leverage = total_exposure / capital if capital > 0 else 0

        # Risk utilization
        dd_util = current_drawdown / self.limits.max_drawdown_pct
        loss_util = abs(min(0, daily_pnl_pct)) / self.limits.daily_loss_limit_pct
        lev_util = leverage / self.limits.max_leverage
        risk_utilization = max(dd_util, loss_util, lev_util)

        # Check breaches
        is_breached = False
        breach_reason = None

        if current_drawdown > self.limits.max_drawdown_pct:
            is_breached = True
            breach_reason = f"Max drawdown exceeded: {current_drawdown:.2%}"

        if daily_pnl_pct < -self.limits.daily_loss_limit_pct:
            is_breached = True
            breach_reason = f"Daily loss limit exceeded: {daily_pnl_pct:.2%}"

        if leverage > self.limits.max_leverage:
            is_breached = True
            breach_reason = f"Max leverage exceeded: {leverage:.2f}x"

        self._current_state = RiskState(
            current_drawdown=current_drawdown,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_exposure=total_exposure,
            max_single_exposure=max_single,
            leverage_used=leverage,
            risk_utilization=risk_utilization,
            is_breached=is_breached,
            breach_reason=breach_reason,
        )

        if is_breached:
            logger.warning(f"Risk breach: {breach_reason}")

        return self._current_state

    def get_volatility_scalar(self) -> float:
        """
        Get volatility-based position scalar.

        Higher volatility -> lower scalar.
        """
        if len(self._volatility_history) < 2:
            return 1.0

        current_vol = np.std(self._volatility_history[-self.lookback_days:])
        if current_vol == 0:
            return 1.0

        # Target vol / current vol (capped)
        scalar = self.target_volatility / current_vol
        return np.clip(scalar, 0.2, 2.0)

    def get_max_position_size(
        self,
        asset: str,
        capital: float,
        current_volatility: float,
    ) -> float:
        """
        Get maximum allowed position size for an asset.

        Args:
            asset: Asset symbol
            capital: Current capital
            current_volatility: Asset volatility

        Returns:
            Maximum position size in USD
        """
        # Base max from limits
        base_max = capital * self.limits.max_position_pct

        # Volatility adjustment
        vol_scalar = self.target_volatility / max(current_volatility, 0.01)
        vol_adjusted = base_max * np.clip(vol_scalar, 0.3, 1.5)

        # Risk utilization adjustment
        risk_headroom = 1 - self._current_state.risk_utilization
        risk_adjusted = vol_adjusted * max(0.1, risk_headroom)

        return risk_adjusted

    def get_max_leverage(
        self,
        current_volatility: float,
        black_swan_prob: float = 0.0,
    ) -> float:
        """
        Get maximum allowed leverage.

        Args:
            current_volatility: Current market volatility
            black_swan_prob: Probability of extreme event

        Returns:
            Maximum leverage
        """
        # Base max from limits
        base_max = self.limits.max_leverage

        # Volatility adjustment (higher vol -> lower leverage)
        vol_factor = self.target_volatility / max(current_volatility, 0.01)
        vol_adjusted = base_max * np.clip(vol_factor, 0.3, 1.0)

        # Black swan adjustment
        swan_factor = 1 - black_swan_prob
        swan_adjusted = vol_adjusted * max(0.2, swan_factor)

        # Risk utilization adjustment
        risk_headroom = 1 - self._current_state.risk_utilization
        final = swan_adjusted * max(0.2, risk_headroom)

        return max(1.0, final)

    def should_reduce_exposure(self) -> bool:
        """Check if exposure should be reduced."""
        return self._current_state.risk_utilization > 0.7

    def get_exposure_reduction(self) -> float:
        """Get recommended exposure reduction factor (0-1)."""
        if self._current_state.risk_utilization < 0.5:
            return 1.0
        elif self._current_state.risk_utilization < 0.7:
            return 0.8
        elif self._current_state.risk_utilization < 0.9:
            return 0.5
        else:
            return 0.2

    def can_open_position(
        self,
        asset: str,
        size: float,
        capital: float,
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Returns:
            Tuple of (can_open, reason)
        """
        if self._current_state.is_breached:
            return False, self._current_state.breach_reason

        # Check concentration
        current_total = sum(abs(v) for v in self._positions.values())
        new_total = current_total + abs(size)

        if new_total / capital > 1.0:  # Over 100% exposure
            return False, "Would exceed total exposure limit"

        # Check single asset limit
        current_in_asset = abs(self._positions.get(asset, 0))
        new_in_asset = current_in_asset + abs(size)

        if new_in_asset / capital > self.limits.max_concentration:
            return False, f"Would exceed concentration limit for {asset}"

        # Check leverage
        new_leverage = new_total / capital
        if new_leverage > self.limits.max_leverage:
            return False, f"Would exceed max leverage: {new_leverage:.2f}x"

        return True, "OK"

    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self._current_state

    def reset_daily(self):
        """Reset daily tracking."""
        self._daily_pnl = []
        self._day_start_capital = self._capital_history[-1] if self._capital_history else 0


class DynamicRiskController(RiskController):
    """
    Neural network-enhanced risk controller.

    Learns optimal risk parameters from market conditions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Neural risk adjustment
        self.risk_adjuster = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # position, leverage, drawdown, loss adjustments
            nn.Sigmoid(),
        )

    def get_dynamic_limits(
        self,
        market_state: torch.Tensor,
    ) -> RiskLimits:
        """
        Get dynamically adjusted limits based on market conditions.
        """
        with torch.no_grad():
            adjustments = self.risk_adjuster(market_state).squeeze()

        # Scale adjustments to reasonable ranges
        return RiskLimits(
            max_position_pct=self.limits.max_position_pct * (0.5 + adjustments[0].item()),
            max_leverage=self.limits.max_leverage * (0.3 + 0.7 * adjustments[1].item()),
            max_drawdown_pct=self.limits.max_drawdown_pct * (0.5 + 0.5 * adjustments[2].item()),
            daily_loss_limit_pct=self.limits.daily_loss_limit_pct * (0.5 + 0.5 * adjustments[3].item()),
        )
