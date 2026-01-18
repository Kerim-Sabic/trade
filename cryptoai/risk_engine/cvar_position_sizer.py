"""
CVaR-Aware Position Sizing with Dynamic Stops.

Implements institutional-grade risk management:
- CVaR (Conditional Value at Risk) integration
- Dynamic drawdown stops
- Uncertainty-aware sizing
- Explicit "do nothing" conditions

Windows 11 Compatible.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class ActionDecision(Enum):
    """Trading action decision."""
    TRADE = "trade"
    HOLD = "hold"  # Do nothing
    REDUCE = "reduce"  # Reduce exposure
    FLATTEN = "flatten"  # Close all positions


@dataclass
class CVaRMetrics:
    """CVaR and tail risk metrics."""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    max_loss_estimate: float  # Estimated max loss
    tail_probability: float  # Probability of tail event


@dataclass
class PositionDecision:
    """Position sizing decision with risk metrics."""
    action: ActionDecision
    position_size: float
    confidence: float
    cvar_metrics: Optional[CVaRMetrics]
    reason: str
    recommended_stop_loss: Optional[float] = None
    recommended_take_profit: Optional[float] = None


class CVaRCalculator:
    """
    Calculate CVaR (Conditional Value at Risk).

    CVaR = E[Loss | Loss > VaR]

    Better than VaR for tail risk as it considers expected loss
    beyond the VaR threshold.
    """

    def __init__(
        self,
        confidence_levels: List[float] = None,
        window_size: int = 100,
    ):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.window_size = window_size
        self._returns_buffer: List[float] = []

    def update(self, return_value: float) -> None:
        """Add new return observation."""
        self._returns_buffer.append(return_value)
        if len(self._returns_buffer) > self.window_size:
            self._returns_buffer = self._returns_buffer[-self.window_size:]

    def calculate(self) -> CVaRMetrics:
        """Calculate current CVaR metrics."""
        if len(self._returns_buffer) < 10:
            return CVaRMetrics(
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                max_loss_estimate=0.0,
                tail_probability=0.0,
            )

        returns = np.array(self._returns_buffer)
        losses = -returns  # Convert to losses (positive values)

        # VaR calculation (percentile of losses)
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)

        # CVaR calculation (mean of losses beyond VaR)
        cvar_95 = losses[losses >= var_95].mean() if (losses >= var_95).any() else var_95
        cvar_99 = losses[losses >= var_99].mean() if (losses >= var_99).any() else var_99

        # Maximum observed loss
        max_loss = losses.max()

        # Tail probability (% of returns in the bottom 5%)
        tail_threshold = np.percentile(returns, 5)
        tail_prob = (returns <= tail_threshold).mean()

        return CVaRMetrics(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            max_loss_estimate=float(max_loss),
            tail_probability=float(tail_prob),
        )

    def reset(self) -> None:
        """Reset return buffer."""
        self._returns_buffer = []


class DynamicDrawdownManager:
    """
    Dynamic drawdown management with adaptive stops.

    Features:
    - Trailing maximum watermark
    - Adaptive stop levels based on volatility
    - Daily and cumulative drawdown limits
    - Partial position reduction at warning levels
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.15,  # 15% max drawdown
        warning_drawdown_pct: float = 0.08,  # 8% warning level
        daily_loss_limit_pct: float = 0.03,  # 3% daily limit
        volatility_multiplier: float = 2.0,  # Adjust stops by volatility
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.warning_drawdown_pct = warning_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.volatility_multiplier = volatility_multiplier

        # State tracking
        self._peak_equity = 0.0
        self._day_start_equity = 0.0
        self._current_day = None
        self._volatility_buffer: List[float] = []

    def update(
        self,
        current_equity: float,
        day_identifier: Optional[str] = None,
        return_value: Optional[float] = None,
    ) -> Tuple[float, ActionDecision, str]:
        """
        Update drawdown state and get recommended action.

        Returns:
            Tuple of (current_drawdown, action_decision, reason)
        """
        # Update peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Check for new day
        if day_identifier and day_identifier != self._current_day:
            self._day_start_equity = current_equity
            self._current_day = day_identifier

        # Update volatility buffer
        if return_value is not None:
            self._volatility_buffer.append(return_value)
            if len(self._volatility_buffer) > 50:
                self._volatility_buffer = self._volatility_buffer[-50:]

        # Calculate drawdowns
        cumulative_dd = 0.0
        if self._peak_equity > 0:
            cumulative_dd = (self._peak_equity - current_equity) / self._peak_equity

        daily_dd = 0.0
        if self._day_start_equity > 0:
            daily_dd = (self._day_start_equity - current_equity) / self._day_start_equity
            daily_dd = max(0, daily_dd)  # Only count losses

        # Calculate volatility-adjusted thresholds
        current_vol = np.std(self._volatility_buffer) if len(self._volatility_buffer) > 10 else 0.01
        vol_adjusted_max = self.max_drawdown_pct * (1 + self.volatility_multiplier * (0.02 - current_vol))
        vol_adjusted_max = max(0.05, min(0.25, vol_adjusted_max))  # Clamp between 5-25%

        # Determine action
        if cumulative_dd >= self.max_drawdown_pct or daily_dd >= self.daily_loss_limit_pct:
            return cumulative_dd, ActionDecision.FLATTEN, f"Max drawdown breached ({cumulative_dd:.2%} cumulative, {daily_dd:.2%} daily)"

        if cumulative_dd >= self.warning_drawdown_pct:
            return cumulative_dd, ActionDecision.REDUCE, f"Warning drawdown level ({cumulative_dd:.2%})"

        if cumulative_dd >= self.warning_drawdown_pct * 0.5:
            return cumulative_dd, ActionDecision.HOLD, f"Elevated drawdown ({cumulative_dd:.2%}) - cautious mode"

        return cumulative_dd, ActionDecision.TRADE, "Normal operation"

    def get_dynamic_stop_loss(
        self,
        entry_price: float,
        position_side: str,  # "long" or "short"
    ) -> float:
        """
        Calculate dynamic stop loss based on current volatility.

        Returns stop loss price.
        """
        current_vol = np.std(self._volatility_buffer) if len(self._volatility_buffer) > 10 else 0.02
        stop_distance = entry_price * current_vol * self.volatility_multiplier

        if position_side == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def reset_day(self, current_equity: float) -> None:
        """Reset daily tracking."""
        self._day_start_equity = current_equity

    def reset(self) -> None:
        """Full reset."""
        self._peak_equity = 0.0
        self._day_start_equity = 0.0
        self._current_day = None
        self._volatility_buffer = []


class InactionThreshold:
    """
    Determines when to explicitly "do nothing".

    Implements:
    - Confidence threshold for action
    - Market regime filtering
    - Uncertainty bounds
    - Edge case detection
    """

    def __init__(
        self,
        min_confidence: float = 0.6,  # Minimum confidence to trade
        max_uncertainty: float = 0.3,  # Maximum allowed uncertainty
        min_expected_return: float = 0.001,  # 0.1% minimum expected return
        regime_sensitivity: float = 0.5,  # How much regime affects threshold
    ):
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
        self.min_expected_return = min_expected_return
        self.regime_sensitivity = regime_sensitivity

    def should_act(
        self,
        confidence: float,
        uncertainty: float,
        expected_return: float,
        regime: str = "normal",
        black_swan_prob: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Determine if we should act or do nothing.

        Returns:
            Tuple of (should_act, reason)
        """
        # Black swan override - always reduce exposure
        if black_swan_prob > 0.5:
            return False, f"Black swan probability too high ({black_swan_prob:.2%})"

        # Confidence check
        regime_adjusted_min_conf = self.min_confidence
        if regime == "volatile":
            regime_adjusted_min_conf += self.regime_sensitivity * 0.1
        elif regime == "crisis":
            regime_adjusted_min_conf += self.regime_sensitivity * 0.2

        if confidence < regime_adjusted_min_conf:
            return False, f"Confidence too low ({confidence:.2%} < {regime_adjusted_min_conf:.2%})"

        # Uncertainty check
        if uncertainty > self.max_uncertainty:
            return False, f"Uncertainty too high ({uncertainty:.2%} > {self.max_uncertainty:.2%})"

        # Expected return check (must exceed transaction costs)
        if abs(expected_return) < self.min_expected_return:
            return False, f"Expected return too small ({expected_return:.4%})"

        # All checks passed
        return True, "All conditions met for trading"


class CVaRPositionSizer(nn.Module):
    """
    CVaR-aware neural position sizer.

    Combines:
    - Traditional sizing methods (volatility targeting, Kelly)
    - CVaR risk adjustment
    - Dynamic drawdown management
    - Inaction thresholds

    This is the main entry point for position sizing decisions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        max_position_pct: float = 0.15,
        target_cvar: float = 0.02,  # Target 2% CVaR
        min_confidence: float = 0.6,
    ):
        super().__init__()

        self.max_position_pct = max_position_pct
        self.target_cvar = target_cvar

        # Neural size predictor
        self.size_net = nn.Sequential(
            nn.Linear(state_dim + 6, hidden_dim),  # +6 for risk metrics
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # size and confidence
        )

        # Uncertainty estimator (aleatoric + epistemic)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive
        )

        # Risk components
        self.cvar_calculator = CVaRCalculator()
        self.drawdown_manager = DynamicDrawdownManager()
        self.inaction_threshold = InactionThreshold(min_confidence=min_confidence)

    def forward(
        self,
        state: torch.Tensor,
        capital: torch.Tensor,
        cvar_95: Optional[torch.Tensor] = None,
        var_99: Optional[torch.Tensor] = None,
        current_drawdown: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
        black_swan_prob: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for position sizing.

        Returns dict with:
        - position_size: Recommended position size
        - size_fraction: As fraction of capital
        - confidence: Confidence in recommendation
        - uncertainty: Estimated uncertainty
        - should_trade: Boolean mask for trading decision
        """
        batch_size = state.shape[0]
        device = state.device

        # Default risk metrics if not provided
        if cvar_95 is None:
            cvar_95 = torch.zeros(batch_size, device=device)
        if var_99 is None:
            var_99 = torch.zeros(batch_size, device=device)
        if current_drawdown is None:
            current_drawdown = torch.zeros(batch_size, device=device)
        if volatility is None:
            volatility = torch.full((batch_size,), 0.02, device=device)
        if black_swan_prob is None:
            black_swan_prob = torch.zeros(batch_size, device=device)
        if regime is None:
            regime = torch.zeros(batch_size, device=device)  # 0 = normal

        # Construct risk features
        risk_features = torch.stack([
            cvar_95,
            var_99,
            current_drawdown,
            volatility,
            black_swan_prob,
            regime,
        ], dim=-1)

        # Concatenate state with risk features
        combined = torch.cat([state, risk_features], dim=-1)

        # Predict size and confidence
        output = self.size_net(combined)
        raw_size = torch.sigmoid(output[:, 0])  # 0-1
        raw_confidence = torch.sigmoid(output[:, 1])  # 0-1

        # Estimate uncertainty
        uncertainty = self.uncertainty_net(state).squeeze(-1)

        # CVaR adjustment: reduce size when CVaR exceeds target
        cvar_ratio = cvar_95 / (self.target_cvar + 1e-8)
        cvar_scalar = torch.clamp(1.0 / (1.0 + cvar_ratio), 0.2, 1.0)

        # Volatility adjustment
        vol_scalar = torch.clamp(0.02 / (volatility + 1e-8), 0.3, 1.5)

        # Drawdown adjustment: reduce size as drawdown increases
        dd_scalar = torch.clamp(1.0 - current_drawdown * 2, 0.1, 1.0)

        # Black swan adjustment
        swan_scalar = torch.clamp(1.0 - black_swan_prob * 2, 0.0, 1.0)

        # Combined scalar
        risk_scalar = cvar_scalar * vol_scalar * dd_scalar * swan_scalar

        # Final size calculation
        base_size_frac = raw_size * self.max_position_pct
        adjusted_size_frac = base_size_frac * risk_scalar

        # Confidence adjustment
        adjusted_confidence = raw_confidence * (1.0 - uncertainty.clamp(0, 0.5) * 2)

        # Trading decision (should we trade at all?)
        min_conf_threshold = 0.5 + 0.2 * (regime / 2.0)  # Higher threshold for volatile regimes
        should_trade = (adjusted_confidence > min_conf_threshold) & (uncertainty < 0.4) & (black_swan_prob < 0.5)

        # Zero out positions where we shouldn't trade
        final_size_frac = adjusted_size_frac * should_trade.float()

        # Calculate actual position size
        position_size = capital * final_size_frac

        return {
            "position_size": position_size,
            "size_fraction": final_size_frac,
            "confidence": adjusted_confidence,
            "uncertainty": uncertainty,
            "should_trade": should_trade,
            "risk_scalar": risk_scalar,
            "cvar_scalar": cvar_scalar,
        }

    def get_position_decision(
        self,
        state: torch.Tensor,
        capital: float,
        current_return: Optional[float] = None,
        current_equity: Optional[float] = None,
        regime: str = "normal",
        black_swan_prob: float = 0.0,
    ) -> PositionDecision:
        """
        Get complete position decision with explanations.

        This is the main API for position sizing.
        """
        # Update risk tracking
        if current_return is not None:
            self.cvar_calculator.update(current_return)

        # Get CVaR metrics
        cvar_metrics = self.cvar_calculator.calculate()

        # Get drawdown decision
        dd = 0.0
        dd_action = ActionDecision.TRADE
        dd_reason = "Normal"

        if current_equity is not None:
            dd, dd_action, dd_reason = self.drawdown_manager.update(current_equity)

        # Handle drawdown-based decisions first
        if dd_action == ActionDecision.FLATTEN:
            return PositionDecision(
                action=ActionDecision.FLATTEN,
                position_size=0.0,
                confidence=1.0,  # High confidence we should flatten
                cvar_metrics=cvar_metrics,
                reason=dd_reason,
            )

        if dd_action == ActionDecision.REDUCE:
            return PositionDecision(
                action=ActionDecision.REDUCE,
                position_size=capital * 0.5 * self.max_position_pct,  # Half normal size
                confidence=0.7,
                cvar_metrics=cvar_metrics,
                reason=dd_reason,
            )

        # Neural network forward pass
        with torch.no_grad():
            regime_tensor = torch.tensor([0 if regime == "normal" else 1 if regime == "volatile" else 2])
            output = self.forward(
                state=state.unsqueeze(0) if state.dim() == 1 else state,
                capital=torch.tensor([capital]),
                cvar_95=torch.tensor([cvar_metrics.cvar_95]),
                var_99=torch.tensor([cvar_metrics.var_99]),
                current_drawdown=torch.tensor([dd]),
                black_swan_prob=torch.tensor([black_swan_prob]),
                regime=regime_tensor.float(),
            )

        should_trade = output["should_trade"][0].item()
        position_size = output["position_size"][0].item()
        confidence = output["confidence"][0].item()
        uncertainty = output["uncertainty"][0].item()

        # Check inaction threshold
        should_act, inaction_reason = self.inaction_threshold.should_act(
            confidence=confidence,
            uncertainty=uncertainty,
            expected_return=0.01,  # Would come from model prediction
            regime=regime,
            black_swan_prob=black_swan_prob,
        )

        if not should_trade or not should_act:
            return PositionDecision(
                action=ActionDecision.HOLD,
                position_size=0.0,
                confidence=confidence,
                cvar_metrics=cvar_metrics,
                reason=inaction_reason if not should_act else "Model uncertainty too high",
            )

        # Normal trading decision
        return PositionDecision(
            action=ActionDecision.TRADE,
            position_size=position_size,
            confidence=confidence,
            cvar_metrics=cvar_metrics,
            reason="All risk checks passed",
            recommended_stop_loss=None,  # Would calculate based on entry
        )


# Factory function
def create_cvar_position_sizer(
    state_dim: int = 256,
    max_position_pct: float = 0.15,
    target_cvar: float = 0.02,
    min_confidence: float = 0.6,
) -> CVaRPositionSizer:
    """Create configured CVaR position sizer."""
    return CVaRPositionSizer(
        state_dim=state_dim,
        max_position_pct=max_position_pct,
        target_cvar=target_cvar,
        min_confidence=min_confidence,
    )
