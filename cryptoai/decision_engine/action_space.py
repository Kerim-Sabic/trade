"""Action space definitions for the trading agent."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


@dataclass
class ContinuousAction:
    """
    Continuous action representation for trading.

    All actions are continuous and bounded.
    """

    direction: float  # -1 (short) to 1 (long), 0 = neutral
    position_size: float  # 0 to 1 (fraction of max position)
    leverage: float  # 1 to max_leverage
    urgency: float  # 0 to 1 (how quickly to execute)

    def to_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Convert to tensor."""
        tensor = torch.tensor([
            self.direction,
            self.position_size,
            self.leverage,
            self.urgency,
        ], dtype=torch.float32)
        if device:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ContinuousAction":
        """Create from tensor."""
        values = tensor.cpu().numpy()
        return cls(
            direction=float(values[0]),
            position_size=float(values[1]),
            leverage=float(values[2]),
            urgency=float(values[3]),
        )

    def to_order(
        self,
        current_price: float,
        capital: float,
        max_leverage: float = 5.0,
        min_order_size: float = 10.0,
    ) -> Dict:
        """
        Convert action to executable order parameters.

        Args:
            current_price: Current market price
            capital: Available capital
            max_leverage: Maximum allowed leverage
            min_order_size: Minimum order size in USD

        Returns:
            Order parameters dict
        """
        # Determine side
        if abs(self.direction) < 0.1:
            return {"action": "hold"}

        side = "buy" if self.direction > 0 else "sell"

        # Calculate position size
        effective_leverage = 1.0 + (max_leverage - 1.0) * self.leverage
        position_value = capital * self.position_size * effective_leverage
        quantity = position_value / current_price

        if position_value < min_order_size:
            return {"action": "hold"}

        return {
            "action": "trade",
            "side": side,
            "quantity": abs(quantity),
            "leverage": effective_leverage,
            "urgency": self.urgency,
            "order_type": "market" if self.urgency > 0.7 else "limit",
        }


class ActionSpace:
    """
    Defines the action space for the trading agent.

    Actions are continuous and bounded.
    """

    def __init__(
        self,
        max_leverage: float = 5.0,
        action_dim: int = 4,
    ):
        self.max_leverage = max_leverage
        self.action_dim = action_dim

        # Action bounds
        self.low = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Action names
        self.action_names = ["direction", "position_size", "leverage", "urgency"]

    def sample(self) -> ContinuousAction:
        """Sample random action."""
        values = np.random.uniform(self.low, self.high)
        return ContinuousAction(
            direction=values[0],
            position_size=values[1],
            leverage=values[2],
            urgency=values[3],
        )

    def clip(self, action: torch.Tensor) -> torch.Tensor:
        """Clip action to valid bounds."""
        low = torch.tensor(self.low, device=action.device)
        high = torch.tensor(self.high, device=action.device)
        return torch.clamp(action, low, high)

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to actual bounds."""
        low = torch.tensor(self.low, device=action.device)
        high = torch.tensor(self.high, device=action.device)
        return low + (action + 1.0) * 0.5 * (high - low)

    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Unscale action from actual bounds to [-1, 1]."""
        low = torch.tensor(self.low, device=action.device)
        high = torch.tensor(self.high, device=action.device)
        return 2.0 * (action - low) / (high - low) - 1.0

    @property
    def shape(self) -> Tuple[int]:
        """Action shape."""
        return (self.action_dim,)


class ActionModifier(nn.Module):
    """
    Modifies actions based on risk constraints and market conditions.

    Applied after policy output to ensure safe actions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        action_dim: int = 4,
    ):
        super().__init__()

        # Risk-based action scaling
        self.risk_scaler = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # Output 0-1 scaling factors
        )

        # Volatility-based leverage cap
        self.leverage_cap = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Max leverage fraction
        )

    def forward(
        self,
        action: torch.Tensor,
        state: torch.Tensor,
        risk_override: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Modify action based on state.

        Args:
            action: Raw policy action (batch, action_dim)
            state: Current state (batch, state_dim)
            risk_override: Optional risk scaling override

        Returns:
            Modified action
        """
        # Get risk-based scaling
        risk_scale = self.risk_scaler(state)

        if risk_override is not None:
            risk_scale = risk_scale * risk_override

        # Apply scaling
        action = action * risk_scale

        # Apply leverage cap
        leverage_cap = self.leverage_cap(state)
        action[:, 2] = action[:, 2] * leverage_cap.squeeze(-1)

        return action


class MetaAction:
    """
    Meta-level action from the meta-controller.

    Controls overall trading behavior.
    """

    def __init__(
        self,
        regime: int,  # Detected regime (0-4)
        aggressiveness: float,  # 0-1
        strategy_weights: Dict[str, float],  # Strategy allocation
        risk_budget: float,  # 0-1
    ):
        self.regime = regime
        self.aggressiveness = aggressiveness
        self.strategy_weights = strategy_weights
        self.risk_budget = risk_budget

    def get_policy_conditioning(self) -> torch.Tensor:
        """Get conditioning vector for policy network."""
        # One-hot regime + continuous values
        regime_onehot = torch.zeros(5)
        regime_onehot[self.regime] = 1.0

        weights = list(self.strategy_weights.values())

        return torch.cat([
            regime_onehot,
            torch.tensor([self.aggressiveness, self.risk_budget]),
            torch.tensor(weights),
        ])
