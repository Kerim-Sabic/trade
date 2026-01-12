"""Reward calculation for the trading agent."""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np


@dataclass
class RewardComponents:
    """Individual reward components for analysis."""

    pnl: float
    fees: float
    slippage: float
    drawdown_penalty: float
    liquidation_penalty: float
    tail_risk_penalty: float
    instability_penalty: float
    sharpe_bonus: float
    total: float


class RewardCalculator:
    """
    Calculates trading rewards.

    R_t = ΔPnL - fees - slippage - liquidation_risk
          - drawdown_penalty - tail_risk_penalty - instability_penalty
    """

    def __init__(
        self,
        pnl_weight: float = 1.0,
        fee_penalty: float = 1.0,
        slippage_penalty: float = 1.5,
        drawdown_penalty: float = 2.0,
        liquidation_penalty: float = 10.0,
        tail_risk_penalty: float = 3.0,
        instability_penalty: float = 1.0,
        sharpe_bonus: float = 0.5,
        risk_free_rate: float = 0.0,
        target_volatility: float = 0.15,
    ):
        self.pnl_weight = pnl_weight
        self.fee_penalty = fee_penalty
        self.slippage_penalty = slippage_penalty
        self.drawdown_penalty = drawdown_penalty
        self.liquidation_penalty = liquidation_penalty
        self.tail_risk_penalty = tail_risk_penalty
        self.instability_penalty = instability_penalty
        self.sharpe_bonus = sharpe_bonus
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

        # Running statistics for normalization
        self._returns_buffer = []
        self._max_capital = 0.0

    def calculate(
        self,
        pnl: float,
        fees: float,
        slippage: float,
        capital: float,
        position_value: float,
        leverage: float,
        volatility: float,
        tail_risk_prob: float = 0.0,
        liquidation_prob: float = 0.0,
        action_change: float = 0.0,
    ) -> RewardComponents:
        """
        Calculate reward with all components.

        Args:
            pnl: Profit/loss for the period
            fees: Trading fees paid
            slippage: Slippage cost
            capital: Current capital
            position_value: Current position value
            leverage: Current leverage
            volatility: Market volatility
            tail_risk_prob: Probability of tail event (from black swan layer)
            liquidation_prob: Probability of liquidation
            action_change: L2 distance from previous action (for stability)

        Returns:
            RewardComponents with all components
        """
        # Update running statistics
        self._max_capital = max(self._max_capital, capital)
        ret = pnl / capital if capital > 0 else 0
        self._returns_buffer.append(ret)
        if len(self._returns_buffer) > 1000:
            self._returns_buffer = self._returns_buffer[-1000:]

        # PnL component (main reward)
        pnl_reward = self.pnl_weight * pnl / capital if capital > 0 else 0

        # Fee penalty
        fee_cost = self.fee_penalty * fees / capital if capital > 0 else 0

        # Slippage penalty (higher for larger orders)
        slip_cost = self.slippage_penalty * slippage / capital if capital > 0 else 0

        # Drawdown penalty
        current_dd = (self._max_capital - capital) / self._max_capital if self._max_capital > 0 else 0
        dd_penalty = self.drawdown_penalty * (current_dd ** 2)

        # Liquidation risk penalty
        liq_penalty = self.liquidation_penalty * liquidation_prob

        # Tail risk penalty (from black swan layer)
        tail_penalty = self.tail_risk_penalty * tail_risk_prob

        # Instability penalty (penalize erratic actions)
        instab_penalty = self.instability_penalty * action_change

        # Sharpe bonus (reward risk-adjusted returns)
        sharpe_bonus_val = 0.0
        if len(self._returns_buffer) > 20:
            returns = np.array(self._returns_buffer[-20:])
            if returns.std() > 0:
                sharpe = (returns.mean() - self.risk_free_rate / 252) / returns.std()
                sharpe_bonus_val = self.sharpe_bonus * max(0, sharpe) * 0.01

        # Total reward
        total = (
            pnl_reward
            - fee_cost
            - slip_cost
            - dd_penalty
            - liq_penalty
            - tail_penalty
            - instab_penalty
            + sharpe_bonus_val
        )

        return RewardComponents(
            pnl=pnl_reward,
            fees=-fee_cost,
            slippage=-slip_cost,
            drawdown_penalty=-dd_penalty,
            liquidation_penalty=-liq_penalty,
            tail_risk_penalty=-tail_penalty,
            instability_penalty=-instab_penalty,
            sharpe_bonus=sharpe_bonus_val,
            total=total,
        )

    def calculate_batch(
        self,
        pnl: torch.Tensor,
        fees: torch.Tensor,
        slippage: torch.Tensor,
        capital: torch.Tensor,
        position_value: torch.Tensor,
        leverage: torch.Tensor,
        volatility: torch.Tensor,
        tail_risk_prob: torch.Tensor,
        liquidation_prob: torch.Tensor,
        action_change: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch reward calculation for training.

        All inputs should be tensors of shape (batch,).
        """
        # PnL reward
        pnl_reward = self.pnl_weight * pnl / (capital + 1e-8)

        # Fee penalty
        fee_cost = self.fee_penalty * fees / (capital + 1e-8)

        # Slippage penalty
        slip_cost = self.slippage_penalty * slippage / (capital + 1e-8)

        # Liquidation penalty
        liq_penalty = self.liquidation_penalty * liquidation_prob

        # Tail risk penalty
        tail_penalty = self.tail_risk_penalty * tail_risk_prob

        # Instability penalty
        instab_penalty = self.instability_penalty * action_change

        # Total
        total = (
            pnl_reward
            - fee_cost
            - slip_cost
            - liq_penalty
            - tail_penalty
            - instab_penalty
        )

        return total

    def reset(self):
        """Reset running statistics."""
        self._returns_buffer = []
        self._max_capital = 0.0


class LearnedRewardModel(nn.Module):
    """
    Learned reward model for reward shaping.

    Learns to predict human-aligned rewards from market outcomes.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict reward for transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Predicted reward
        """
        x = torch.cat([state, action, next_state], dim=-1)
        return self.model(x).squeeze(-1)


class IntrinsicMotivation(nn.Module):
    """
    Intrinsic motivation for exploration.

    Rewards novel states to encourage exploration.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        intrinsic_scale: float = 0.1,
    ):
        super().__init__()

        self.intrinsic_scale = intrinsic_scale

        # State predictor (predict next state from current)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Target network (frozen random features)
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward based on prediction error.

        Novel states have high prediction error.
        """
        pred = self.predictor(state)
        target = self.target(state)

        # Prediction error as intrinsic reward
        error = F.mse_loss(pred, target, reduction="none").mean(dim=-1)

        return self.intrinsic_scale * error
