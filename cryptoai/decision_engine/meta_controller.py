"""Meta-Controller for hierarchical decision making."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cryptoai.encoders.base import TransformerBlock, GatedLinearUnit


class MetaController(nn.Module):
    """
    Level 1 Meta-Controller.

    Determines:
    - Current market regime
    - Aggressiveness level (0-1)
    - Strategy weights
    - Risk budget allocation
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
        num_regimes: int = 5,
        num_strategies: int = 4,
        update_frequency: int = 60,  # Minutes between updates
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        self.num_strategies = num_strategies
        self.update_frequency = update_frequency

        # Regime names
        self.regime_names = [
            "trending_up",
            "trending_down",
            "ranging",
            "volatile",
            "crisis",
        ]

        # Strategy names
        self.strategy_names = [
            "momentum",
            "mean_reversion",
            "breakout",
            "defensive",
        ]

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal context (for regime stability)
        self.temporal_encoder = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout
        )

        # Regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_regimes),
        )

        # Aggressiveness head
        self.aggressiveness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Strategy allocation head
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim + num_regimes, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_strategies),
        )

        # Risk budget head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + num_regimes, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Value head (for PPO training)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Regime transition matrix (learned prior)
        self.transition_prior = nn.Parameter(
            torch.eye(num_regimes) * 0.8 + torch.ones(num_regimes, num_regimes) * 0.04
        )

    def forward(
        self,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        prev_regime: Optional[torch.Tensor] = None,
        return_value: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Current state (batch, state_dim)
            state_history: Historical states (batch, seq_len, state_dim)
            prev_regime: Previous regime (batch,)
            return_value: Return value estimate

        Returns:
            Dict with regime, aggressiveness, strategy weights, risk budget
        """
        B = state.shape[0]

        # Encode current state
        encoded = self.state_encoder(state)

        # Add temporal context if history available
        if state_history is not None:
            hist_encoded = self.state_encoder(state_history)
            _, h_n = self.temporal_encoder(hist_encoded)
            temporal_context = h_n[-1]
            encoded = encoded + temporal_context

        # Regime classification
        regime_logits = self.regime_head(encoded)

        # Apply transition prior if previous regime known
        if prev_regime is not None:
            transition_probs = F.softmax(self.transition_prior[prev_regime], dim=-1)
            regime_logits = regime_logits + torch.log(transition_probs + 1e-8)

        regime_probs = F.softmax(regime_logits, dim=-1)
        regime = regime_probs.argmax(dim=-1)

        # Aggressiveness
        aggressiveness = self.aggressiveness_head(encoded)

        # Strategy allocation (conditioned on regime)
        strategy_input = torch.cat([encoded, regime_probs], dim=-1)
        strategy_logits = self.strategy_head(strategy_input)
        strategy_weights = F.softmax(strategy_logits, dim=-1)

        # Risk budget (conditioned on regime)
        risk_budget = self.risk_head(strategy_input)

        output = {
            "regime": regime,
            "regime_probs": regime_probs,
            "regime_logits": regime_logits,
            "aggressiveness": aggressiveness.squeeze(-1),
            "strategy_weights": strategy_weights,
            "strategy_logits": strategy_logits,
            "risk_budget": risk_budget.squeeze(-1),
        }

        if return_value:
            output["value"] = self.value_head(encoded).squeeze(-1)

        return output

    def get_action(
        self,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        prev_regime: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get meta-action for policy conditioning.

        Returns:
            Tuple of (action_dict, log_prob)
        """
        output = self.forward(state, state_history, prev_regime, return_value=True)

        if deterministic:
            regime = output["regime"]
            strategy = output["strategy_weights"].argmax(dim=-1)
        else:
            # Sample from distributions
            regime_dist = torch.distributions.Categorical(output["regime_probs"])
            regime = regime_dist.sample()

            strategy_dist = torch.distributions.Categorical(output["strategy_weights"])
            strategy = strategy_dist.sample()

        # Calculate log probability
        log_prob = (
            torch.distributions.Categorical(output["regime_probs"]).log_prob(regime)
            + torch.distributions.Categorical(output["strategy_weights"]).log_prob(strategy)
        )

        return output, log_prob

    def get_conditioning_vector(
        self,
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get conditioning vector for policy network.

        Args:
            output: Meta-controller output

        Returns:
            Conditioning vector (batch, cond_dim)
        """
        return torch.cat([
            output["regime_probs"],
            output["aggressiveness"].unsqueeze(-1),
            output["strategy_weights"],
            output["risk_budget"].unsqueeze(-1),
        ], dim=-1)


class AdaptiveMetaController(MetaController):
    """
    Adaptive meta-controller that adjusts based on performance.

    Implements self-regulation:
    - Reduce exposure when alpha decays
    - Increase defense during uncertainty
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
        performance_dim: int = 10,
        **kwargs,
    ):
        super().__init__(state_dim, hidden_dim, **kwargs)

        self.performance_dim = performance_dim

        # Performance encoder
        self.performance_encoder = nn.Sequential(
            nn.Linear(performance_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Alpha decay detector
        self.alpha_detector = nn.Sequential(
            nn.Linear(performance_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(state_dim + performance_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Adaptation gate
        self.adapt_gate = nn.Sequential(
            nn.Linear(hidden_dim // 4 + 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 3),  # Scale for aggr, risk, exposure
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
        performance_metrics: Optional[torch.Tensor] = None,
        state_history: Optional[torch.Tensor] = None,
        prev_regime: Optional[torch.Tensor] = None,
        return_value: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptation.

        Args:
            state: Current state
            performance_metrics: Recent performance metrics (batch, perf_dim)
            state_history: Historical states
            prev_regime: Previous regime

        Returns:
            Adapted meta-control outputs
        """
        # Get base outputs
        output = super().forward(state, state_history, prev_regime, return_value)

        if performance_metrics is not None:
            # Encode performance
            perf_encoded = self.performance_encoder(performance_metrics)

            # Detect alpha decay
            alpha_decay = self.alpha_detector(performance_metrics)

            # Estimate uncertainty
            combined = torch.cat([state, performance_metrics], dim=-1)
            uncertainty = self.uncertainty_estimator(combined)

            # Compute adaptation scaling
            adapt_input = torch.cat([perf_encoded, alpha_decay, uncertainty], dim=-1)
            adapt_scale = self.adapt_gate(adapt_input)

            # Apply adaptation
            # When alpha decays or uncertainty high -> reduce aggressiveness and exposure
            output["aggressiveness"] = output["aggressiveness"] * adapt_scale[:, 0]
            output["risk_budget"] = output["risk_budget"] * adapt_scale[:, 1]

            # Shift strategy weights toward defensive when needed
            defense_boost = 1 - adapt_scale[:, 2]  # Higher when alpha decays
            output["strategy_weights"][:, -1] = (
                output["strategy_weights"][:, -1] + defense_boost * 0.3
            )
            output["strategy_weights"] = F.normalize(
                output["strategy_weights"], p=1, dim=-1
            )

            # Add adaptation info
            output["alpha_decay_prob"] = alpha_decay.squeeze(-1)
            output["uncertainty"] = uncertainty.squeeze(-1)
            output["adapt_scale"] = adapt_scale

        return output
