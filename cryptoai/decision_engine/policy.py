"""Policy networks for trading decisions."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from cryptoai.encoders.base import TransformerBlock, GatedLinearUnit


class PolicyNetwork(nn.Module):
    """
    Base policy network for trading decisions.

    Level 2 - Decides:
    - Direction (-1 to 1)
    - Position size (0 to 1)
    - Leverage (1 to max)
    - Timing/urgency (0 to 1)
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        meta_cond_dim: int = 12,  # Conditioning from meta-controller
        dropout: float = 0.1,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # State + meta conditioning encoder
        input_dim = state_dim + meta_cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Policy backbone
        self.backbone = nn.ModuleList()
        for _ in range(num_layers):
            self.backbone.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Action mean head
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        # Action log std head
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize output layers
        self._init_output_layers()

    def _init_output_layers(self):
        """Initialize output layers with small weights."""
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(
        self,
        state: torch.Tensor,
        meta_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Current state (batch, state_dim)
            meta_conditioning: Conditioning from meta-controller (batch, cond_dim)

        Returns:
            Tuple of (action_mean, action_log_std)
        """
        # Concatenate with meta conditioning
        if meta_conditioning is not None:
            x = torch.cat([state, meta_conditioning], dim=-1)
        else:
            x = F.pad(state, (0, 12))  # Pad if no conditioning

        # Encode
        x = self.encoder(x)

        # Backbone with residuals
        for layer in self.backbone:
            x = x + layer(x)

        # Action distribution parameters
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(
        self,
        state: torch.Tensor,
        meta_conditioning: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: Current state
            meta_conditioning: Meta-controller conditioning
            deterministic: Use mean action

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state, meta_conditioning)

        if deterministic:
            action = torch.tanh(mean)
            return action, torch.zeros(action.shape[0], device=action.device)

        # Sample from Gaussian
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()  # Reparameterization trick

        # Squash to [-1, 1]
        action = torch.tanh(x_t)

        # Log probability with Jacobian correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        meta_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of action.

        Args:
            state: State
            action: Action taken
            meta_conditioning: Meta-controller conditioning

        Returns:
            Tuple of (log_prob, entropy)
        """
        mean, log_std = self.forward(state, meta_conditioning)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Inverse tanh
        action_unbounded = torch.atanh(action.clamp(-0.999, 0.999))

        log_prob = dist.log_prob(action_unbounded)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class SACPolicy(PolicyNetwork):
    """
    Soft Actor-Critic policy with automatic temperature tuning.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        meta_cond_dim: int = 12,
        init_temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            state_dim, action_dim, hidden_dim, num_layers, meta_cond_dim, **kwargs
        )

        # Temperature parameter (entropy coefficient)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.target_entropy = -action_dim  # Heuristic

    @property
    def alpha(self) -> torch.Tensor:
        """Get temperature."""
        return self.log_alpha.exp()

    def get_alpha_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Compute alpha (temperature) loss."""
        return (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()


class PPOPolicy(PolicyNetwork):
    """
    PPO policy with value head.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        meta_cond_dim: int = 12,
        **kwargs,
    ):
        super().__init__(
            state_dim, action_dim, hidden_dim, num_layers, meta_cond_dim, **kwargs
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Store old policy for PPO ratio
        self._old_mean = None
        self._old_log_std = None

    def forward(
        self,
        state: torch.Tensor,
        meta_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with value.

        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        # Concatenate with meta conditioning
        if meta_conditioning is not None:
            x = torch.cat([state, meta_conditioning], dim=-1)
        else:
            x = F.pad(state, (0, 12))

        x = self.encoder(x)

        for layer in self.backbone:
            x = x + layer(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        value = self.value_head(x).squeeze(-1)

        return mean, log_std, value

    def save_old_policy(self, state: torch.Tensor, meta_conditioning: Optional[torch.Tensor] = None):
        """Save current policy parameters for PPO ratio calculation."""
        with torch.no_grad():
            mean, log_std, _ = self.forward(state, meta_conditioning)
            self._old_mean = mean.clone()
            self._old_log_std = log_std.clone()

    def get_ppo_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        advantage: torch.Tensor,
        meta_conditioning: Optional[torch.Tensor] = None,
        clip_ratio: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss.

        Args:
            state: States
            action: Actions taken
            advantage: Advantage estimates
            meta_conditioning: Meta conditioning
            clip_ratio: PPO clip ratio

        Returns:
            Dict with policy loss, value loss, entropy
        """
        mean, log_std, value = self.forward(state, meta_conditioning)

        # Current policy distribution
        std = log_std.exp()
        dist = Normal(mean, std)

        action_unbounded = torch.atanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(action_unbounded)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # Old policy distribution
        if self._old_mean is not None:
            old_std = self._old_log_std.exp()
            old_dist = Normal(self._old_mean, old_std)
            old_log_prob = old_dist.log_prob(action_unbounded)
            old_log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            old_log_prob = old_log_prob.sum(dim=-1)
        else:
            old_log_prob = log_prob.detach()

        # PPO ratio
        ratio = (log_prob - old_log_prob).exp()

        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(
            ratio * advantage,
            clipped_ratio * advantage,
        ).mean()

        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()

        return {
            "policy_loss": policy_loss,
            "value": value,
            "entropy": entropy,
            "ratio": ratio.mean(),
        }


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation (used with SAC).
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_critics: int = 2,  # Twin critics for SAC
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_critics = num_critics

        # Create twin critics
        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            critic = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers - 1):
                critic.append(nn.Linear(hidden_dim, hidden_dim))
                critic.append(nn.LayerNorm(hidden_dim))
                critic.append(nn.GELU())
                critic.append(nn.Dropout(dropout))
            critic.append(nn.Linear(hidden_dim, 1))
            self.critics.append(critic)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through all critics.

        Returns:
            Tuple of Q-values from each critic
        """
        x = torch.cat([state, action], dim=-1)
        return tuple(critic(x).squeeze(-1) for critic in self.critics)

    def q_min(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value (pessimistic estimate)."""
        q_values = self.forward(state, action)
        return torch.min(torch.stack(q_values), dim=0)[0]
