"""Latent dynamics model for market state evolution."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class LatentDynamicsModel(nn.Module):
    """
    Models latent market dynamics.

    Learns:
    - State transition function
    - Latent regime identification
    - Uncertainty propagation
    """

    def __init__(
        self,
        state_dim: int = 256,
        latent_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_regimes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_regimes = num_regimes

        # State to latent encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_var
        )

        # Latent to state decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

        # Transition model (latent dynamics)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_var
        )

        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_regimes),
        )

        # Regime-specific transition adjustments
        self.regime_transitions = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim)
            for _ in range(num_regimes)
        ])

        # Uncertainty prediction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state to latent distribution."""
        params = self.encoder(state)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to state."""
        return self.decoder(latent)

    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample from latent distribution."""
        if deterministic:
            return mean
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def transition_step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single transition step in latent space.

        Returns next latent distribution.
        """
        # Base transition
        combined = torch.cat([latent, action], dim=-1)
        params = self.transition(combined)
        mean, log_var = params.chunk(2, dim=-1)

        # Apply regime-specific adjustment
        if regime is not None:
            regime_probs = F.softmax(regime, dim=-1)
            regime_adjustments = torch.stack([
                trans(latent) for trans in self.regime_transitions
            ], dim=1)  # (B, num_regimes, latent_dim)
            adjustment = torch.einsum("br,brd->bd", regime_probs, regime_adjustments)
            mean = mean + adjustment

        return mean, log_var

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Current state
            action: Action taken
            deterministic: Use mean latent

        Returns:
            Dict with next state prediction and latent info
        """
        # Encode current state
        z_mean, z_log_var = self.encode(state)
        z = self.reparameterize(z_mean, z_log_var, deterministic)

        # Classify regime
        regime_logits = self.regime_classifier(z)

        # Transition to next latent
        z_next_mean, z_next_log_var = self.transition_step(z, action, regime_logits)
        z_next = self.reparameterize(z_next_mean, z_next_log_var, deterministic)

        # Decode next state
        next_state = self.decode(z_next)

        # Uncertainty
        uncertainty = self.uncertainty_head(z_next)

        return {
            "next_state": next_state,
            "latent": z,
            "latent_next": z_next,
            "latent_mean": z_mean,
            "latent_log_var": z_log_var,
            "next_mean": z_next_mean,
            "next_log_var": z_next_log_var,
            "regime_logits": regime_logits,
            "uncertainty": uncertainty,
        }

    def rollout(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Rollout trajectory in latent space.

        Args:
            initial_state: Starting state (batch, state_dim)
            action_sequence: Actions (batch, horizon, action_dim)
            deterministic: Use deterministic transitions

        Returns:
            Dict with predicted trajectory
        """
        B, H, _ = action_sequence.shape
        device = initial_state.device

        # Storage
        states = [initial_state]
        latents = []
        uncertainties = []

        # Encode initial state
        z_mean, z_log_var = self.encode(initial_state)
        z = self.reparameterize(z_mean, z_log_var, deterministic)
        latents.append(z)

        # Rollout
        for t in range(H):
            action = action_sequence[:, t, :]
            regime_logits = self.regime_classifier(z)

            z_next_mean, z_next_log_var = self.transition_step(z, action, regime_logits)
            z = self.reparameterize(z_next_mean, z_next_log_var, deterministic)

            next_state = self.decode(z)
            uncertainty = self.uncertainty_head(z)

            states.append(next_state)
            latents.append(z)
            uncertainties.append(uncertainty)

        return {
            "states": torch.stack(states, dim=1),
            "latents": torch.stack(latents, dim=1),
            "uncertainties": torch.cat(uncertainties, dim=1),
        }

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state
            beta: KL weight (beta-VAE style)

        Returns:
            Dict with loss components
        """
        output = self.forward(state, action)

        # Reconstruction loss
        recon_loss = F.mse_loss(output["next_state"], next_state)

        # KL divergence (regularize latent)
        kl_loss = -0.5 * torch.mean(
            1 + output["latent_log_var"]
            - output["latent_mean"].pow(2)
            - output["latent_log_var"].exp()
        )

        # Transition KL
        trans_kl = -0.5 * torch.mean(
            1 + output["next_log_var"]
            - output["next_mean"].pow(2)
            - output["next_log_var"].exp()
        )

        total_loss = recon_loss + beta * (kl_loss + trans_kl)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "trans_kl_loss": trans_kl,
        }


class HierarchicalLatentModel(nn.Module):
    """
    Hierarchical latent model with multiple time scales.

    Captures both fast market microstructure and slow regime changes.
    """

    def __init__(
        self,
        state_dim: int = 256,
        fast_latent_dim: int = 32,
        slow_latent_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fast_latent_dim = fast_latent_dim
        self.slow_latent_dim = slow_latent_dim

        # Fast dynamics (market microstructure)
        self.fast_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fast_latent_dim * 2),
        )

        self.fast_transition = nn.GRUCell(
            fast_latent_dim + action_dim,
            fast_latent_dim,
        )

        # Slow dynamics (regime changes)
        self.slow_encoder = nn.Sequential(
            nn.Linear(state_dim + fast_latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, slow_latent_dim * 2),
        )

        self.slow_transition = nn.GRUCell(
            slow_latent_dim + fast_latent_dim,
            slow_latent_dim,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(fast_latent_dim + slow_latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Update rates (learned)
        self.slow_update_rate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical latents.
        """
        B, T, _ = states.shape

        # Initialize latents
        z_fast = torch.zeros(B, self.fast_latent_dim, device=states.device)
        z_slow = torch.zeros(B, self.slow_latent_dim, device=states.device)

        fast_latents = []
        slow_latents = []
        predictions = []

        for t in range(T):
            state = states[:, t, :]
            action = actions[:, t, :] if t < actions.shape[1] else torch.zeros(B, actions.shape[2], device=states.device)

            # Encode fast latent
            fast_params = self.fast_encoder(state)
            fast_mean, fast_log_var = fast_params.chunk(2, dim=-1)
            z_fast_new = fast_mean + torch.exp(0.5 * fast_log_var) * torch.randn_like(fast_mean)

            # Fast transition
            z_fast = self.fast_transition(
                torch.cat([z_fast_new, action], dim=-1),
                z_fast,
            )

            # Encode slow latent (less frequent updates)
            slow_input = torch.cat([state, z_fast], dim=-1)
            slow_params = self.slow_encoder(slow_input)
            slow_mean, slow_log_var = slow_params.chunk(2, dim=-1)
            z_slow_new = slow_mean + torch.exp(0.5 * slow_log_var) * torch.randn_like(slow_mean)

            # Slow transition (with learned update rate)
            update_rate = torch.sigmoid(self.slow_update_rate)
            z_slow_trans = self.slow_transition(
                torch.cat([z_slow_new, z_fast], dim=-1),
                z_slow,
            )
            z_slow = update_rate * z_slow_trans + (1 - update_rate) * z_slow

            # Decode
            combined = torch.cat([z_fast, z_slow], dim=-1)
            pred = self.decoder(combined)

            fast_latents.append(z_fast)
            slow_latents.append(z_slow)
            predictions.append(pred)

        return {
            "predictions": torch.stack(predictions, dim=1),
            "fast_latents": torch.stack(fast_latents, dim=1),
            "slow_latents": torch.stack(slow_latents, dim=1),
        }
