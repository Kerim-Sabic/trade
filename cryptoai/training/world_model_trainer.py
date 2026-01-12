"""Specialized trainer for the World Model."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from loguru import logger

from cryptoai.training.ddp import DDPTrainer, DDPConfig, is_main_process
from cryptoai.world_model import WorldModel, TemporalTransformer, LatentDynamics
from cryptoai.encoders import UnifiedStateEncoder


@dataclass
class WorldModelConfig:
    """Configuration for world model training."""

    # Architecture
    state_dim: int = 200
    latent_dim: int = 128
    action_dim: int = 4
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    sequence_length: int = 100

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    num_epochs: int = 100

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.01
    prediction_weight: float = 1.0
    contrastive_weight: float = 0.1

    # KL annealing
    kl_warmup_epochs: int = 10
    kl_max_weight: float = 0.1

    # Prediction horizons
    prediction_horizons: Tuple[int, ...] = (1, 5, 10, 20)


class WorldModelTrainer(DDPTrainer):
    """
    Trainer for the World Model.

    The world model learns:
    1. Latent state dynamics (next state prediction)
    2. Uncertainty quantification (epistemic and aleatoric)
    3. Causal structure between features
    4. Multiple prediction horizons

    Training objectives:
    - VAE reconstruction loss
    - KL divergence regularization
    - Multi-step prediction loss
    - Contrastive predictive coding
    """

    def __init__(
        self,
        config: WorldModelConfig,
        ddp_config: DDPConfig,
        encoder: Optional[UnifiedStateEncoder] = None,
        rank: int = 0,
    ):
        self.wm_config = config

        # Build models
        if encoder is None:
            self.encoder = UnifiedStateEncoder(
                state_dim=config.state_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.latent_dim,
            )
        else:
            self.encoder = encoder

        # Main world model
        self.world_model = WorldModel(
            state_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Temporal transformer for sequence modeling
        self.temporal_model = TemporalTransformer(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            sequence_length=config.sequence_length,
        )

        # Latent dynamics model
        self.latent_dynamics = LatentDynamics(
            state_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Combine models
        self.models = nn.ModuleDict({
            "encoder": self.encoder,
            "world_model": self.world_model,
            "temporal": self.temporal_model,
            "dynamics": self.latent_dynamics,
        })

        # Initialize parent
        super().__init__(self.models, ddp_config, rank)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.world_model.parameters()) +
            list(self.temporal_model.parameters()) +
            list(self.latent_dynamics.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate / 10,
        )

        # KL weight for annealing
        self._kl_weight = 0.0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for world model."""
        states = batch["state"]  # (B, T, F)
        actions = batch["action"]  # (B, A)
        next_states = batch["next_state"]  # (B, T, F)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            # Encode sequences
            z_seq = self._encode_sequence(states)  # (B, T, D)
            z_next_seq = self._encode_sequence(next_states)

            # Get last timestep representation
            z = z_seq[:, -1]  # (B, D)
            z_next_target = z_next_seq[:, -1]

            # World model forward (single step)
            z_next_pred, z_mean, z_logvar = self.world_model(z, actions)

            # Calculate losses
            losses = self._compute_losses(
                z_next_pred=z_next_pred,
                z_next_target=z_next_target,
                z_mean=z_mean,
                z_logvar=z_logvar,
                z_seq=z_seq,
                actions=actions,
            )

            total_loss = losses["total_loss"]

        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), 1.0)
            self.optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def _encode_sequence(self, states: torch.Tensor) -> torch.Tensor:
        """Encode state sequence to latent space."""
        batch_size, seq_len, feature_dim = states.shape

        # Encode each timestep
        states_flat = states.view(-1, feature_dim)
        z_flat = self.encoder(states_flat.unsqueeze(1))  # Add sequence dim
        z_seq = z_flat.view(batch_size, seq_len, -1)

        return z_seq

    def _compute_losses(
        self,
        z_next_pred: torch.Tensor,
        z_next_target: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        z_seq: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all training losses."""
        config = self.wm_config

        # 1. Reconstruction loss (next state prediction)
        recon_loss = nn.functional.mse_loss(z_next_pred, z_next_target)

        # 2. KL divergence (VAE regularization)
        kl_loss = -0.5 * torch.mean(
            1 + z_logvar - z_mean.pow(2) - z_logvar.exp()
        )

        # 3. Multi-step prediction loss
        multistep_loss = self._compute_multistep_loss(z_seq, actions)

        # 4. Contrastive predictive coding loss
        cpc_loss = self._compute_cpc_loss(z_seq)

        # 5. Uncertainty calibration loss
        uncertainty_loss = self._compute_uncertainty_loss(z_mean, z_logvar, z_next_target, z_next_pred)

        # Total loss with weights
        total_loss = (
            config.reconstruction_weight * recon_loss +
            self._kl_weight * kl_loss +
            config.prediction_weight * multistep_loss +
            config.contrastive_weight * cpc_loss +
            0.1 * uncertainty_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "multistep_loss": multistep_loss,
            "cpc_loss": cpc_loss,
            "uncertainty_loss": uncertainty_loss,
        }

    def _compute_multistep_loss(
        self,
        z_seq: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-step prediction loss."""
        batch_size, seq_len, latent_dim = z_seq.shape
        total_loss = 0.0
        n_predictions = 0

        # Use temporal model for multi-step prediction
        z_context = z_seq[:, :seq_len // 2]  # First half as context

        # Predict future states
        for horizon in self.wm_config.prediction_horizons:
            target_idx = seq_len // 2 + horizon - 1
            if target_idx >= seq_len:
                continue

            # Get temporal context
            temporal_out = self.temporal_model(z_context)
            z_pred = temporal_out[:, -1]  # Last output

            # Target
            z_target = z_seq[:, target_idx]

            # Loss
            horizon_loss = nn.functional.mse_loss(z_pred, z_target)
            total_loss += horizon_loss
            n_predictions += 1

        return total_loss / max(n_predictions, 1)

    def _compute_cpc_loss(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Contrastive predictive coding loss."""
        batch_size, seq_len, latent_dim = z_seq.shape

        # Split sequence
        context_len = seq_len // 2
        z_context = z_seq[:, :context_len]
        z_future = z_seq[:, context_len:]

        # Get context representation
        temporal_out = self.temporal_model(z_context)
        c = temporal_out[:, -1]  # Context vector

        # Predict k steps ahead
        k = min(4, z_future.shape[1])
        total_loss = 0.0

        for i in range(k):
            # Positive sample
            z_pos = z_future[:, i]

            # Negative samples (other sequences in batch)
            z_neg = z_pos.roll(shifts=1, dims=0)

            # Prediction (simple linear for CPC)
            pred = c  # Could add a prediction head

            # InfoNCE loss
            pos_score = torch.sum(pred * z_pos, dim=-1)
            neg_score = torch.sum(pred * z_neg, dim=-1)

            scores = torch.stack([pos_score, neg_score], dim=-1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=z_seq.device)

            loss = nn.functional.cross_entropy(scores, labels)
            total_loss += loss

        return total_loss / k

    def _compute_uncertainty_loss(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        z_target: torch.Tensor,
        z_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Uncertainty calibration loss."""
        # Prediction error
        error = (z_target - z_pred).pow(2).mean(dim=-1)

        # Predicted uncertainty
        var = z_logvar.exp().mean(dim=-1)

        # Calibration: high uncertainty should correlate with high error
        correlation = torch.corrcoef(torch.stack([error, var]))[0, 1]

        # Loss encourages positive correlation
        return -correlation if not torch.isnan(correlation) else torch.tensor(0.0, device=z_mean.device)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with KL annealing."""
        # KL weight annealing
        if epoch < self.wm_config.kl_warmup_epochs:
            self._kl_weight = (
                self.wm_config.kl_max_weight * epoch / self.wm_config.kl_warmup_epochs
            )
        else:
            self._kl_weight = self.wm_config.kl_max_weight

        # Call parent train_epoch
        metrics = super().train_epoch(dataloader, epoch)

        # Update scheduler
        self.scheduler.step()

        if is_main_process():
            logger.info(
                f"Epoch {epoch}: kl_weight={self._kl_weight:.4f}, "
                f"lr={self.scheduler.get_last_lr()[0]:.6f}"
            )

        return metrics

    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        n_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future trajectory given initial state and action sequence.

        Args:
            initial_state: Initial state (B, F)
            actions: Action sequence (B, n_steps, A)
            n_steps: Number of prediction steps

        Returns:
            predicted_states: Predicted state sequence (B, n_steps, D)
            uncertainties: Uncertainty estimates (B, n_steps)
        """
        self.models.eval()
        batch_size = initial_state.shape[0]

        predicted_states = []
        uncertainties = []

        # Encode initial state
        with torch.no_grad():
            z = self.encoder(initial_state.unsqueeze(1))[:, 0]

            for step in range(n_steps):
                action = actions[:, step]

                # Predict next state
                z_next, z_mean, z_logvar = self.world_model(z, action)

                # Uncertainty from variance
                uncertainty = z_logvar.exp().mean(dim=-1)

                predicted_states.append(z_next)
                uncertainties.append(uncertainty)

                z = z_next

        predicted_states = torch.stack(predicted_states, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)

        return predicted_states, uncertainties

    def imagine_rollout(
        self,
        initial_state: torch.Tensor,
        policy,
        n_steps: int = 20,
        n_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine rollout for model-based planning.

        Uses world model to simulate trajectories for planning.

        Args:
            initial_state: Initial state
            policy: Policy network for action selection
            n_steps: Rollout horizon
            n_samples: Number of parallel rollouts

        Returns:
            Dictionary with imagined states, actions, rewards
        """
        self.models.eval()
        batch_size = initial_state.shape[0]

        # Expand for multiple samples
        z = self.encoder(initial_state.unsqueeze(1))[:, 0]
        z = z.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, z.shape[-1])

        imagined_states = [z]
        imagined_actions = []
        imagined_rewards = []

        with torch.no_grad():
            for step in range(n_steps):
                # Get action from policy
                action_dist, value = policy(z)
                action = action_dist.sample()

                # Predict next state with stochastic sampling
                z_next, z_mean, z_logvar = self.world_model(z, action)

                # Add stochasticity
                std = z_logvar.mul(0.5).exp()
                eps = torch.randn_like(std)
                z_next_sampled = z_mean + std * eps

                # Estimate reward (simplified)
                reward = value.squeeze()

                imagined_states.append(z_next_sampled)
                imagined_actions.append(action)
                imagined_rewards.append(reward)

                z = z_next_sampled

        return {
            "states": torch.stack(imagined_states, dim=1),
            "actions": torch.stack(imagined_actions, dim=1),
            "rewards": torch.stack(imagined_rewards, dim=1),
        }
