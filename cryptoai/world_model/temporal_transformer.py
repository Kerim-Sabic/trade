"""Temporal Transformer World Model for market dynamics prediction."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from cryptoai.encoders.base import (
    TransformerBlock,
    PositionalEncoding,
    GatedLinearUnit,
)


class StochasticLatentState(nn.Module):
    """
    Stochastic latent state with reparameterization.

    Models uncertainty in latent market state.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        min_std: float = 0.01,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.min_std = min_std

        # Encoder to latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )

    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization.

        Args:
            x: Input features
            num_samples: Number of latent samples
            deterministic: Use mean only

        Returns:
            Tuple of (samples, mean, std)
        """
        params = self.encoder(x)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std

        if deterministic:
            return mean, mean, std

        # Reparameterization trick
        if num_samples > 1:
            mean = repeat(mean, "b d -> b n d", n=num_samples)
            std = repeat(std, "b d -> b n d", n=num_samples)

        eps = torch.randn_like(std)
        samples = mean + std * eps

        return samples, mean, std

    def kl_divergence(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        prior_mean: Optional[torch.Tensor] = None,
        prior_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence from prior."""
        if prior_mean is None:
            prior_mean = torch.zeros_like(mean)
        if prior_std is None:
            prior_std = torch.ones_like(std)

        kl = (
            torch.log(prior_std / std)
            + (std ** 2 + (mean - prior_mean) ** 2) / (2 * prior_std ** 2)
            - 0.5
        )
        return kl.sum(dim=-1).mean()


class TemporalTransformerWorldModel(nn.Module):
    """
    Temporal Transformer World Model.

    Learns:
    - Market transition dynamics
    - Latent regime structure
    - Causal relationships between flows, price, and events
    """

    def __init__(
        self,
        state_dim: int = 256,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        sequence_length: int = 100,
        prediction_horizons: List[int] = [1, 5, 15, 60],
        num_latent_samples: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.num_latent_samples = num_latent_samples

        # State embedding
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, sequence_length * 2, dropout)

        # Stochastic latent encoder
        self.latent_encoder = StochasticLatentState(
            hidden_dim, latent_dim, hidden_dim
        )

        # Prior network (predicts next latent from previous)
        self.prior_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Latent to hidden projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict()
        for horizon in prediction_horizons:
            self.prediction_heads[f"h{horizon}"] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, len(prediction_horizons)),
            nn.Softplus(),
        )

        # Regime prediction head
        self.regime_head = nn.Linear(hidden_dim, 5)

        # Causal attention (learns what causes what)
        self.causal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - predict future states.

        Args:
            states: Historical state sequence (batch, seq_len, state_dim)
            actions: Optional action sequence (batch, seq_len, action_dim)
            return_latents: Return latent states
            deterministic: Use deterministic latent

        Returns:
            Dict with predictions for each horizon and uncertainty
        """
        B, T, _ = states.shape

        # Embed states
        x = self.state_embedding(states)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Encode to stochastic latent
        latent_samples, latent_mean, latent_std = self.latent_encoder(
            x[:, -1, :], self.num_latent_samples, deterministic
        )

        # Add latent to sequence
        latent_hidden = self.latent_proj(
            latent_mean if deterministic else latent_samples.mean(dim=1)
        )
        x[:, -1, :] = x[:, -1, :] + latent_hidden

        # Transformer processing
        for block in self.transformer_blocks:
            x = block(x)

        # Get final representation
        final_hidden = x[:, -1, :]

        # Multi-horizon predictions
        predictions = {}
        for horizon in self.prediction_horizons:
            pred = self.prediction_heads[f"h{horizon}"](final_hidden)
            predictions[f"state_h{horizon}"] = pred

        # Uncertainty for each horizon
        uncertainty = self.uncertainty_head(final_hidden)
        for i, horizon in enumerate(self.prediction_horizons):
            predictions[f"uncertainty_h{horizon}"] = uncertainty[:, i]

        # Regime prediction
        predictions["regime_logits"] = self.regime_head(final_hidden)

        # KL divergence for training
        prior_params = self.prior_network(
            latent_mean if T > 1 else torch.zeros_like(latent_mean)
        )
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.01

        predictions["kl_loss"] = self.latent_encoder.kl_divergence(
            latent_mean, latent_std, prior_mean, prior_std
        )

        if return_latents:
            predictions["latent_mean"] = latent_mean
            predictions["latent_std"] = latent_std
            predictions["latent_samples"] = latent_samples

        return predictions

    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        steps: int,
        action_sequence: Optional[torch.Tensor] = None,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future trajectory with uncertainty.

        Args:
            initial_state: Starting state (batch, state_dim)
            steps: Number of steps to predict
            action_sequence: Optional actions (batch, steps, action_dim)
            num_samples: Number of trajectory samples

        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        B = initial_state.shape[0]
        device = initial_state.device

        # Initialize trajectory
        trajectories = torch.zeros(num_samples, B, steps, self.state_dim, device=device)
        uncertainties = torch.zeros(B, steps, device=device)

        # Current state
        current_state = initial_state.unsqueeze(0).expand(num_samples, -1, -1)

        for t in range(steps):
            # Stack history (simplified - use rolling window)
            history = current_state.reshape(num_samples * B, 1, -1)

            # Predict next state
            preds = self.forward(history, deterministic=False)

            # Get prediction for minimum horizon
            min_horizon = min(self.prediction_horizons)
            pred_states = preds[f"state_h{min_horizon}"]
            pred_states = pred_states.reshape(num_samples, B, -1)

            # Store
            trajectories[:, :, t, :] = pred_states
            uncertainties[:, t] = preds[f"uncertainty_h{min_horizon}"].reshape(B)

            # Update current state
            current_state = pred_states

        return trajectories, uncertainties

    def imagine(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        horizon: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine outcome of taking an action.

        Used for planning and counterfactual reasoning.

        Args:
            state: Current state
            action: Proposed action
            horizon: Prediction horizon

        Returns:
            Imagined outcomes
        """
        B = state.shape[0]

        # Create state-action sequence
        state_seq = state.unsqueeze(1)  # (B, 1, state_dim)

        # Forward pass
        preds = self.forward(state_seq, deterministic=False, return_latents=True)

        # Get prediction for specified horizon
        closest_horizon = min(self.prediction_horizons, key=lambda x: abs(x - horizon))

        return {
            "predicted_state": preds[f"state_h{closest_horizon}"],
            "uncertainty": preds[f"uncertainty_h{closest_horizon}"],
            "latent": preds["latent_mean"],
            "regime_probs": F.softmax(preds["regime_logits"], dim=-1),
        }


class WorldModelEnsemble(nn.Module):
    """
    Ensemble of world models for robust prediction.

    Disagreement indicates epistemic uncertainty.
    """

    def __init__(
        self,
        num_models: int = 5,
        **model_kwargs,
    ):
        super().__init__()

        self.num_models = num_models

        self.models = nn.ModuleList([
            TemporalTransformerWorldModel(**model_kwargs)
            for _ in range(num_models)
        ])

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.

        Returns mean prediction and epistemic uncertainty.
        """
        all_predictions = [model(states, actions) for model in self.models]

        # Aggregate predictions
        result = {}

        # For each prediction key, compute mean and std across ensemble
        for key in all_predictions[0].keys():
            if key.startswith("state_"):
                preds = torch.stack([p[key] for p in all_predictions], dim=0)
                result[key] = preds.mean(dim=0)
                result[f"{key}_epistemic_std"] = preds.std(dim=0)

            elif key.startswith("uncertainty_"):
                # Aleatoric uncertainty - average
                preds = torch.stack([p[key] for p in all_predictions], dim=0)
                result[key] = preds.mean(dim=0)

            elif key == "regime_logits":
                preds = torch.stack([p[key] for p in all_predictions], dim=0)
                result[key] = preds.mean(dim=0)

            elif key == "kl_loss":
                result[key] = sum(p[key] for p in all_predictions) / self.num_models

        return result

    def get_disagreement(
        self,
        states: torch.Tensor,
        horizon: int = 1,
    ) -> torch.Tensor:
        """
        Get prediction disagreement (epistemic uncertainty).
        """
        all_predictions = [model(states) for model in self.models]

        closest_horizon = min(
            self.models[0].prediction_horizons,
            key=lambda x: abs(x - horizon)
        )

        preds = torch.stack(
            [p[f"state_h{closest_horizon}"] for p in all_predictions],
            dim=0
        )

        return preds.std(dim=0).mean(dim=-1)
