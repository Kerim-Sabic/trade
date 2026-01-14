"""Self-supervised pre-training for encoders."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

# Handle PyTorch 2.0+ autocast API
try:
    from torch.amp import autocast as _autocast
    def autocast(enabled=True):
        return _autocast(device_type="cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast

from cryptoai.training.ddp import DDPTrainer, DDPConfig, is_main_process
from cryptoai.encoders import (
    UnifiedStateEncoder,
    OrderFlowEncoder,
    LiquidityEncoder,
    DerivativesEncoder,
    OnChainEncoder,
    EventEncoder,
)


@dataclass
class EncoderConfig:
    """Configuration for encoder pre-training."""

    # Architecture
    state_dim: int = 200
    hidden_dim: int = 256
    output_dim: int = 128

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    num_epochs: int = 50

    # Contrastive learning
    temperature: float = 0.07
    projection_dim: int = 128

    # Masking
    mask_ratio: float = 0.15
    mask_strategy: str = "random"  # random, block, feature

    # Augmentation
    noise_scale: float = 0.01
    dropout_rate: float = 0.1


class EncoderPretrainer(DDPTrainer):
    """
    Self-supervised pre-training for encoders.

    Training objectives:
    1. Masked autoencoding (MAE)
    2. Contrastive learning (SimCLR-style)
    3. Temporal prediction
    4. Feature reconstruction

    This pre-trains the encoders before RL training to learn
    good representations of market data.
    """

    def __init__(
        self,
        config: EncoderConfig,
        ddp_config: DDPConfig,
        rank: int = 0,
    ):
        self.encoder_config = config

        # Build encoder
        self.encoder = UnifiedStateEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

        # Temporal prediction head
        self.temporal_head = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Combine models
        self.models = nn.ModuleDict({
            "encoder": self.encoder,
            "projection": self.projection_head,
            "decoder": self.decoder,
            "temporal": self.temporal_head,
        })

        # Initialize parent
        super().__init__(self.models, ddp_config, rank)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.models.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Combined pre-training step."""
        states = batch["state"]  # (B, T, F)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            # 1. Masked autoencoding loss
            mae_loss, mae_metrics = self._masked_autoencoding(states)

            # 2. Contrastive loss
            contrastive_loss, cont_metrics = self._contrastive_learning(states)

            # 3. Temporal prediction loss
            temporal_loss, temp_metrics = self._temporal_prediction(states)

            # Total loss
            total_loss = mae_loss + 0.5 * contrastive_loss + 0.3 * temporal_loss

        # Backward
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

        return {
            "loss": total_loss.item(),
            "mae_loss": mae_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "temporal_loss": temporal_loss.item(),
            **mae_metrics,
            **cont_metrics,
            **temp_metrics,
        }

    def _masked_autoencoding(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Masked autoencoding objective.

        Randomly mask portions of input and reconstruct.
        """
        config = self.encoder_config
        batch_size, seq_len, feature_dim = states.shape

        # Create mask
        mask = self._create_mask(states.shape)

        # Apply mask
        masked_states = states.clone()
        masked_states[mask] = 0

        # Encode masked input
        z = self.encoder(masked_states)

        # Decode
        z_flat = z.view(-1, z.shape[-1])
        reconstructed_flat = self.decoder(z_flat)
        reconstructed = reconstructed_flat.view(batch_size, seq_len, feature_dim)

        # Loss only on masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(states)
        reconstruction_loss = nn.functional.mse_loss(
            reconstructed[mask_expanded],
            states[mask_expanded],
        )

        # Metrics
        with torch.no_grad():
            full_recon_error = nn.functional.mse_loss(reconstructed, states)

        return reconstruction_loss, {
            "recon_error": full_recon_error.item(),
            "mask_ratio": mask.float().mean().item(),
        }

    def _create_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create mask based on strategy."""
        config = self.encoder_config
        batch_size, seq_len, _ = shape
        device = next(self.models.parameters()).device

        if config.mask_strategy == "random":
            # Random masking of timesteps
            mask = torch.rand(batch_size, seq_len, device=device) < config.mask_ratio

        elif config.mask_strategy == "block":
            # Block masking (mask contiguous regions)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            block_size = int(seq_len * config.mask_ratio)

            for b in range(batch_size):
                start = torch.randint(0, seq_len - block_size, (1,)).item()
                mask[b, start:start + block_size] = True

        elif config.mask_strategy == "feature":
            # Mask entire features (not implemented here, would need different shape)
            mask = torch.rand(batch_size, seq_len, device=device) < config.mask_ratio

        else:
            mask = torch.rand(batch_size, seq_len, device=device) < config.mask_ratio

        return mask

    def _contrastive_learning(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Contrastive learning objective (SimCLR-style).

        Creates positive pairs through augmentation.
        """
        config = self.encoder_config
        batch_size = states.shape[0]

        # Create two augmented views
        view1 = self._augment(states)
        view2 = self._augment(states)

        # Encode both views
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)

        # Global pooling
        z1_pooled = z1.mean(dim=1)  # (B, D)
        z2_pooled = z2.mean(dim=1)

        # Project to contrastive space
        proj1 = self.projection_head(z1_pooled)
        proj2 = self.projection_head(z2_pooled)

        # Normalize
        proj1 = nn.functional.normalize(proj1, dim=-1)
        proj2 = nn.functional.normalize(proj2, dim=-1)

        # Compute similarity matrix
        similarity = torch.mm(proj1, proj2.t()) / config.temperature

        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=states.device)

        # Cross-entropy loss (both directions)
        loss_12 = nn.functional.cross_entropy(similarity, labels)
        loss_21 = nn.functional.cross_entropy(similarity.t(), labels)
        contrastive_loss = (loss_12 + loss_21) / 2

        # Metrics
        with torch.no_grad():
            accuracy = (similarity.argmax(dim=1) == labels).float().mean()

        return contrastive_loss, {
            "contrastive_acc": accuracy.item(),
        }

    def _augment(self, states: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to states."""
        config = self.encoder_config

        augmented = states.clone()

        # Add Gaussian noise
        noise = torch.randn_like(augmented) * config.noise_scale
        augmented = augmented + noise

        # Random dropout
        dropout_mask = torch.rand_like(augmented) > config.dropout_rate
        augmented = augmented * dropout_mask

        # Random scaling (per sample)
        scale = 0.9 + 0.2 * torch.rand(augmented.shape[0], 1, 1, device=augmented.device)
        augmented = augmented * scale

        return augmented

    def _temporal_prediction(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Temporal prediction objective.

        Predict future latent states from past.
        """
        batch_size, seq_len, feature_dim = states.shape

        # Split into context and target
        context_len = seq_len // 2
        context = states[:, :context_len]
        target = states[:, context_len:]

        # Encode context
        z_context = self.encoder(context)
        z_context_last = z_context[:, -1]  # Last context representation

        # Predict future
        z_pred = self.temporal_head(z_context_last)

        # Encode target (stop gradient)
        with torch.no_grad():
            z_target = self.encoder(target)
            z_target_first = z_target[:, 0]  # First target representation

        # Prediction loss
        temporal_loss = nn.functional.mse_loss(z_pred, z_target_first)

        # Metrics
        with torch.no_grad():
            cosine_sim = nn.functional.cosine_similarity(z_pred, z_target_first).mean()

        return temporal_loss, {
            "temporal_cosine_sim": cosine_sim.item(),
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        metrics = super().train_epoch(dataloader, epoch)

        # Step scheduler
        self.scheduler.step()

        if is_main_process():
            lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}: lr={lr:.6f}")

        return metrics

    def get_representations(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Get learned representations for states."""
        self.models.eval()
        with torch.no_grad():
            z = self.encoder(states)
        return z

    def evaluate_representations(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate quality of learned representations."""
        self.models.eval()

        all_embeddings = []
        all_labels = []  # If available

        with torch.no_grad():
            for batch in dataloader:
                states = batch["state"].to(self.device)
                z = self.encoder(states)
                z_pooled = z.mean(dim=1)
                all_embeddings.append(z_pooled.cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()

        # Compute embedding statistics
        mean_norm = np.linalg.norm(embeddings, axis=1).mean()
        embedding_std = embeddings.std(axis=0).mean()

        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings[:1000])  # Limit for efficiency

        return {
            "mean_norm": float(mean_norm),
            "embedding_std": float(embedding_std),
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
        }


class MultiModalEncoderPretrainer(EncoderPretrainer):
    """
    Pre-trainer for multi-modal encoders.

    Separately pre-trains each modality encoder then
    learns cross-modal alignment.
    """

    def __init__(
        self,
        config: EncoderConfig,
        ddp_config: DDPConfig,
        rank: int = 0,
    ):
        # Individual encoders
        self.order_flow_encoder = OrderFlowEncoder(
            input_dim=50,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim // 4,
        )

        self.liquidity_encoder = LiquidityEncoder(
            input_dim=30,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim // 4,
        )

        self.derivatives_encoder = DerivativesEncoder(
            input_dim=40,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim // 4,
        )

        self.onchain_encoder = OnChainEncoder(
            input_dim=30,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim // 4,
        )

        # Cross-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Override parent's encoder with fusion of all
        self.encoder = nn.ModuleDict({
            "order_flow": self.order_flow_encoder,
            "liquidity": self.liquidity_encoder,
            "derivatives": self.derivatives_encoder,
            "onchain": self.onchain_encoder,
            "fusion": self.fusion_layer,
        })

        # Initialize from grandparent (skip parent's encoder init)
        DDPTrainer.__init__(self, self.encoder, ddp_config, rank)

        self.encoder_config = config

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-modal encoders."""
        # Encode each modality
        z_order_flow = self.order_flow_encoder(batch.get("order_flow", torch.zeros(1)))
        z_liquidity = self.liquidity_encoder(batch.get("liquidity", torch.zeros(1)))
        z_derivatives = self.derivatives_encoder(batch.get("derivatives", torch.zeros(1)))
        z_onchain = self.onchain_encoder(batch.get("onchain", torch.zeros(1)))

        # Concatenate and fuse
        z_concat = torch.cat([z_order_flow, z_liquidity, z_derivatives, z_onchain], dim=-1)
        z_fused = self.fusion_layer(z_concat)

        return z_fused

    def _cross_modal_alignment(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Cross-modal alignment objective.

        Encourages different modalities to align in latent space.
        """
        # Encode each modality
        z_of = self.order_flow_encoder(batch["order_flow"])
        z_liq = self.liquidity_encoder(batch["liquidity"])
        z_der = self.derivatives_encoder(batch["derivatives"])
        z_oc = self.onchain_encoder(batch["onchain"])

        # Pool to single vector
        z_of = z_of.mean(dim=1)
        z_liq = z_liq.mean(dim=1)
        z_der = z_der.mean(dim=1)
        z_oc = z_oc.mean(dim=1)

        # Compute pairwise alignment losses
        alignments = []

        pairs = [(z_of, z_liq), (z_of, z_der), (z_of, z_oc),
                 (z_liq, z_der), (z_liq, z_oc), (z_der, z_oc)]

        for z1, z2 in pairs:
            # Cosine similarity encourages alignment
            sim = nn.functional.cosine_similarity(z1, z2)
            alignments.append(-sim.mean())  # Negative because we minimize

        alignment_loss = sum(alignments) / len(alignments)

        return alignment_loss, {
            "cross_modal_alignment": -alignment_loss.item(),
        }
