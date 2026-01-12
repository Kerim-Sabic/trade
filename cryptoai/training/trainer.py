"""Unified trainer for the complete trading AI system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from loguru import logger

from cryptoai.training.ddp import (
    DDPTrainer,
    DDPConfig,
    is_main_process,
    synchronize,
    reduce_tensor,
)
from cryptoai.training.experience_replay import (
    PrioritizedExperienceReplay,
    ReplayConfig,
    Experience,
)
from cryptoai.encoders import UnifiedStateEncoder
from cryptoai.world_model import WorldModel
from cryptoai.decision_engine import PolicyNetwork, MetaController
from cryptoai.black_swan import BlackSwanDetector
from cryptoai.risk_engine import RiskController


@dataclass
class TrainingConfig:
    """Configuration for unified training."""

    # Model dimensions
    state_dim: int = 200
    action_dim: int = 4
    hidden_dim: int = 256
    latent_dim: int = 128

    # Training phases
    encoder_pretrain_epochs: int = 10
    world_model_epochs: int = 50
    policy_epochs: int = 100

    # Learning rates
    encoder_lr: float = 1e-4
    world_model_lr: float = 3e-4
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Batch sizes
    batch_size: int = 64
    sequence_length: int = 100

    # RL settings
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Replay buffer
    replay_capacity: int = 1_000_000
    n_step: int = 3

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 1000

    # Evaluation
    eval_frequency: int = 10
    n_eval_episodes: int = 10


class UnifiedTrainer(DDPTrainer):
    """
    Unified trainer for the complete trading AI system.

    Training phases:
    1. Encoder pre-training (self-supervised)
    2. World model training (next-state prediction)
    3. Policy training (reinforcement learning)
    4. Black swan detector training (anomaly detection)
    5. Online adaptation (continual learning)
    """

    def __init__(
        self,
        config: TrainingConfig,
        ddp_config: DDPConfig,
        rank: int = 0,
    ):
        self.training_config = config

        # Build models
        self.encoder = UnifiedStateEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
        )

        self.world_model = WorldModel(
            state_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        self.policy = PolicyNetwork(
            state_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        self.meta_controller = MetaController(
            state_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
        )

        self.black_swan = BlackSwanDetector(
            state_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
        )

        self.risk_controller = RiskController()

        # Combine models for DDP wrapper
        self.models = nn.ModuleDict({
            "encoder": self.encoder,
            "world_model": self.world_model,
            "policy": self.policy,
            "meta_controller": self.meta_controller,
            "black_swan": self.black_swan,
        })

        # Initialize parent with combined models
        super().__init__(self.models, ddp_config, rank)

        # Setup optimizers
        self._setup_optimizers()

        # Experience replay
        replay_config = ReplayConfig(
            capacity=config.replay_capacity,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            sequence_length=config.sequence_length,
            n_step=config.n_step,
            gamma=config.gamma,
        )
        self.replay_buffer = PrioritizedExperienceReplay(replay_config)

        # Training state
        self.phase = "encoder"  # encoder, world_model, policy, online
        self.phase_epoch = 0

    def _setup_optimizers(self) -> None:
        """Setup optimizers for all models."""
        config = self.training_config

        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=config.encoder_lr,
            weight_decay=0.01,
        )

        self.world_model_optimizer = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=config.world_model_lr,
            weight_decay=0.01,
        )

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.policy_lr,
            weight_decay=0.01,
        )

        self.meta_optimizer = torch.optim.AdamW(
            self.meta_controller.parameters(),
            lr=config.policy_lr,
            weight_decay=0.01,
        )

        # Set main optimizer for parent class
        self.optimizer = self.policy_optimizer

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step dispatched by phase."""
        if self.phase == "encoder":
            return self._encoder_step(batch)
        elif self.phase == "world_model":
            return self._world_model_step(batch)
        elif self.phase == "policy":
            return self._policy_step(batch)
        else:
            return self._online_step(batch)

    def _encoder_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Self-supervised encoder pre-training step."""
        states = batch["state"]  # (B, T, F)

        # Masked autoencoding objective
        mask_ratio = 0.15
        mask = torch.rand_like(states[:, :, 0]) < mask_ratio
        masked_states = states.clone()
        masked_states[mask.unsqueeze(-1).expand_as(states)] = 0

        # Forward pass
        encoded = self.encoder(masked_states)

        # Reconstruction loss (predict masked positions)
        # Using simple MSE for demonstration
        reconstruction = self.encoder.decode(encoded) if hasattr(self.encoder, 'decode') else encoded

        # Contrastive loss
        batch_size = states.shape[0]
        z = encoded.mean(dim=1)  # Global representation

        # Simple contrastive: positive pairs are augmented versions
        z_pos = self.encoder(states + 0.01 * torch.randn_like(states)).mean(dim=1)

        # InfoNCE loss
        similarity = torch.mm(z, z_pos.t()) / 0.07
        labels = torch.arange(batch_size, device=states.device)
        contrastive_loss = nn.functional.cross_entropy(similarity, labels)

        # Total loss
        loss = contrastive_loss

        # Backward
        self.encoder_optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.encoder_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.encoder_optimizer.step()

        return {
            "loss": loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def _world_model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """World model training step."""
        states = batch["state"]
        actions = batch["action"]
        next_states = batch["next_state"]

        # Encode states
        with torch.no_grad():
            z = self.encoder(states)
            z_next_target = self.encoder(next_states)

        # Predict next state
        z_next_pred, z_mean, z_logvar = self.world_model(z, actions)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(z_next_pred, z_next_target)

        # KL divergence (VAE regularization)
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # Total loss
        loss = recon_loss + 0.01 * kl_loss

        # Backward
        self.world_model_optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.world_model_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.world_model_optimizer.step()

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def _policy_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PPO policy training step."""
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]

        # Encode states
        with torch.no_grad():
            z = self.encoder(states)

        # Get policy output
        action_dist, values = self.policy(z)

        # Calculate advantages (simplified GAE)
        with torch.no_grad():
            next_z = self.encoder(batch["next_state"])
            _, next_values = self.policy(next_z)
            td_target = rewards + self.training_config.gamma * next_values.squeeze() * (1 - dones)
            advantages = td_target - values.squeeze()

        # Policy loss (PPO clip)
        log_probs = action_dist.log_prob(actions).sum(-1)
        old_log_probs = log_probs.detach()

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.training_config.clip_ratio,
            1 + self.training_config.clip_ratio,
        )

        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(), td_target)

        # Entropy bonus
        entropy = action_dist.entropy().mean()

        # Total loss
        loss = (
            policy_loss
            + self.training_config.value_loss_coef * value_loss
            - self.training_config.entropy_coef * entropy
        )

        # Backward
        self.policy_optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.policy_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.policy_optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def _online_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Online adaptation step combining all components."""
        metrics = {}

        # World model update
        wm_metrics = self._world_model_step(batch)
        metrics.update({f"wm_{k}": v for k, v in wm_metrics.items()})

        # Policy update
        policy_metrics = self._policy_step(batch)
        metrics.update({f"policy_{k}": v for k, v in policy_metrics.items()})

        # Black swan detector update (less frequent)
        if self.global_step % 10 == 0:
            bs_metrics = self._black_swan_step(batch)
            metrics.update({f"bs_{k}": v for k, v in bs_metrics.items()})

        return metrics

    def _black_swan_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Black swan detector training step."""
        states = batch["state"]

        # Encode
        with torch.no_grad():
            z = self.encoder(states)

        # Get anomaly scores
        anomaly_scores, recon = self.black_swan(z)

        # VAE reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, z)

        # We don't have labels, so use reconstruction as proxy
        loss = recon_loss

        return {
            "loss": loss.item(),
            "mean_anomaly": anomaly_scores.mean().item(),
        }

    def train_phase(
        self,
        phase: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
    ) -> Dict[str, list]:
        """Train a specific phase."""
        self.phase = phase
        self.phase_epoch = 0

        # Select optimizer
        if phase == "encoder":
            self.optimizer = self.encoder_optimizer
        elif phase == "world_model":
            self.optimizer = self.world_model_optimizer
        else:
            self.optimizer = self.policy_optimizer

        if is_main_process():
            logger.info(f"Starting {phase} training for {num_epochs} epochs")

        return self.fit(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=5,
        )

    def full_training(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Full training pipeline through all phases."""
        config = self.training_config
        history = {}

        # Phase 1: Encoder pre-training
        if is_main_process():
            logger.info("Phase 1: Encoder pre-training")
        history["encoder"] = self.train_phase(
            "encoder",
            train_loader,
            val_loader,
            config.encoder_pretrain_epochs,
        )

        # Phase 2: World model training
        if is_main_process():
            logger.info("Phase 2: World model training")
        history["world_model"] = self.train_phase(
            "world_model",
            train_loader,
            val_loader,
            config.world_model_epochs,
        )

        # Phase 3: Policy training
        if is_main_process():
            logger.info("Phase 3: Policy training")
        history["policy"] = self.train_phase(
            "policy",
            train_loader,
            val_loader,
            config.policy_epochs,
        )

        # Save final checkpoint
        if is_main_process():
            final_path = Path(config.checkpoint_dir) / "final_model.pt"
            self.save_checkpoint(str(final_path))

        return history

    def collect_experience(
        self,
        env,
        n_steps: int = 1000,
    ) -> None:
        """Collect experience from environment."""
        self.models.eval()

        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(n_steps):
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                z = self.encoder(state_tensor)

                # Get meta-controller decision
                regime, confidence = self.meta_controller(z)

                # Get action from policy
                action_dist, _ = self.policy(z)
                action = action_dist.sample().cpu().numpy()[0]

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )
            self.replay_buffer.add(experience)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                if is_main_process():
                    logger.info(
                        f"Episode finished: reward={episode_reward:.2f}, length={episode_length}"
                    )
                state = env.reset()
                episode_reward = 0
                episode_length = 0

    def online_training(
        self,
        env,
        n_iterations: int = 10000,
        collect_steps: int = 100,
        train_steps: int = 50,
        batch_size: int = 64,
    ) -> Dict[str, list]:
        """Online training loop (interact and learn)."""
        self.phase = "online"
        history = {"rewards": [], "losses": []}

        for iteration in range(n_iterations):
            # Collect experience
            self.collect_experience(env, collect_steps)

            # Train on replay buffer
            if len(self.replay_buffer) >= batch_size:
                for _ in range(train_steps):
                    batch, indices, weights = self.replay_buffer.sample(batch_size)

                    # Move to device
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Training step
                    with autocast(enabled=self.use_amp):
                        metrics = self._online_step(batch)

                    history["losses"].append(metrics.get("loss", 0))

                    # Update priorities based on TD error
                    td_errors = np.abs(batch["reward"].cpu().numpy())
                    self.replay_buffer.update_priorities(indices, td_errors)

            # Logging
            if iteration % 100 == 0 and is_main_process():
                avg_loss = np.mean(history["losses"][-100:]) if history["losses"] else 0
                logger.info(
                    f"Iteration {iteration}/{n_iterations}: "
                    f"buffer_size={len(self.replay_buffer)}, avg_loss={avg_loss:.4f}"
                )

            # Checkpointing
            if iteration % self.training_config.save_frequency == 0:
                checkpoint_path = Path(self.training_config.checkpoint_dir) / f"online_{iteration}.pt"
                self.save_checkpoint(str(checkpoint_path))

            synchronize()

        return history

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
    ) -> Dict[str, float]:
        """Evaluate policy on environment."""
        self.models.eval()

        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    z = self.encoder(state_tensor)
                    action_dist, _ = self.policy(z)
                    action = action_dist.mean.cpu().numpy()[0]  # Use mean for evaluation

                state, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "max_reward": np.max(episode_rewards),
            "min_reward": np.min(episode_rewards),
        }
