"""Specialized trainer for the Policy Network using PPO/SAC."""

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

from cryptoai.training.ddp import DDPTrainer, DDPConfig, is_main_process, reduce_tensor
from cryptoai.training.experience_replay import (
    PrioritizedExperienceReplay,
    ReplayConfig,
    Experience,
)
from cryptoai.decision_engine import PolicyNetwork, MetaController
from cryptoai.encoders import UnifiedStateEncoder


@dataclass
class PolicyConfig:
    """Configuration for policy training."""

    # Architecture
    state_dim: int = 128  # Latent state dimension
    action_dim: int = 4
    hidden_dim: int = 256

    # PPO settings
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # GAE settings
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    learning_rate: float = 3e-4
    critic_lr: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 10  # PPO epochs per update
    n_minibatches: int = 4

    # SAC settings (if using SAC)
    use_sac: bool = False
    tau: float = 0.005  # Target network update rate
    alpha: float = 0.2  # Temperature parameter
    auto_alpha: bool = True

    # Reward shaping
    reward_scale: float = 1.0
    reward_clip: float = 10.0


class PolicyTrainer(DDPTrainer):
    """
    Trainer for the Policy Network.

    Supports both PPO and SAC algorithms.

    PPO (Proximal Policy Optimization):
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function baseline

    SAC (Soft Actor-Critic):
    - Maximum entropy RL
    - Twin Q-networks
    - Automatic temperature tuning
    """

    def __init__(
        self,
        config: PolicyConfig,
        ddp_config: DDPConfig,
        encoder: Optional[UnifiedStateEncoder] = None,
        rank: int = 0,
    ):
        self.policy_config = config

        # Encoder (frozen during policy training)
        if encoder is None:
            self.encoder = UnifiedStateEncoder(
                state_dim=200,  # Raw state dim
                hidden_dim=config.hidden_dim,
                output_dim=config.state_dim,
            )
        else:
            self.encoder = encoder

        # Policy network
        self.policy = PolicyNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Meta controller
        self.meta_controller = MetaController(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
        )

        # SAC components
        if config.use_sac:
            self._setup_sac()

        # Combine models
        models_dict = {
            "encoder": self.encoder,
            "policy": self.policy,
            "meta_controller": self.meta_controller,
        }
        if config.use_sac:
            models_dict["q1"] = self.q1
            models_dict["q2"] = self.q2
            models_dict["q1_target"] = self.q1_target
            models_dict["q2_target"] = self.q2_target

        self.models = nn.ModuleDict(models_dict)

        # Initialize parent
        super().__init__(self.models, ddp_config, rank)

        # Setup optimizers
        self._setup_optimizers()

        # Rollout buffer for PPO
        self._rollout_buffer: List[Dict] = []

    def _setup_sac(self) -> None:
        """Setup SAC-specific components."""
        config = self.policy_config

        # Twin Q-networks
        self.q1 = self._build_q_network()
        self.q2 = self._build_q_network()

        # Target networks
        self.q1_target = self._build_q_network()
        self.q2_target = self._build_q_network()
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Temperature parameter
        if config.auto_alpha:
            self.log_alpha = nn.Parameter(torch.zeros(1))
            self.target_entropy = -config.action_dim
        else:
            self.log_alpha = None

    def _build_q_network(self) -> nn.Module:
        """Build Q-network for SAC."""
        config = self.policy_config
        return nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _setup_optimizers(self) -> None:
        """Setup optimizers."""
        config = self.policy_config

        # Policy optimizer
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.meta_controller.parameters()),
            lr=config.learning_rate,
        )

        self.optimizer = self.policy_optimizer

        if config.use_sac:
            # Q-network optimizer
            self.q_optimizer = torch.optim.Adam(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                lr=config.critic_lr,
            )

            # Alpha optimizer
            if config.auto_alpha:
                self.alpha_optimizer = torch.optim.Adam(
                    [self.log_alpha],
                    lr=config.learning_rate,
                )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step - dispatches to PPO or SAC."""
        if self.policy_config.use_sac:
            return self._sac_step(batch)
        else:
            return self._ppo_step(batch)

    def _ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PPO training step."""
        config = self.policy_config

        states = batch["state"]
        actions = batch["action"]
        old_log_probs = batch["log_prob"]
        advantages = batch["advantage"]
        returns = batch["return"]

        # Encode states (frozen encoder)
        with torch.no_grad():
            z = self.encoder(states)

        # Get current policy outputs
        action_dist, values = self.policy(z)
        log_probs = action_dist.log_prob(actions).sum(-1)
        entropy = action_dist.entropy().mean()

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio,
            1 - config.clip_ratio,
            1 + config.clip_ratio,
        )

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)

        # Total loss
        loss = (
            policy_loss
            + config.value_loss_coef * value_loss
            - config.entropy_coef * entropy
        )

        # Backward
        self.policy_optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.policy_optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
            self.scaler.step(self.policy_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
            self.policy_optimizer.step()

        # Metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clip_fraction = (torch.abs(ratio - 1) > config.clip_ratio).float().mean()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

    def _sac_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """SAC training step."""
        config = self.policy_config

        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]

        # Encode states
        with torch.no_grad():
            z = self.encoder(states)
            z_next = self.encoder(next_states)

        # Current alpha
        alpha = self.log_alpha.exp() if config.auto_alpha else config.alpha

        # --- Q-network update ---
        with torch.no_grad():
            # Sample next actions
            next_action_dist, _ = self.policy(z_next)
            next_actions = next_action_dist.rsample()
            next_log_probs = next_action_dist.log_prob(next_actions).sum(-1)

            # Target Q-values
            q1_target = self.q1_target(torch.cat([z_next, next_actions], dim=-1))
            q2_target = self.q2_target(torch.cat([z_next, next_actions], dim=-1))
            q_target = torch.min(q1_target, q2_target).squeeze(-1)

            # Soft target
            target = rewards + config.gamma * (1 - dones) * (q_target - alpha * next_log_probs)

        # Current Q-values
        q1 = self.q1(torch.cat([z, actions], dim=-1)).squeeze(-1)
        q2 = self.q2(torch.cat([z, actions], dim=-1)).squeeze(-1)

        q1_loss = nn.functional.mse_loss(q1, target)
        q2_loss = nn.functional.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Policy update ---
        action_dist, _ = self.policy(z)
        new_actions = action_dist.rsample()
        log_probs = action_dist.log_prob(new_actions).sum(-1)

        q1_new = self.q1(torch.cat([z, new_actions], dim=-1))
        q2_new = self.q2(torch.cat([z, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new).squeeze(-1)

        policy_loss = (alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --- Alpha update ---
        alpha_loss = torch.tensor(0.0)
        if config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # --- Target network update ---
        self._soft_update(self.q1, self.q1_target, config.tau)
        self._soft_update(self.q2, self.q2_target, config.tau)

        return {
            "loss": (q_loss + policy_loss).item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": alpha.item() if isinstance(alpha, torch.Tensor) else alpha,
            "alpha_loss": alpha_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module,
        tau: float,
    ) -> None:
        """Soft update of target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards (T,)
            values: Value estimates (T,)
            dones: Done flags (T,)
            next_value: Bootstrap value

        Returns:
            advantages: GAE advantages (T,)
            returns: Returns for value function (T,)
        """
        config = self.policy_config
        gamma = config.gamma
        gae_lambda = config.gae_lambda

        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def collect_rollout(
        self,
        env,
        n_steps: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollout for PPO training.

        Args:
            env: Trading environment
            n_steps: Number of steps to collect

        Returns:
            Rollout buffer as dictionary of tensors
        """
        self.models.eval()

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        state = env.reset()

        for step in range(n_steps):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                z = self.encoder(state_tensor)
                action_dist, value = self.policy(z)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(-1)

            action_np = action.cpu().numpy()[0]
            next_state, reward, done, info = env.step(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())

            state = next_state
            if done:
                state = env.reset()

        # Get bootstrap value
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            z = self.encoder(state_tensor)
            _, next_value = self.policy(z)

        # Convert to tensors
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        values = torch.tensor(values).float()
        log_probs = torch.tensor(log_probs).float()

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value.item())

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "state": states,
            "action": actions,
            "log_prob": log_probs,
            "advantage": advantages,
            "return": returns,
            "value": values,
        }

    def ppo_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.

        Args:
            rollout: Rollout buffer from collect_rollout

        Returns:
            Training metrics
        """
        config = self.policy_config
        n_samples = len(rollout["state"])
        batch_size = n_samples // config.n_minibatches

        all_metrics = []

        for epoch in range(config.n_epochs):
            # Random permutation
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch = {
                    k: v[batch_indices].to(self.device)
                    for k, v in rollout.items()
                }

                with autocast(enabled=self.use_amp):
                    metrics = self._ppo_step(batch)

                all_metrics.append(metrics)

                # Early stopping if KL divergence too high
                if metrics["approx_kl"] > 0.02:
                    if is_main_process():
                        logger.info(f"Early stopping at epoch {epoch} due to high KL")
                    break

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def train_online(
        self,
        env,
        n_iterations: int = 1000,
        rollout_steps: int = 2048,
        eval_frequency: int = 10,
    ) -> Dict[str, List]:
        """
        Online PPO training loop.

        Args:
            env: Trading environment
            n_iterations: Number of training iterations
            rollout_steps: Steps per rollout
            eval_frequency: Evaluation frequency

        Returns:
            Training history
        """
        history = {"rewards": [], "metrics": []}

        for iteration in range(n_iterations):
            # Collect rollout
            rollout = self.collect_rollout(env, rollout_steps)

            # Move to device
            rollout = {k: v.to(self.device) for k, v in rollout.items()}

            # PPO update
            metrics = self.ppo_update(rollout)
            history["metrics"].append(metrics)

            # Logging
            if iteration % 10 == 0 and is_main_process():
                logger.info(
                    f"Iteration {iteration}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"entropy={metrics['entropy']:.4f}"
                )

            # Evaluation
            if iteration % eval_frequency == 0:
                eval_metrics = self.evaluate(env, n_episodes=5)
                history["rewards"].append(eval_metrics["mean_reward"])

                if is_main_process():
                    logger.info(
                        f"Evaluation: mean_reward={eval_metrics['mean_reward']:.2f}, "
                        f"std={eval_metrics['std_reward']:.2f}"
                    )

        return history

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
    ) -> Dict[str, float]:
        """Evaluate policy."""
        self.models.eval()

        episode_rewards = []

        for ep in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    z = self.encoder(state_tensor)
                    action_dist, _ = self.policy(z)
                    action = action_dist.mean.cpu().numpy()[0]

                state, reward, done, _ = env.step(action)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }
