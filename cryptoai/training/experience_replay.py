"""Experience replay buffers for reinforcement learning."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
import torch
from loguru import logger


class Experience(NamedTuple):
    """Single experience tuple."""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Optional[Dict] = None


@dataclass
class ReplayConfig:
    """Configuration for experience replay."""

    capacity: int = 1_000_000
    state_dim: int = 200
    action_dim: int = 4
    sequence_length: int = 100

    # Prioritized replay settings
    alpha: float = 0.6  # Priority exponent
    beta_start: float = 0.4  # Initial importance sampling weight
    beta_frames: int = 100000  # Frames to anneal beta
    epsilon: float = 1e-6  # Small constant for priorities

    # N-step returns
    n_step: int = 3
    gamma: float = 0.99


class ExperienceReplay:
    """
    Standard experience replay buffer.

    Stores transitions and provides random sampling.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        state_shape: Tuple[int, ...] = (100, 200),
        action_dim: int = 4,
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_dim = action_dim

        # Storage arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Buffer state
        self._ptr = 0
        self._size = 0

    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        idx = self._ptr

        self.states[idx] = experience.state
        self.actions[idx] = experience.action
        self.rewards[idx] = experience.reward
        self.next_states[idx] = experience.next_state
        self.dones[idx] = float(experience.done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add batch of experiences."""
        batch_size = len(states)

        for i in range(batch_size):
            self.add(Experience(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=dones[i],
            ))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch."""
        indices = np.random.randint(0, self._size, size=batch_size)

        return {
            "states": torch.from_numpy(self.states[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_states": torch.from_numpy(self.next_states[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity


class PrioritizedExperienceReplay:
    """
    Prioritized experience replay buffer.

    Samples transitions based on TD-error priority.
    Uses sum-tree for efficient sampling.
    """

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.capacity = config.capacity

        # Storage
        self.states = np.zeros(
            (config.capacity, config.sequence_length, config.state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros((config.capacity, config.action_dim), dtype=np.float32)
        self.rewards = np.zeros(config.capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (config.capacity, config.sequence_length, config.state_dim),
            dtype=np.float32,
        )
        self.dones = np.zeros(config.capacity, dtype=np.float32)

        # Priority tree
        tree_capacity = 1
        while tree_capacity < config.capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity

        self.sum_tree = np.zeros(2 * tree_capacity - 1, dtype=np.float64)
        self.min_tree = np.full(2 * tree_capacity - 1, float("inf"), dtype=np.float64)

        # State
        self._ptr = 0
        self._size = 0
        self._max_priority = 1.0

        # Beta annealing
        self._beta = config.beta_start
        self._frame = 0

    def _update_tree(self, idx: int, priority: float) -> None:
        """Update priority trees."""
        tree_idx = idx + self.tree_capacity - 1

        # Update sum tree
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += change

        # Update min tree
        tree_idx = idx + self.tree_capacity - 1
        self.min_tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = 2 * tree_idx + 1
            right = 2 * tree_idx + 2
            self.min_tree[tree_idx] = min(
                self.min_tree[left],
                self.min_tree[right] if right < len(self.min_tree) else float("inf"),
            )

    def _get_leaf(self, value: float) -> int:
        """Get leaf index for given value using sum tree."""
        idx = 0

        while idx < self.tree_capacity - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2

            if value <= self.sum_tree[left]:
                idx = left
            else:
                value -= self.sum_tree[left]
                idx = right

        return idx - (self.tree_capacity - 1)

    def add(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add experience with priority."""
        idx = self._ptr

        # Store experience
        self.states[idx] = experience.state
        self.actions[idx] = experience.action
        self.rewards[idx] = experience.reward
        self.next_states[idx] = experience.next_state
        self.dones[idx] = float(experience.done)

        # Set priority
        if priority is None:
            priority = self._max_priority

        priority = (priority + self.config.epsilon) ** self.config.alpha
        self._update_tree(idx, priority)
        self._max_priority = max(self._max_priority, priority)

        # Update pointers
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample prioritized batch.

        Returns:
            batch: Dictionary of tensors
            indices: Sampled indices
            weights: Importance sampling weights
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)

        # Update beta
        self._frame += 1
        self._beta = min(
            1.0,
            self.config.beta_start + self._frame * (1.0 - self.config.beta_start) / self.config.beta_frames,
        )

        # Calculate segment size
        total_priority = self.sum_tree[0]
        segment_size = total_priority / batch_size

        # Min probability for normalization
        min_prob = self.min_tree[0] / total_priority
        max_weight = (min_prob * self._size) ** (-self._beta)

        for i in range(batch_size):
            # Sample from segment
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)

            idx = self._get_leaf(value)
            indices[i] = idx

            # Calculate importance weight
            prob = self.sum_tree[idx + self.tree_capacity - 1] / total_priority
            weight = (prob * self._size) ** (-self._beta) / max_weight
            weights[i] = weight

        batch = {
            "states": torch.from_numpy(self.states[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_states": torch.from_numpy(self.next_states[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities after learning."""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.config.epsilon) ** self.config.alpha
            self._update_tree(idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self._size


class NStepReplayBuffer:
    """
    N-step return experience replay.

    Computes n-step returns for more efficient learning.
    """

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.n_step = config.n_step
        self.gamma = config.gamma

        # Base buffer
        self.buffer = PrioritizedExperienceReplay(config)

        # N-step buffer (stores partial transitions)
        self._n_step_buffer: List[Experience] = []

    def add(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add experience with n-step return computation."""
        self._n_step_buffer.append(experience)

        # Check if we have enough for n-step return
        if len(self._n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        n_step_return = 0.0
        for i in range(self.n_step):
            n_step_return += (self.gamma ** i) * self._n_step_buffer[i].reward

        # Create n-step transition
        first = self._n_step_buffer[0]
        last = self._n_step_buffer[-1]

        n_step_experience = Experience(
            state=first.state,
            action=first.action,
            reward=n_step_return,
            next_state=last.next_state,
            done=last.done,
        )

        # Add to base buffer
        self.buffer.add(n_step_experience, priority)

        # Remove oldest
        self._n_step_buffer.pop(0)

        # Handle episode termination
        if experience.done:
            self._flush_n_step_buffer()

    def _flush_n_step_buffer(self) -> None:
        """Flush remaining transitions at episode end."""
        while self._n_step_buffer:
            # Compute partial n-step return
            n_step_return = 0.0
            for i, exp in enumerate(self._n_step_buffer):
                n_step_return += (self.gamma ** i) * exp.reward

            first = self._n_step_buffer[0]
            last = self._n_step_buffer[-1]

            experience = Experience(
                state=first.state,
                action=first.action,
                reward=n_step_return,
                next_state=last.next_state,
                done=last.done,
            )

            self.buffer.add(experience)
            self._n_step_buffer.pop(0)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample from buffer."""
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities."""
        self.buffer.update_priorities(indices, priorities)

    def __len__(self) -> int:
        return len(self.buffer)


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) for goal-conditioned learning.

    Useful for learning from failures by relabeling goals.
    """

    def __init__(
        self,
        config: ReplayConfig,
        goal_dim: int = 4,
        k_future: int = 4,
    ):
        self.config = config
        self.goal_dim = goal_dim
        self.k_future = k_future

        # Base buffer
        self.buffer = PrioritizedExperienceReplay(config)

        # Episode buffer
        self._episode_buffer: List[Tuple[Experience, np.ndarray]] = []

    def add(
        self,
        experience: Experience,
        goal: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> None:
        """Add experience with goal information."""
        self._episode_buffer.append((experience, achieved_goal))

        if experience.done:
            self._store_episode()

    def _store_episode(self) -> None:
        """Store episode with hindsight goal relabeling."""
        episode_length = len(self._episode_buffer)

        for t, (experience, achieved_goal) in enumerate(self._episode_buffer):
            # Original experience
            self.buffer.add(experience)

            # Future relabeling
            future_indices = np.random.randint(t, episode_length, size=self.k_future)

            for future_t in future_indices:
                _, future_achieved = self._episode_buffer[future_t]

                # Relabel with future achieved goal
                relabeled = Experience(
                    state=experience.state,
                    action=experience.action,
                    reward=self._compute_reward(achieved_goal, future_achieved),
                    next_state=experience.next_state,
                    done=np.allclose(achieved_goal, future_achieved),
                )

                self.buffer.add(relabeled)

        self._episode_buffer.clear()

    def _compute_reward(
        self,
        achieved: np.ndarray,
        desired: np.ndarray,
        threshold: float = 0.05,
    ) -> float:
        """Compute sparse reward based on goal achievement."""
        distance = np.linalg.norm(achieved - desired)
        return 0.0 if distance < threshold else -1.0

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample from buffer."""
        return self.buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DistributedReplayBuffer:
    """
    Distributed replay buffer for multi-GPU training.

    Each GPU maintains a local buffer, with periodic synchronization.
    """

    def __init__(
        self,
        config: ReplayConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Local buffer
        self.local_buffer = PrioritizedExperienceReplay(config)

        # Sync settings
        self.sync_frequency = 1000  # Sync every N additions
        self._additions_since_sync = 0

    def add(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add to local buffer."""
        self.local_buffer.add(experience, priority)

        self._additions_since_sync += 1
        if self._additions_since_sync >= self.sync_frequency:
            self._sync_buffers()
            self._additions_since_sync = 0

    def _sync_buffers(self) -> None:
        """Synchronize high-priority experiences across GPUs."""
        if self.world_size <= 1:
            return

        import torch.distributed as dist

        # Get top priorities from local buffer
        top_k = min(100, len(self.local_buffer))
        if top_k == 0:
            return

        # Find top priority indices
        priorities = self.local_buffer.sum_tree[
            self.local_buffer.tree_capacity - 1:
            self.local_buffer.tree_capacity - 1 + len(self.local_buffer)
        ]
        top_indices = np.argsort(priorities)[-top_k:]

        # Gather states to share
        shared_states = self.local_buffer.states[top_indices]

        # All-gather across processes
        gathered_states = [
            torch.zeros_like(torch.from_numpy(shared_states))
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_states, torch.from_numpy(shared_states))

        logger.debug(f"Rank {self.rank}: Synced {top_k} experiences across {self.world_size} GPUs")

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample from local buffer."""
        return self.local_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update local priorities."""
        self.local_buffer.update_priorities(indices, priorities)

    def __len__(self) -> int:
        return len(self.local_buffer)
