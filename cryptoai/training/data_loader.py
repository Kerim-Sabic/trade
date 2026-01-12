"""Data loading utilities for training."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from loguru import logger
import h5py
import json


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Data paths
    data_dir: str = "data"
    train_files: List[str] = None
    val_files: List[str] = None

    # Sequence settings
    sequence_length: int = 100
    prediction_horizon: int = 10

    # Batch settings
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2

    # Normalization
    normalize: bool = True
    clip_outliers: bool = True
    outlier_threshold: float = 5.0


class MarketDataset(Dataset):
    """
    Dataset for market data stored in HDF5 format.

    Expected HDF5 structure:
    - /states: (N, T, F) - State sequences
    - /actions: (N, A) - Actions taken
    - /rewards: (N,) - Rewards received
    - /next_states: (N, T, F) - Next state sequences
    - /dones: (N,) - Episode termination flags
    - /metadata: JSON string with normalization stats
    """

    def __init__(
        self,
        file_paths: List[str],
        sequence_length: int = 100,
        prediction_horizon: int = 10,
        normalize: bool = True,
        transform: Optional[callable] = None,
    ):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.transform = transform

        # Index mapping: (file_idx, sample_idx)
        self._indices: List[Tuple[int, int]] = []
        self._file_handles: Dict[int, h5py.File] = {}
        self._normalization_stats: Dict[str, np.ndarray] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Build index mapping for all samples across files."""
        for file_idx, file_path in enumerate(self.file_paths):
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                continue

            with h5py.File(file_path, "r") as f:
                n_samples = f["states"].shape[0]

                # Load normalization stats from first file
                if self.normalize and "metadata" in f.attrs:
                    metadata = json.loads(f.attrs["metadata"])
                    if "mean" in metadata and "std" in metadata:
                        self._normalization_stats = {
                            "mean": np.array(metadata["mean"], dtype=np.float32),
                            "std": np.array(metadata["std"], dtype=np.float32),
                        }

                for sample_idx in range(n_samples):
                    self._indices.append((file_idx, sample_idx))

        logger.info(f"Built index with {len(self._indices)} samples from {len(self.file_paths)} files")

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        """Get or create file handle."""
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = h5py.File(
                self.file_paths[file_idx], "r", swmr=True
            )
        return self._file_handles[file_idx]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, sample_idx = self._indices[idx]
        f = self._get_file_handle(file_idx)

        # Load data
        state = f["states"][sample_idx].astype(np.float32)
        action = f["actions"][sample_idx].astype(np.float32)
        reward = f["rewards"][sample_idx].astype(np.float32)
        next_state = f["next_states"][sample_idx].astype(np.float32)
        done = f["dones"][sample_idx].astype(np.float32)

        # Normalize
        if self.normalize and self._normalization_stats:
            mean = self._normalization_stats["mean"]
            std = self._normalization_stats["std"] + 1e-8
            state = (state - mean) / std
            next_state = (next_state - mean) / std

        # Convert to tensors
        sample = {
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward),
            "next_state": torch.from_numpy(next_state),
            "done": torch.tensor(done),
        }

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def close(self) -> None:
        """Close all file handles."""
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()


class StreamingMarketDataset(Dataset):
    """
    Streaming dataset for continuous market data.

    Generates sequences on-the-fly from raw tick data.
    """

    def __init__(
        self,
        data_dir: str,
        assets: List[str],
        sequence_length: int = 100,
        stride: int = 1,
        feature_dim: int = 200,
    ):
        self.data_dir = Path(data_dir)
        self.assets = assets
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_dim = feature_dim

        # Memory-mapped data arrays
        self._data: Dict[str, np.memmap] = {}
        self._lengths: Dict[str, int] = {}
        self._total_samples = 0

        self._load_data()

    def _load_data(self) -> None:
        """Load memory-mapped data files."""
        for asset in self.assets:
            data_path = self.data_dir / f"{asset}_features.npy"

            if data_path.exists():
                # Memory-map for efficient access
                data = np.load(str(data_path), mmap_mode="r")
                self._data[asset] = data
                self._lengths[asset] = max(0, len(data) - self.sequence_length)
                self._total_samples += self._lengths[asset] // self.stride

                logger.info(f"Loaded {asset}: {len(data)} timesteps")
            else:
                logger.warning(f"Data file not found: {data_path}")

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which asset this index belongs to
        current_idx = idx
        for asset in self.assets:
            asset_samples = self._lengths[asset] // self.stride

            if current_idx < asset_samples:
                start = current_idx * self.stride
                end = start + self.sequence_length

                data = self._data[asset]
                sequence = data[start:end].astype(np.float32)

                return {
                    "sequence": torch.from_numpy(sequence),
                    "asset": asset,
                    "start_idx": start,
                }

            current_idx -= asset_samples

        raise IndexError(f"Index {idx} out of range")


class ExperienceDataset(Dataset):
    """
    Dataset for RL experience replay.

    Stores transitions from trading episodes.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = 200,
        action_dim: int = 4,
        sequence_length: int = 100,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length

        # Pre-allocate arrays for efficiency
        self.states = np.zeros(
            (capacity, sequence_length, state_dim), dtype=np.float32
        )
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (capacity, sequence_length, state_dim), dtype=np.float32
        )
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Metadata
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.timestamps = np.zeros(capacity, dtype=np.int64)

        self._size = 0
        self._ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
    ) -> None:
        """Add a transition to the dataset."""
        idx = self._ptr

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.priorities[idx] = priority
        self.timestamps[idx] = self._size

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": torch.from_numpy(self.states[idx]),
            "action": torch.from_numpy(self.actions[idx]),
            "reward": torch.tensor(self.rewards[idx]),
            "next_state": torch.from_numpy(self.next_states[idx]),
            "done": torch.tensor(self.dones[idx]),
            "priority": torch.tensor(self.priorities[idx]),
        }

    def sample_batch(
        self,
        batch_size: int,
        prioritized: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        if prioritized:
            # Prioritized sampling
            probs = self.priorities[: self._size] / self.priorities[: self._size].sum()
            indices = np.random.choice(self._size, size=batch_size, p=probs)
        else:
            indices = np.random.randint(0, self._size, size=batch_size)

        return {
            "state": torch.from_numpy(self.states[indices]),
            "action": torch.from_numpy(self.actions[indices]),
            "reward": torch.from_numpy(self.rewards[indices]),
            "next_state": torch.from_numpy(self.next_states[indices]),
            "done": torch.from_numpy(self.dones[indices]),
            "indices": torch.from_numpy(indices),
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """Update priorities for prioritized experience replay."""
        self.priorities[indices] = priorities

    def save(self, path: str) -> None:
        """Save dataset to HDF5."""
        with h5py.File(path, "w") as f:
            f.create_dataset("states", data=self.states[: self._size])
            f.create_dataset("actions", data=self.actions[: self._size])
            f.create_dataset("rewards", data=self.rewards[: self._size])
            f.create_dataset("next_states", data=self.next_states[: self._size])
            f.create_dataset("dones", data=self.dones[: self._size])
            f.create_dataset("priorities", data=self.priorities[: self._size])
            f.attrs["size"] = self._size

        logger.info(f"Saved {self._size} transitions to {path}")

    def load(self, path: str) -> None:
        """Load dataset from HDF5."""
        with h5py.File(path, "r") as f:
            size = f.attrs["size"]
            self.states[:size] = f["states"][:]
            self.actions[:size] = f["actions"][:]
            self.rewards[:size] = f["rewards"][:]
            self.next_states[:size] = f["next_states"][:]
            self.dones[:size] = f["dones"][:]
            self.priorities[:size] = f["priorities"][:]
            self._size = size
            self._ptr = size % self.capacity

        logger.info(f"Loaded {size} transitions from {path}")


class MarketDataLoader:
    """
    High-level data loader factory for market data.

    Handles dataset creation, distributed sampling, and batching.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def create_train_loader(
        self,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        """Create training data loader with distributed sampling."""
        dataset = MarketDataset(
            file_paths=self.config.train_files or [],
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            normalize=self.config.normalize,
        )

        return create_distributed_loader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=True,
        )

    def create_val_loader(
        self,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        """Create validation data loader."""
        dataset = MarketDataset(
            file_paths=self.config.val_files or [],
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            normalize=self.config.normalize,
        )

        return create_distributed_loader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
        )


def create_distributed_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a DataLoader with distributed sampling support.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        rank: Process rank
        world_size: Total number of processes
        shuffle: Whether to shuffle data
        drop_last: Drop last incomplete batch
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch

    Returns:
        Configured DataLoader
    """
    sampler = None

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def create_synthetic_data(
    output_path: str,
    n_samples: int = 10000,
    sequence_length: int = 100,
    feature_dim: int = 200,
    action_dim: int = 4,
) -> None:
    """
    Create synthetic training data for testing.

    Args:
        output_path: Path to save HDF5 file
        n_samples: Number of samples
        sequence_length: Sequence length
        feature_dim: Feature dimension
        action_dim: Action dimension
    """
    logger.info(f"Creating synthetic data: {n_samples} samples")

    # Generate synthetic data
    states = np.random.randn(n_samples, sequence_length, feature_dim).astype(np.float32)
    actions = np.random.randn(n_samples, action_dim).astype(np.float32)
    rewards = np.random.randn(n_samples).astype(np.float32)
    next_states = np.random.randn(n_samples, sequence_length, feature_dim).astype(np.float32)
    dones = np.random.binomial(1, 0.01, n_samples).astype(np.float32)

    # Compute normalization stats
    mean = states.mean(axis=(0, 1))
    std = states.std(axis=(0, 1))

    metadata = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": n_samples,
        "sequence_length": sequence_length,
        "feature_dim": feature_dim,
    }

    # Save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("states", data=states, compression="gzip")
        f.create_dataset("actions", data=actions, compression="gzip")
        f.create_dataset("rewards", data=rewards, compression="gzip")
        f.create_dataset("next_states", data=next_states, compression="gzip")
        f.create_dataset("dones", data=dones, compression="gzip")
        f.attrs["metadata"] = json.dumps(metadata)

    logger.info(f"Saved synthetic data to {output_path}")
