"""Reproducibility utilities for CryptoAI."""

import os
import random
from typing import Optional
import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set environment variable for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed}")


def ensure_reproducibility(
    seed: int = 42,
    deterministic_algorithms: bool = True,
    warn_only: bool = True,
) -> None:
    """
    Ensure full reproducibility for experiments.

    Args:
        seed: Random seed value
        deterministic_algorithms: Use deterministic algorithms
        warn_only: Only warn about non-deterministic operations
    """
    set_seed(seed)

    if deterministic_algorithms:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # Set cuDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for CUDA
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        logger.info("Deterministic algorithms enabled")
    else:
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled (non-deterministic)")


def get_experiment_hash(
    config: dict,
    include_timestamp: bool = False,
) -> str:
    """
    Generate a unique hash for an experiment configuration.

    Args:
        config: Configuration dictionary
        include_timestamp: Include timestamp in hash

    Returns:
        Experiment hash string
    """
    import hashlib
    import json
    from datetime import datetime

    # Convert config to sorted JSON string
    config_str = json.dumps(config, sort_keys=True, default=str)

    if include_timestamp:
        config_str += datetime.utcnow().isoformat()

    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class CheckpointManager:
    """Manage model checkpoints for reproducibility."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list[str] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        state: dict,
        name: str,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint.

        Args:
            state: State dictionary to save
            name: Checkpoint name
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save(state, checkpoint_path)

        self.checkpoints.append(checkpoint_path)

        # Remove old checkpoints if exceeding max
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint) and "best" not in old_checkpoint:
                os.remove(old_checkpoint)

        # Save best checkpoint separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def load(
        self,
        name: Optional[str] = None,
        map_location: Optional[str] = None,
    ) -> dict:
        """
        Load checkpoint.

        Args:
            name: Checkpoint name (None for latest)
            map_location: Device mapping location

        Returns:
            Loaded state dictionary
        """
        if name is None:
            # Load latest checkpoint
            if not self.checkpoints:
                raise FileNotFoundError("No checkpoints available")
            checkpoint_path = self.checkpoints[-1]
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return state

    def load_best(self, map_location: Optional[str] = None) -> dict:
        """Load best checkpoint."""
        return self.load("best", map_location=map_location)
