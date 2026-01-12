"""Device and distributed computing utilities for CryptoAI."""

import os
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger


def get_device(device_str: str = "auto", gpu_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device_str: Device specification ("auto", "cuda", "cpu")
        gpu_id: Specific GPU ID to use

    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

    if device_str == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")

        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")

        return torch.device("cuda")

    return torch.device("cpu")


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> Tuple[int, int, int]:
    """
    Setup distributed training environment.

    Args:
        backend: Distributed backend ("nccl", "gloo")
        init_method: Initialization method

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    # Check if distributed environment variables are set
    if "RANK" not in os.environ:
        logger.info("Distributed environment not detected, running in single-GPU mode")
        return 0, 0, 1

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

    # Set device for this process
    torch.cuda.set_device(local_rank)

    logger.info(f"Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list[int]] = None,
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: PyTorch model
        device_ids: GPU device IDs
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    if device_ids is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_ids = [local_rank]

    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def get_amp_context(precision: str = "amp", device_type: str = "cuda"):
    """
    Get automatic mixed precision context.

    Args:
        precision: Precision mode ("fp32", "fp16", "bf16", "amp")
        device_type: Device type for AMP

    Returns:
        AMP autocast context manager
    """
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    elif precision == "fp16":
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    elif precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:  # amp - automatic
        return torch.autocast(device_type=device_type)


class GradScaler:
    """Wrapper for gradient scaling with mixed precision."""

    def __init__(self, precision: str = "amp", enabled: bool = True):
        self.precision = precision
        self.enabled = enabled and precision in ("fp16", "amp")

        if self.enabled:
            self._scaler = torch.amp.GradScaler()
        else:
            self._scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients and step optimizer."""
        if self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        """Update scaler."""
        if self._scaler is not None:
            self._scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients."""
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
