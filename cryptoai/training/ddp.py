"""Distributed Data Parallel (DDP) utilities for multi-GPU training."""

import os
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger

# Handle PyTorch 2.0+ AMP API changes
try:
    from torch.amp import GradScaler, autocast as _autocast
    def autocast(enabled=True, device_type="cuda"):
        return _autocast(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast


@dataclass
class DDPConfig:
    """Configuration for DDP training."""

    # Distributed settings
    world_size: int = 2  # Number of GPUs (2x RTX 5080)
    backend: str = "nccl"  # NCCL for GPU communication
    master_addr: str = "localhost"
    master_port: str = "12355"

    # Training settings
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    find_unused_parameters: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """
    Initialize distributed process group.

    Args:
        rank: Process rank (GPU index)
        world_size: Total number of processes
        backend: Communication backend (nccl for GPU)
        master_addr: Master node address
        master_port: Master node port
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # Set device for this process
    torch.cuda.set_device(rank)

    logger.info(f"DDP initialized: rank {rank}/{world_size}")


def cleanup_ddp() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP cleanup completed")


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        reduction: Reduction operation (mean, sum)

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    if reduction == "mean":
        rt /= world_size

    return rt


class DDPTrainer:
    """
    Base trainer class with DDP support.

    Designed for dual RTX 5080 GPUs with:
    - Automatic Mixed Precision (AMP)
    - Gradient accumulation
    - Checkpoint management
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        config: DDPConfig,
        rank: int = 0,
    ):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")

        # Move model to device and wrap with DDP
        self.model = model.to(self.device)

        if config.world_size > 1 and dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=config.find_unused_parameters,
            )
            self.is_distributed = True
        else:
            self.is_distributed = False

        # AMP scaler - explicitly specify device for PyTorch 2.0+
        if config.use_amp:
            try:
                self.scaler = GradScaler(device="cuda")
            except TypeError:
                # Fallback for older PyTorch versions
                self.scaler = GradScaler()
        else:
            self.scaler = None
        self.use_amp = config.use_amp

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")

        # Optimizer and scheduler (to be set by subclasses)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # Metrics tracking
        self.metrics_history: Dict[str, list] = {}

    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrapped from DDP)."""
        if self.is_distributed:
            return self.model.module
        return self.model

    def save_checkpoint(
        self,
        path: str,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save training checkpoint."""
        if not is_main_process():
            return

        state = {
            "model_state_dict": self.get_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if extra_state:
            state.update(extra_state)

        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        # Map to correct device
        map_location = {"cuda:0": f"cuda:{self.rank}"}
        state = torch.load(path, map_location=map_location)

        self.get_model().load_state_dict(state["model_state_dict"])

        if self.optimizer and state.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self.scaler and state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        self.best_metric = state.get("best_metric", float("inf"))

        logger.info(f"Checkpoint loaded: {path}")
        return state

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step with AMP support.

        Override in subclasses for specific training logic.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of loss values
        """
        raise NotImplementedError("Subclasses must implement train_step")

    def _backward_step(
        self,
        loss: torch.Tensor,
        step_optimizer: bool = True,
    ) -> None:
        """
        Backward pass with AMP and gradient accumulation.

        Args:
            loss: Loss tensor
            step_optimizer: Whether to step optimizer
        """
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()

            if step_optimizer:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()

            if step_optimizer:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Average metrics for the epoch
        """
        self.model.train()
        self.epoch = epoch

        epoch_metrics: Dict[str, list] = {}
        accumulation_count = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Determine if we should step optimizer
            accumulation_count += 1
            step_optimizer = (
                accumulation_count % self.config.gradient_accumulation_steps == 0
            )

            # Training step
            with autocast(enabled=self.use_amp):
                metrics = self.train_step(batch)

            # Backward pass
            if "loss" in metrics:
                self._backward_step(
                    torch.tensor(metrics["loss"], device=self.device),
                    step_optimizer=step_optimizer,
                )

            # Track metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Increment step
            if step_optimizer:
                self.global_step += 1

                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()

                # Checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"checkpoint_step_{self.global_step}.pt",
                    )
                    self.save_checkpoint(checkpoint_path)

            # Logging
            if batch_idx % 100 == 0 and is_main_process():
                loss_str = ", ".join(
                    f"{k}: {v[-1]:.4f}" for k, v in epoch_metrics.items()
                )
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: {loss_str}"
                )

        # Average metrics across epoch
        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

        # Reduce across processes
        if self.is_distributed:
            for key in avg_metrics:
                avg_metrics[key] = reduce_tensor(
                    torch.tensor(avg_metrics[key], device=self.device)
                ).item()

        return avg_metrics

    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validation pass.

        Args:
            dataloader: Validation data loader

        Returns:
            Validation metrics
        """
        self.model.eval()
        val_metrics: Dict[str, list] = {}

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with autocast(enabled=self.use_amp):
                    metrics = self.train_step(batch)

                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)

        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}

        # Reduce across processes
        if self.is_distributed:
            for key in avg_metrics:
                avg_metrics[key] = reduce_tensor(
                    torch.tensor(avg_metrics[key], device=self.device)
                ).item()

        return avg_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training history
        """
        history = {"train": [], "val": []}
        patience_counter = 0

        for epoch in range(num_epochs):
            # Update sampler for distributed training
            if hasattr(train_loader, "sampler") and isinstance(
                train_loader.sampler, DistributedSampler
            ):
                train_loader.sampler.set_epoch(epoch)

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train"].append(train_metrics)

            if is_main_process():
                logger.info(f"Epoch {epoch} training: {train_metrics}")

            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                history["val"].append(val_metrics)

                if is_main_process():
                    logger.info(f"Epoch {epoch} validation: {val_metrics}")

                # Early stopping
                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    patience_counter = 0

                    # Save best model
                    best_path = os.path.join(
                        self.config.checkpoint_dir, "best_model.pt"
                    )
                    self.save_checkpoint(best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            synchronize()

        return history

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for tracking."""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((step, value))


def launch_distributed(
    train_fn: Callable,
    world_size: int = 2,
    config: Optional[DDPConfig] = None,
    **kwargs,
) -> None:
    """
    Launch distributed training.

    Args:
        train_fn: Training function that takes (rank, world_size, config, **kwargs)
        world_size: Number of GPUs
        config: DDP configuration
        **kwargs: Additional arguments to pass to train_fn
    """
    import torch.multiprocessing as mp

    config = config or DDPConfig(world_size=world_size)

    mp.spawn(
        _distributed_worker,
        args=(world_size, train_fn, config, kwargs),
        nprocs=world_size,
        join=True,
    )


def _distributed_worker(
    rank: int,
    world_size: int,
    train_fn: Callable,
    config: DDPConfig,
    kwargs: Dict[str, Any],
) -> None:
    """Worker function for distributed training."""
    setup_ddp(rank, world_size, config.backend, config.master_addr, config.master_port)

    try:
        train_fn(rank, world_size, config, **kwargs)
    finally:
        cleanup_ddp()
