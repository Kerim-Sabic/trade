#!/usr/bin/env python3
"""Script to run distributed training on dual RTX 5080 GPUs."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from loguru import logger

from cryptoai.training import UnifiedTrainer
from cryptoai.training.ddp import DDPConfig, launch_distributed
from cryptoai.training.trainer import TrainingConfig
from cryptoai.training.data_loader import (
    MarketDataLoader,
    DataConfig,
    create_synthetic_data,
)
from cryptoai.utils.config import load_config
from cryptoai.utils.logging import setup_logging


def train_worker(rank: int, world_size: int, ddp_config: DDPConfig, **kwargs):
    """Training worker function."""
    config = kwargs.get("config", {})
    data_config = kwargs.get("data_config")

    # Setup logging for this rank
    if rank == 0:
        setup_logging(level="INFO")

    logger.info(f"Rank {rank}/{world_size}: Initializing training")

    # Training configuration
    training_config = TrainingConfig(
        state_dim=config.get("model", {}).get("state_dim", 200),
        action_dim=config.get("model", {}).get("action_dim", 4),
        hidden_dim=config.get("model", {}).get("hidden_dim", 256),
        latent_dim=config.get("model", {}).get("latent_dim", 128),
        batch_size=config.get("training", {}).get("batch_size", 64),
        encoder_pretrain_epochs=config.get("training", {}).get("encoder_epochs", 10),
        world_model_epochs=config.get("training", {}).get("world_model_epochs", 50),
        policy_epochs=config.get("training", {}).get("policy_epochs", 100),
    )

    # Initialize trainer
    trainer = UnifiedTrainer(training_config, ddp_config, rank)

    # Create data loaders
    data_loader = MarketDataLoader(data_config)
    train_loader = data_loader.create_train_loader(rank, world_size)
    val_loader = data_loader.create_val_loader(rank, world_size)

    # Run training
    logger.info(f"Rank {rank}: Starting full training pipeline")
    history = trainer.full_training(train_loader, val_loader)

    if rank == 0:
        logger.info(f"Training complete. History: {history}")


def main():
    parser = argparse.ArgumentParser(description="Run distributed training")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data for testing",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="Number of GPUs to use",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Generate synthetic data if requested
    if args.synthetic:
        logger.info("Generating synthetic training data")
        os.makedirs(args.data_dir, exist_ok=True)
        create_synthetic_data(
            os.path.join(args.data_dir, "train.h5"),
            n_samples=10000,
        )
        create_synthetic_data(
            os.path.join(args.data_dir, "val.h5"),
            n_samples=2000,
        )

    # Data configuration
    data_config = DataConfig(
        data_dir=args.data_dir,
        train_files=[os.path.join(args.data_dir, "train.h5")],
        val_files=[os.path.join(args.data_dir, "val.h5")],
        batch_size=config.get("training", {}).get("batch_size", 64),
    )

    # DDP configuration
    world_size = min(args.gpus, torch.cuda.device_count())
    ddp_config = DDPConfig(world_size=world_size)

    logger.info(f"Starting distributed training on {world_size} GPUs")

    # Launch training
    launch_distributed(
        train_worker,
        world_size=world_size,
        config=ddp_config,
        config=config,
        data_config=data_config,
    )


if __name__ == "__main__":
    main()
