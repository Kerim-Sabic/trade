"""Main entry point for the CryptoAI Trading System."""

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from cryptoai.utils.config import load_config
from cryptoai.utils.logging import setup_logging
from cryptoai.utils.device import setup_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CryptoAI Trading System")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["shadow", "paper", "live", "backtest", "train"],
        default="shadow",
        help="Execution mode",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=["BTCUSDT"],
        help="Assets to trade",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


async def run_trading(config: dict, args) -> None:
    """Run trading loop."""
    from cryptoai.deployment import DeploymentOrchestrator, DeploymentConfig
    from cryptoai.deployment.model_server import InferenceEngine, InferenceConfig
    from cryptoai.monitoring import MetricsCollector, DriftDetector, AlertManager
    from cryptoai.monitoring.dashboard import DashboardServer, DashboardConfig
    from cryptoai.monitoring.drift_detector import DriftConfig
    from cryptoai.execution import TradingExecutor, ExecutionConfig
    from cryptoai.execution.exchange_client import create_exchange_client, ExchangeType
    from cryptoai.risk_engine import RiskController

    logger.info(f"Starting CryptoAI Trading System in {args.mode} mode")

    # Initialize monitoring
    metrics = MetricsCollector()
    drift_detector = DriftDetector(DriftConfig())
    alert_manager = AlertManager()

    # Start dashboard
    dashboard = DashboardServer(
        metrics=metrics,
        drift_detector=drift_detector,
        alert_manager=alert_manager,
        config=DashboardConfig(),
    )
    dashboard.start()

    # Initialize deployment
    deployment_config = DeploymentConfig(
        execution_mode=args.mode,
        checkpoint_path=args.model or config.get("model", {}).get("checkpoint_path", ""),
    )
    orchestrator = DeploymentOrchestrator(deployment_config)

    # Load models if checkpoint provided
    if args.model:
        try:
            from cryptoai.encoders import UnifiedStateEncoder
            from cryptoai.decision_engine import PolicyNetwork
            from cryptoai.world_model import WorldModel
            from cryptoai.black_swan import BlackSwanDetector

            device = setup_device()

            # Load checkpoint
            checkpoint = torch.load(args.model, map_location=device)

            # Initialize models (dimensions from config)
            encoder = UnifiedStateEncoder(
                state_dim=config.get("model", {}).get("state_dim", 200),
                hidden_dim=config.get("model", {}).get("hidden_dim", 256),
                output_dim=config.get("model", {}).get("latent_dim", 128),
            ).to(device)

            policy = PolicyNetwork(
                state_dim=config.get("model", {}).get("latent_dim", 128),
                action_dim=config.get("model", {}).get("action_dim", 4),
                hidden_dim=config.get("model", {}).get("hidden_dim", 256),
            ).to(device)

            world_model = WorldModel(
                state_dim=config.get("model", {}).get("latent_dim", 128),
                action_dim=config.get("model", {}).get("action_dim", 4),
                hidden_dim=config.get("model", {}).get("hidden_dim", 256),
            ).to(device)

            black_swan = BlackSwanDetector(
                state_dim=config.get("model", {}).get("latent_dim", 128),
                hidden_dim=config.get("model", {}).get("hidden_dim", 256),
            ).to(device)

            # Load weights
            if "encoder" in checkpoint:
                encoder.load_state_dict(checkpoint["encoder"])
            if "policy" in checkpoint:
                policy.load_state_dict(checkpoint["policy"])
            if "world_model" in checkpoint:
                world_model.load_state_dict(checkpoint["world_model"])
            if "black_swan" in checkpoint:
                black_swan.load_state_dict(checkpoint["black_swan"])

            # Initialize inference engine
            inference_config = InferenceConfig(
                device=str(device),
                use_amp=True,
            )
            inference_engine = InferenceEngine(
                encoder=encoder,
                policy=policy,
                world_model=world_model,
                black_swan=black_swan,
                config=inference_config,
            )

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return

    # Initialize exchange client (for live/paper modes)
    if args.mode in ["paper", "live"]:
        exchange = create_exchange_client(
            exchange=ExchangeType.BINANCE,
            api_key=config.get("exchange", {}).get("api_key", ""),
            api_secret=config.get("exchange", {}).get("api_secret", ""),
            testnet=args.mode == "paper",
        )
        await exchange.connect()

        # Initialize executor
        risk_controller = RiskController()
        executor = TradingExecutor(
            exchange_client=exchange,
            risk_controller=risk_controller,
            config=ExecutionConfig(mode=args.mode),
        )

    # Main trading loop
    logger.info("Starting trading loop")

    try:
        while orchestrator.state.value == "running":
            # Check drift
            drift_results = drift_detector.detect_all()
            for drift_type, result in drift_results.items():
                if result.detected:
                    alert_manager.create_alert(
                        level=alert_manager.AlertLevel.WARNING,
                        title=f"Drift Detected: {drift_type.value}",
                        message=f"Score: {result.score:.3f}",
                        source="drift_detector",
                    )

            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        logger.info("Trading loop cancelled")

    finally:
        # Cleanup
        if args.mode in ["paper", "live"]:
            await exchange.disconnect()

        dashboard.stop()
        orchestrator.stop()

    logger.info("CryptoAI Trading System stopped")


async def run_backtest(config: dict, args) -> None:
    """Run backtesting."""
    from cryptoai.backtesting import BacktestEngine, BacktestConfig

    logger.info("Starting backtest")

    backtest_config = BacktestConfig(
        initial_capital=config.get("backtest", {}).get("initial_capital", 100000),
        assets=args.assets,
    )

    engine = BacktestEngine(backtest_config)

    # Run backtest
    results = engine.run()

    # Print results
    logger.info(f"Backtest Results:")
    logger.info(f"  Total Return: {results.get('total_return', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")


async def run_training(config: dict, args) -> None:
    """Run training."""
    from cryptoai.training import UnifiedTrainer
    from cryptoai.training.ddp import DDPConfig, launch_distributed

    logger.info("Starting training")

    def train_fn(rank, world_size, ddp_config, **kwargs):
        from cryptoai.training.trainer import TrainingConfig

        training_config = TrainingConfig(
            state_dim=config.get("model", {}).get("state_dim", 200),
            action_dim=config.get("model", {}).get("action_dim", 4),
            hidden_dim=config.get("model", {}).get("hidden_dim", 256),
            latent_dim=config.get("model", {}).get("latent_dim", 128),
        )

        trainer = UnifiedTrainer(training_config, ddp_config, rank)

        # Training would use actual data loaders
        logger.info(f"Rank {rank}: Training initialized")

    # Launch distributed training
    ddp_config = DDPConfig(world_size=torch.cuda.device_count())
    launch_distributed(train_fn, world_size=ddp_config.world_size, config=ddp_config)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Load configuration
    config = load_config(args.config)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run based on mode
    if args.mode == "backtest":
        asyncio.run(run_backtest(config, args))
    elif args.mode == "train":
        asyncio.run(run_training(config, args))
    else:
        asyncio.run(run_trading(config, args))


if __name__ == "__main__":
    main()
