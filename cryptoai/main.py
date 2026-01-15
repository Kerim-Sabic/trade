"""Main entry point for the CryptoAI Trading System.

Windows 11 Compatible - handles signal differences between Windows and Unix.
"""

import argparse
import asyncio
import signal
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import torch
import numpy as np
from loguru import logger

from cryptoai.utils.config import load_config
from cryptoai.utils.logging import setup_logging
from cryptoai.utils.device import get_device

# Platform detection
IS_WINDOWS = sys.platform == "win32"


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

    # Track exchange for cleanup
    exchange = None
    inference_engine = None

    # Load models if checkpoint provided
    if args.model:
        try:
            from cryptoai.encoders import UnifiedStateEncoder
            from cryptoai.decision_engine import PolicyNetwork
            from cryptoai.world_model import WorldModel
            from cryptoai.black_swan import BlackSwanDetector

            device = get_device()

            # Load checkpoint
            checkpoint = torch.load(args.model, map_location=device, weights_only=False)

            # Get model dimensions from config
            model_cfg = config.get("model", {})
            hidden_dim = model_cfg.get("hidden_dim", 256)
            latent_dim = model_cfg.get("latent_dim", 128)
            action_dim = model_cfg.get("action_dim", 4)

            # Initialize encoder with correct parameters
            encoder = UnifiedStateEncoder(
                microstructure_dim=model_cfg.get("microstructure_dim", 40),
                derivatives_dim=model_cfg.get("derivatives_dim", 42),
                onchain_dim=model_cfg.get("onchain_dim", 33),
                event_dim=model_cfg.get("event_dim", 15),
                unified_output=latent_dim,
                unified_hidden=hidden_dim,
            ).to(device)

            policy = PolicyNetwork(
                state_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            ).to(device)

            world_model = WorldModel(
                state_dim=latent_dim,
                hidden_dim=hidden_dim,
            ).to(device)

            black_swan = BlackSwanDetector(
                state_dim=latent_dim,
                hidden_dim=hidden_dim,
            ).to(device)

            # Load weights if available in checkpoint
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
            import traceback
            traceback.print_exc()
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
        if exchange is not None and args.mode in ["paper", "live"]:
            await exchange.disconnect()

        dashboard.stop()
        orchestrator.stop()

    logger.info("CryptoAI Trading System stopped")


async def run_backtest(config: dict, args) -> None:
    """Run backtesting."""
    from cryptoai.backtesting import BacktestEngine
    from cryptoai.backtesting.engine import BacktestConfig

    logger.info("Starting backtest")

    # Get backtest configuration
    backtest_cfg = config.get("backtesting", {})

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=backtest_cfg.get("days", 365))

    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=backtest_cfg.get("initial_capital", 100000),
    )

    # Generate synthetic market data for demonstration
    # In production, this would load real historical data
    logger.info("Generating synthetic market data for demonstration...")
    n_periods = min(100000, int((end_date - start_date).total_seconds() / 60))

    market_data = {}
    for asset in args.assets:
        # Generate random walk price data
        returns = np.random.normal(0.0001, 0.001, n_periods)
        prices = 50000 * np.exp(np.cumsum(returns))  # Starting from ~50k
        market_data[asset] = prices

    # Create engine with market data
    engine = BacktestEngine(backtest_config, market_data)

    # Run backtest simulation
    logger.info(f"Running backtest for {len(args.assets)} assets over {n_periods} periods...")

    # Step through simulation
    for i in range(min(10000, n_periods)):
        current_prices = {asset: market_data[asset][i] for asset in args.assets}
        engine.step(current_prices)

    # Get and print results
    logger.info("Backtest Results:")
    logger.info(f"  Final Equity: ${engine.state.equity:,.2f}")
    logger.info(f"  Initial Capital: ${backtest_config.initial_capital:,.2f}")
    total_return = (engine.state.equity / backtest_config.initial_capital) - 1
    logger.info(f"  Total Return: {total_return:.2%}")
    logger.info(f"  Max Drawdown: {engine.state.drawdown:.2%}")
    logger.info(f"  Total Fees: ${engine.state.total_fees:,.2f}")
    logger.info(f"  Total Slippage: ${engine.state.total_slippage:,.2f}")
    logger.info(f"  Total Trades: {len(engine.state.filled_orders)}")


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
        logger.info(f"Rank {rank}: Training initialized")

    # Launch distributed training
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    ddp_config = DDPConfig(world_size=world_size)
    launch_distributed(train_fn, world_size=ddp_config.world_size, config=ddp_config)


def setup_signal_handlers():
    """Setup signal handlers with Windows compatibility.

    Windows only supports SIGINT, SIGTERM, SIGABRT, and SIGFPE.
    SIGTERM doesn't work reliably on Windows, so we primarily rely on SIGINT.
    """
    def signal_handler(signum, frame):
        logger.info(f"Received shutdown signal ({signum})")
        # Emit governance state for Electron app
        print("GOVERNANCE_STATE: STOPPED")
        sys.exit(0)

    # SIGINT works on both Windows and Unix
    signal.signal(signal.SIGINT, signal_handler)

    # SIGTERM is Unix-specific and doesn't work well on Windows
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging with platform-aware output
    setup_logging(level=args.log_level)

    # Log startup info including governance state for Electron app
    logger.info(f"CryptoAI v0.2.0 starting on {'Windows' if IS_WINDOWS else 'Unix'}")
    print("GOVERNANCE_STATE: OPERATIONAL")

    # Load configuration
    config = load_config(args.config)

    # Setup signal handlers (Windows-compatible)
    setup_signal_handlers()

    # Run based on mode
    try:
        if args.mode == "backtest":
            asyncio.run(run_backtest(config, args))
        elif args.mode == "train":
            asyncio.run(run_training(config, args))
        else:
            asyncio.run(run_trading(config, args))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        print("GOVERNANCE_STATE: STOPPED")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        print("GOVERNANCE_STATE: HALTED")
        raise


if __name__ == "__main__":
    main()
