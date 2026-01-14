"""Command-line interface for CryptoAI Trading System."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from loguru import logger

app = typer.Typer(
    name="cryptoai",
    help="CryptoAI - Autonomous Crypto Trading Intelligence",
    add_completion=True,
)

console = Console()


@app.command()
def run(
    mode: str = typer.Option(
        "shadow",
        "--mode", "-m",
        help="Execution mode: shadow, paper, live, backtest, train",
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Path to model checkpoint",
    ),
    assets: Optional[str] = typer.Option(
        "BTCUSDT",
        "--assets", "-a",
        help="Comma-separated list of assets to trade",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    ),
):
    """Run the CryptoAI trading system."""
    from cryptoai.main import main as run_main

    # Build sys.argv for main
    argv = [
        "cryptoai",
        "--mode", mode,
        "--config", config,
        "--log-level", log_level,
    ]

    if model:
        argv.extend(["--model", model])

    if assets:
        argv.extend(["--assets"] + assets.split(","))

    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = argv

    try:
        run_main()
    finally:
        sys.argv = old_argv


@app.command()
def train(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    data_dir: str = typer.Option(
        "data",
        "--data-dir", "-d",
        help="Data directory",
    ),
    synthetic: bool = typer.Option(
        False,
        "--synthetic", "-s",
        help="Generate synthetic data for testing",
    ),
    gpus: int = typer.Option(
        2,
        "--gpus", "-g",
        help="Number of GPUs to use",
    ),
):
    """Run distributed training."""
    import os
    import torch
    from cryptoai.utils.logging import setup_logging
    from cryptoai.utils.config import load_config
    from cryptoai.training import UnifiedTrainer
    from cryptoai.training.ddp import DDPConfig, launch_distributed
    from cryptoai.training.trainer import TrainingConfig
    from cryptoai.training.data_loader import (
        MarketDataLoader,
        DataConfig,
        create_synthetic_data,
    )

    setup_logging(level="INFO")

    console.print("[bold green]Starting CryptoAI Training[/bold green]")
    console.print(f"  Config: {config}")
    console.print(f"  Data directory: {data_dir}")
    console.print(f"  GPUs: {gpus}")
    console.print(f"  Synthetic data: {synthetic}")

    if not torch.cuda.is_available():
        console.print("[yellow]Warning: CUDA not available, using CPU[/yellow]")
        gpus = 0

    # Load configuration
    cfg = load_config(config)

    # Generate synthetic data if requested
    if synthetic:
        console.print("[cyan]Generating synthetic training data...[/cyan]")
        os.makedirs(data_dir, exist_ok=True)
        create_synthetic_data(
            os.path.join(data_dir, "train.h5"),
            n_samples=10000,
        )
        create_synthetic_data(
            os.path.join(data_dir, "val.h5"),
            n_samples=2000,
        )

    # Data configuration
    data_config = DataConfig(
        data_dir=data_dir,
        train_files=[os.path.join(data_dir, "train.h5")],
        val_files=[os.path.join(data_dir, "val.h5")],
        batch_size=cfg.get("training", {}).get("batch_size", 64),
    )

    # DDP configuration
    world_size = min(gpus, torch.cuda.device_count()) if gpus > 0 and torch.cuda.is_available() else 1

    def train_worker(rank: int, ws: int, ddp_config: DDPConfig, **kwargs):
        """Training worker function."""
        # Training configuration
        training_config = TrainingConfig(
            state_dim=cfg.get("model", {}).get("state_dim", 200),
            action_dim=cfg.get("model", {}).get("action_dim", 4),
            hidden_dim=cfg.get("model", {}).get("hidden_dim", 256),
            latent_dim=cfg.get("model", {}).get("latent_dim", 128),
            batch_size=cfg.get("training", {}).get("batch_size", 64),
            encoder_pretrain_epochs=cfg.get("training", {}).get("encoder_epochs", 10),
            world_model_epochs=cfg.get("training", {}).get("world_model_epochs", 50),
            policy_epochs=cfg.get("training", {}).get("policy_epochs", 100),
        )

        trainer = UnifiedTrainer(training_config, ddp_config, rank)
        data_loader = MarketDataLoader(kwargs["data_config"])
        train_loader = data_loader.create_train_loader(rank, ws)
        val_loader = data_loader.create_val_loader(rank, ws)

        logger.info(f"Rank {rank}: Starting full training pipeline")
        history = trainer.full_training(train_loader, val_loader)

        if rank == 0:
            logger.info(f"Training complete. History keys: {list(history.keys())}")

    ddp_config = DDPConfig(world_size=world_size)

    console.print(f"[cyan]Starting distributed training on {world_size} GPU(s)[/cyan]")

    launch_distributed(
        train_worker,
        world_size=world_size,
        config=ddp_config,
        data_config=data_config,
    )


@app.command()
def backtest(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    assets: str = typer.Option(
        "BTCUSDT",
        "--assets", "-a",
        help="Comma-separated list of assets",
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD)",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD)",
    ),
    capital: float = typer.Option(
        100000.0,
        "--capital",
        help="Initial capital",
    ),
):
    """Run backtesting simulation."""
    from datetime import datetime
    from cryptoai.utils.logging import setup_logging
    from cryptoai.utils.config import load_config
    from cryptoai.backtesting import BacktestEngine
    from cryptoai.backtesting.engine import BacktestConfig

    setup_logging(level="INFO")

    console.print("[bold green]Starting CryptoAI Backtest[/bold green]")

    # Parse dates
    start_date = datetime.strptime(start, "%Y-%m-%d") if start else datetime(2023, 1, 1)
    end_date = datetime.strptime(end, "%Y-%m-%d") if end else datetime(2024, 1, 1)

    console.print(f"  Assets: {assets}")
    console.print(f"  Period: {start_date.date()} to {end_date.date()}")
    console.print(f"  Initial capital: ${capital:,.2f}")

    # Note: Would need real market data
    console.print("[yellow]Note: Backtest requires market data files. See README for data setup.[/yellow]")


@app.command()
def status():
    """Show system status and health check."""
    import torch

    console.print("[bold]CryptoAI System Status[/bold]\n")

    # Version info
    from cryptoai import __version__
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("CryptoAI Version", __version__)
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            table.add_row(f"  GPU {i}", torch.cuda.get_device_name(i))

    console.print(table)

    # Check for config file
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        console.print("\n[green]Configuration file found[/green]")
    else:
        console.print("\n[yellow]Warning: configs/default.yaml not found[/yellow]")

    # Check for data directory
    data_path = Path("data")
    if data_path.exists():
        console.print("[green]Data directory found[/green]")
    else:
        console.print("[yellow]Warning: data/ directory not found[/yellow]")


@app.command()
def dashboard(
    port: int = typer.Option(
        8081,
        "--port", "-p",
        help="Dashboard port",
    ),
):
    """Start the monitoring dashboard."""
    from cryptoai.monitoring import MetricsCollector, DriftDetector, AlertManager
    from cryptoai.monitoring.dashboard import DashboardServer, DashboardConfig
    from cryptoai.monitoring.drift_detector import DriftConfig

    console.print(f"[bold green]Starting CryptoAI Dashboard on port {port}[/bold green]")

    metrics = MetricsCollector()
    drift_detector = DriftDetector(DriftConfig())
    alert_manager = AlertManager()

    dashboard_config = DashboardConfig(port=port)
    server = DashboardServer(
        metrics=metrics,
        drift_detector=drift_detector,
        alert_manager=alert_manager,
        config=dashboard_config,
    )

    server.start()

    console.print(f"Dashboard running at http://localhost:{port}")
    console.print("Press Ctrl+C to stop")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        console.print("\nDashboard stopped")


@app.command()
def version():
    """Show version information."""
    from cryptoai import __version__
    console.print(f"CryptoAI version {__version__}")


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
