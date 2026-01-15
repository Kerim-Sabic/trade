"""Pytest configuration and fixtures for CryptoAI tests.

Windows 11 Compatible.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from datetime import datetime, timedelta

import pytest
import numpy as np
import torch

# Ensure we're using CPU for tests (Windows CI doesn't have GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CRYPTOAI_TEST_MODE"] = "true"


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the device for tensor operations."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def seed() -> int:
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_seed(seed: int):
    """Set random seeds before each test."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_market_data() -> dict:
    """Generate sample market data for testing."""
    n_samples = 1000
    np.random.seed(42)

    # Generate realistic-ish price data
    returns = np.random.normal(0.0001, 0.002, n_samples)
    prices = 50000 * np.exp(np.cumsum(returns))

    return {
        "BTCUSDT": prices.astype(np.float32),
        "ETHUSDT": (3000 * np.exp(np.cumsum(returns * 1.2))).astype(np.float32),
    }


@pytest.fixture(scope="session")
def sample_state_tensor(device: torch.device) -> torch.Tensor:
    """Generate sample state tensor for encoder tests."""
    batch_size = 4
    seq_len = 100
    feature_dim = 200

    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, feature_dim, device=device)


@pytest.fixture(scope="session")
def sample_action_tensor(device: torch.device) -> torch.Tensor:
    """Generate sample action tensor for policy tests."""
    batch_size = 4
    action_dim = 4

    torch.manual_seed(42)
    return torch.randn(batch_size, action_dim, device=device)


@pytest.fixture(scope="session")
def backtest_config():
    """Create backtest configuration for tests."""
    from cryptoai.backtesting.engine import BacktestConfig

    return BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000.0,
    )


@pytest.fixture(scope="session")
def default_config() -> dict:
    """Load default configuration."""
    from cryptoai.utils.config import load_config

    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        return load_config(str(config_path))
    return {}


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "windows: marks tests specific to Windows")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    skip_gpu = pytest.mark.skip(reason="No GPU available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
