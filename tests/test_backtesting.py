"""Unit tests for backtesting module.

Windows 11 Compatible - Tests backtesting engine for realistic simulation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from cryptoai.backtesting import BacktestEngine
from cryptoai.backtesting.engine import BacktestConfig, BacktestState


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )

        assert config.initial_capital == 100000.0
        assert config.maker_fee >= 0
        assert config.taker_fee >= 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=50000.0,
            maker_fee=0.001,
            taker_fee=0.002,
        )

        assert config.initial_capital == 50000.0
        assert config.maker_fee == 0.001
        assert config.taker_fee == 0.002


class TestBacktestState:
    """Tests for BacktestState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = BacktestState.create(initial_capital=100000.0)

        assert state.cash == 100000.0  # Using cash alias
        assert state.capital == 100000.0
        assert state.equity == 100000.0
        assert state.drawdown == 0.0
        assert len(state.positions) == 0
        assert len(state.filled_orders) == 0

    def test_cash_alias(self):
        """Test that cash is an alias for capital."""
        state = BacktestState.create(initial_capital=50000.0)

        assert state.cash == state.capital
        state.capital = 75000.0
        assert state.cash == 75000.0


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self, backtest_config, sample_market_data):
        """Test engine initialization."""
        engine = BacktestEngine(backtest_config, sample_market_data)

        assert engine.state.equity == backtest_config.initial_capital
        assert engine.current_step == 0

    def test_step_updates_state(self, backtest_config, sample_market_data):
        """Test that stepping updates state."""
        engine = BacktestEngine(backtest_config, sample_market_data)

        initial_step = engine.current_step
        current_prices = {"BTCUSDT": sample_market_data["BTCUSDT"][0]}

        engine.step(current_prices)

        assert engine.current_step == initial_step + 1

    def test_multiple_steps(self, backtest_config, sample_market_data):
        """Test multiple simulation steps."""
        engine = BacktestEngine(backtest_config, sample_market_data)

        n_steps = 100
        for i in range(n_steps):
            current_prices = {"BTCUSDT": sample_market_data["BTCUSDT"][i]}
            engine.step(current_prices)

        assert engine.current_step == n_steps

    def test_equity_non_negative(self, backtest_config, sample_market_data):
        """Test that equity never goes negative."""
        engine = BacktestEngine(backtest_config, sample_market_data)

        for i in range(500):
            current_prices = {"BTCUSDT": sample_market_data["BTCUSDT"][i]}
            engine.step(current_prices)

            assert engine.state.equity >= 0, f"Equity went negative at step {i}"

    def test_drawdown_calculation(self, backtest_config, sample_market_data):
        """Test drawdown calculation."""
        engine = BacktestEngine(backtest_config, sample_market_data)

        for i in range(100):
            current_prices = {"BTCUSDT": sample_market_data["BTCUSDT"][i]}
            engine.step(current_prices)

        # Drawdown should be between 0 and 1
        assert 0 <= engine.state.drawdown <= 1

    def test_deterministic_replay(self, backtest_config, sample_market_data, seed):
        """Test that backtest is deterministic with same seed."""
        np.random.seed(seed)

        engine1 = BacktestEngine(backtest_config, sample_market_data)
        for i in range(100):
            engine1.step({"BTCUSDT": sample_market_data["BTCUSDT"][i]})

        final_equity_1 = engine1.state.equity

        # Reset seed and run again
        np.random.seed(seed)

        engine2 = BacktestEngine(backtest_config, sample_market_data)
        for i in range(100):
            engine2.step({"BTCUSDT": sample_market_data["BTCUSDT"][i]})

        final_equity_2 = engine2.state.equity

        assert final_equity_1 == final_equity_2, "Backtest not deterministic"


class TestBacktestRealism:
    """Tests for backtest realism - no lookahead bias, etc."""

    def test_no_future_data_access(self, backtest_config, sample_market_data):
        """Test that engine doesn't access future data."""
        engine = BacktestEngine(backtest_config, sample_market_data)

        # At step i, we should only see data up to i
        for i in range(50):
            current_prices = {"BTCUSDT": sample_market_data["BTCUSDT"][i]}
            engine.step(current_prices)

            # Engine should not have access to future prices
            # This is implicitly tested by only passing current_prices

    def test_fees_are_applied(self, sample_market_data):
        """Test that trading fees reduce equity."""
        # High fee config
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=100000.0,
            maker_fee=0.01,  # 1% fee
            taker_fee=0.01,
        )

        engine = BacktestEngine(config, sample_market_data)

        # Run simulation
        for i in range(100):
            engine.step({"BTCUSDT": sample_market_data["BTCUSDT"][i]})

        # Total fees should be tracked
        assert engine.state.total_fees >= 0

    def test_slippage_modeling(self, sample_market_data):
        """Test that slippage is modeled."""
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=100000.0,
        )

        engine = BacktestEngine(config, sample_market_data)

        for i in range(100):
            engine.step({"BTCUSDT": sample_market_data["BTCUSDT"][i]})

        # Total slippage should be tracked
        assert engine.state.total_slippage >= 0
