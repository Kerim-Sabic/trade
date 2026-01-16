"""Integration tests for CryptoAI trading system.

Windows 11 Compatible - Tests end-to-end workflows.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path


class TestDataToModelPipeline:
    """Tests for data -> feature -> model pipeline."""

    def test_feature_encoding(self, sample_state_tensor, device):
        """Test feature encoding through encoder."""
        from cryptoai.encoders.base import TransformerBlock

        d_model = sample_state_tensor.shape[-1]
        block = TransformerBlock(d_model=d_model).to(device)

        output = block(sample_state_tensor)

        assert output.shape == sample_state_tensor.shape
        assert not torch.isnan(output).any()

    def test_unified_encoder_forward(self, device):
        """Test unified encoder forward pass."""
        from cryptoai.encoders.unified import UnifiedMarketEncoder

        batch_size = 4
        seq_len = 100
        input_dim = 50

        encoder = UnifiedMarketEncoder(input_dim=input_dim).to(device)

        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        # Forward pass
        output = encoder(x)

        assert "state" in output
        assert not torch.isnan(output["state"]).any()

    def test_regime_encoder(self, device):
        """Test regime encoder."""
        from cryptoai.encoders.regime import RegimeEncoder

        batch_size = 4
        seq_len = 100
        input_dim = 64

        encoder = RegimeEncoder(input_dim=input_dim).to(device)

        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        output = encoder(x)

        assert "regime_probs" in output
        assert output["regime_probs"].shape[1] == encoder.num_regimes


class TestDecisionEnginePipeline:
    """Tests for decision engine pipeline."""

    def test_action_space_creation(self):
        """Test action space creation."""
        from cryptoai.decision_engine.action_space import (
            CryptoActionSpace,
            ContinuousActionSpace,
        )

        # Test discrete
        discrete_space = CryptoActionSpace(position_granularity=0.1)
        assert len(discrete_space.actions) > 0

        # Test continuous
        continuous_space = ContinuousActionSpace()
        assert continuous_space.shape == (4,)

    def test_policy_forward(self, device):
        """Test policy network forward pass."""
        from cryptoai.decision_engine.policy import ConservativePolicy

        state_dim = 256
        action_dim = 4
        batch_size = 4

        policy = ConservativePolicy(state_dim=state_dim, action_dim=action_dim).to(device)

        state = torch.randn(batch_size, state_dim, device=device)

        output = policy(state)

        assert "action" in output
        assert "log_prob" in output
        assert output["action"].shape == (batch_size, action_dim)

    def test_reward_computation(self):
        """Test reward function computation."""
        from cryptoai.decision_engine.reward import RiskAwareRewardFunction

        reward_fn = RiskAwareRewardFunction()

        # Simulate some returns
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])

        reward = reward_fn.compute_step_reward(
            pnl=0.01,
            position_return=0.02,
            drawdown=0.05,
            holding_time=10,
        )

        # Reward should be finite
        assert np.isfinite(reward)


class TestRiskEnginePipeline:
    """Tests for risk engine pipeline."""

    def test_risk_controller(self):
        """Test risk controller."""
        from cryptoai.risk_engine import RiskController

        controller = RiskController()
        assert controller is not None

    def test_position_sizer(self):
        """Test position sizing."""
        from cryptoai.risk_engine import PositionSizer

        sizer = PositionSizer()

        # Test basic sizing
        capital = 100000.0
        risk_per_trade = 0.02

        # Should compute valid size
        assert sizer is not None

    def test_kill_switch(self):
        """Test kill switch functionality."""
        from cryptoai.risk_engine import KillSwitch

        kill_switch = KillSwitch()

        # Should be triggerable
        assert hasattr(kill_switch, "trigger") or hasattr(kill_switch, "is_triggered")


class TestBacktestingPipeline:
    """Tests for backtesting pipeline."""

    def test_backtest_config_creation(self):
        """Test backtest configuration creation."""
        from cryptoai.backtesting.engine import BacktestConfig

        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            initial_capital=100000.0,
        )

        assert config.initial_capital == 100000.0
        assert config.maker_fee is not None
        assert config.taker_fee is not None

    def test_backtest_engine_run(self, backtest_config, sample_market_data):
        """Test backtest engine execution."""
        from cryptoai.backtesting import BacktestEngine

        engine = BacktestEngine(backtest_config, sample_market_data)

        # Run a few steps
        for i in range(10):
            prices = {"BTCUSDT": sample_market_data["BTCUSDT"][i]}
            engine.step(prices)

        assert engine.current_step == 10
        assert engine.state.equity > 0

    def test_walk_forward_validation(self):
        """Test walk-forward validation."""
        from cryptoai.backtesting.validation import (
            ValidationConfig,
            WalkForwardValidator,
        )

        config = ValidationConfig(
            train_window_days=30,
            test_window_days=10,
            step_days=5,
        )

        validator = WalkForwardValidator(config)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)

        windows = list(validator.generate_windows(start, end))

        assert len(windows) > 0
        for window in windows:
            assert window.train_end < window.test_start  # No overlap


class TestTrainingPipeline:
    """Tests for training pipeline."""

    def test_data_loader_creation(self, temp_dir):
        """Test data loader creation."""
        from cryptoai.training.data_loader import (
            DataConfig,
            ExperienceDataset,
            create_synthetic_data,
        )

        # Create synthetic data
        data_path = str(temp_dir / "test_data.h5")
        create_synthetic_data(
            output_path=data_path,
            n_samples=100,
            sequence_length=50,
            feature_dim=100,
            action_dim=4,
        )

        assert Path(data_path).exists()

    def test_experience_replay(self):
        """Test experience replay buffer."""
        from cryptoai.training.experience_replay import (
            ExperienceReplay,
            Experience,
        )

        buffer = ExperienceReplay(
            capacity=1000,
            state_shape=(50, 100),
            action_dim=4,
        )

        # Add experiences
        for _ in range(100):
            exp = Experience(
                state=np.random.randn(50, 100).astype(np.float32),
                action=np.random.randn(4).astype(np.float32),
                reward=np.random.randn(),
                next_state=np.random.randn(50, 100).astype(np.float32),
                done=False,
            )
            buffer.add(exp)

        assert len(buffer) == 100

        # Sample batch
        batch = buffer.sample(32)
        assert batch["states"].shape[0] == 32

    def test_ddp_trainer_creation(self, device):
        """Test DDP trainer creation."""
        from cryptoai.training.ddp import DDPConfig, DDPTrainer
        import torch.nn as nn

        model = nn.Linear(10, 10)
        config = DDPConfig(world_size=1)

        trainer = DDPTrainer(model, config, rank=0)

        assert trainer.device is not None


class TestBlackSwanDetection:
    """Tests for black swan detection."""

    def test_black_swan_detector(self, device):
        """Test black swan detector."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        state_dim = 256
        batch_size = 4

        detector = BlackSwanDetector(state_dim=state_dim).to(device)

        state = torch.randn(batch_size, state_dim, device=device)

        output = detector(state)

        assert "risk_score" in output
        assert output["risk_score"].shape == (batch_size,)

    def test_anomaly_detection(self, device):
        """Test anomaly detection."""
        from cryptoai.black_swan.anomaly import AnomalyDetector

        state_dim = 256
        batch_size = 4

        detector = AnomalyDetector(state_dim=state_dim).to(device)

        state = torch.randn(batch_size, state_dim, device=device)

        output = detector(state)

        assert "anomaly_score" in output
        assert "is_anomaly" in output

    def test_tail_risk_estimation(self, device):
        """Test tail risk estimation."""
        from cryptoai.black_swan.tail_risk import TailRiskEstimator

        state_dim = 256
        batch_size = 4

        estimator = TailRiskEstimator(state_dim=state_dim).to(device)

        state = torch.randn(batch_size, state_dim, device=device)

        output = estimator(state)

        assert "var_95" in output
        assert "expected_shortfall" in output


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.integration
    def test_full_inference_pipeline(self, device):
        """Test complete inference pipeline."""
        from cryptoai.encoders.unified import UnifiedMarketEncoder
        from cryptoai.decision_engine.policy import ConservativePolicy

        # Setup
        input_dim = 50
        state_dim = 256
        action_dim = 4
        batch_size = 4
        seq_len = 100

        encoder = UnifiedMarketEncoder(input_dim=input_dim, output_dim=state_dim).to(device)
        policy = ConservativePolicy(state_dim=state_dim, action_dim=action_dim).to(device)

        # Input data
        market_data = torch.randn(batch_size, seq_len, input_dim, device=device)

        # Forward pass
        encoded = encoder(market_data)
        state = encoded["state"]

        # Get last timestep for policy
        if len(state.shape) == 3:
            state = state[:, -1, :]

        action_output = policy(state)

        # Verify outputs
        assert action_output["action"].shape == (batch_size, action_dim)
        assert not torch.isnan(action_output["action"]).any()

    @pytest.mark.integration
    def test_backtest_with_simple_strategy(self, sample_market_data):
        """Test backtest with a simple strategy."""
        from cryptoai.backtesting.engine import BacktestConfig, BacktestEngine
        from datetime import datetime, timedelta

        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=100000.0,
        )

        engine = BacktestEngine(config, sample_market_data)

        # Run backtest with simple price following
        prices_btc = sample_market_data["BTCUSDT"]
        prev_price = prices_btc[0]

        for i in range(min(500, len(prices_btc))):
            current_price = prices_btc[i]
            engine.step({"BTCUSDT": current_price})
            prev_price = current_price

        # Verify results
        assert engine.state.equity > 0
        assert engine.current_step == min(500, len(prices_btc))
        assert engine.state.drawdown >= 0
        assert engine.state.drawdown <= 1
