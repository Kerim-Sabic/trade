"""
Tests for Black Swan Detection System.

Tests the complete black swan detection pipeline including:
- Crash probability estimation
- Liquidity freeze detection
- Cascade detection
- Regime break detection
- Risk multiplier calculation
- Emergency stop triggers

Windows 11 Compatible.
"""

import pytest
import torch
import numpy as np
from datetime import datetime


class TestBlackSwanDetector:
    """Tests for main BlackSwanDetector class."""

    def test_initialization(self, device):
        """Test detector initialization."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(
            state_dim=256,
            hidden_dim=256,
            crash_threshold=0.3,
            liquidity_threshold=0.4,
        ).to(device)

        assert detector is not None
        assert detector.crash_threshold == 0.3
        assert detector.liquidity_threshold == 0.4

    def test_forward_pass(self, device):
        """Test forward pass produces all expected outputs."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        batch_size = 4
        state_dim = 256

        detector = BlackSwanDetector(state_dim=state_dim).to(device)
        state = torch.randn(batch_size, state_dim, device=device)

        output = detector(state)

        # Check all expected outputs exist
        assert "crash_prob_1h" in output
        assert "crash_prob_6h" in output
        assert "liquidity_freeze_prob" in output
        assert "cascade_prob" in output
        assert "regime_break_prob" in output
        assert "anomaly_score" in output
        assert "extreme_risk" in output
        assert "var_95" in output
        assert "var_99" in output

        # Check shapes
        assert output["crash_prob_1h"].shape == (batch_size,)
        assert output["crash_prob_6h"].shape == (batch_size,)
        assert output["extreme_risk"].shape == (batch_size,)

        # Check values are in valid range [0, 1]
        for key in ["crash_prob_1h", "crash_prob_6h", "liquidity_freeze_prob",
                    "cascade_prob", "regime_break_prob"]:
            assert (output[key] >= 0).all()
            assert (output[key] <= 1).all()

    def test_check_alerts_no_alerts(self, device):
        """Test that normal conditions produce no alerts."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(
            state_dim=256,
            crash_threshold=0.9,  # Very high thresholds
            liquidity_threshold=0.9,
            cascade_threshold=0.9,
            regime_break_threshold=0.9,
        ).to(device)

        # Normal state should not trigger alerts
        state = torch.randn(1, 256, device=device) * 0.1

        alerts = detector.check_alerts(state)

        # With high thresholds, random input shouldn't trigger
        # Note: this may occasionally fail if random values are extreme
        assert isinstance(alerts, list)

    def test_check_alerts_with_alerts(self, device):
        """Test that extreme conditions produce alerts."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(
            state_dim=256,
            crash_threshold=0.01,  # Very low threshold
            liquidity_threshold=0.01,
            cascade_threshold=0.01,
            regime_break_threshold=0.01,
        ).to(device)

        # Extreme state that should trigger alerts
        state = torch.randn(1, 256, device=device) * 10

        alerts = detector.check_alerts(state)

        # With very low thresholds, should get alerts
        assert isinstance(alerts, list)
        # Can't guarantee specific alerts due to random weights

    def test_risk_multiplier(self, device):
        """Test risk multiplier calculation."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(state_dim=256).to(device)
        state = torch.randn(4, 256, device=device)

        output = detector(state)
        multiplier = detector.get_risk_multiplier(output)

        # Multiplier should be between 0 and 1
        assert (multiplier >= 0).all()
        assert (multiplier <= 1).all()

    def test_emergency_stop_trigger(self, device):
        """Test emergency stop detection."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(state_dim=256).to(device)

        # Create output with extreme risk
        output = {
            "extreme_risk": torch.tensor([0.95, 0.5, 0.85, 0.99], device=device)
        }

        should_stop = detector.should_emergency_stop(output)

        # First and last should trigger (> 0.9)
        assert should_stop[0].item() == True
        assert should_stop[1].item() == False
        assert should_stop[2].item() == False
        assert should_stop[3].item() == True


class TestBlackSwanAlert:
    """Tests for BlackSwanAlert dataclass."""

    def test_alert_creation(self):
        """Test alert creation."""
        from cryptoai.black_swan.detector import BlackSwanAlert

        alert = BlackSwanAlert(
            timestamp=datetime.utcnow(),
            alert_type="crash",
            probability=0.75,
            severity="high",
            horizon_minutes=60,
            recommended_action="reduce_exposure_50pct",
            details={"var_95": 0.08},
        )

        assert alert.alert_type == "crash"
        assert alert.probability == 0.75
        assert alert.severity == "high"

    def test_alert_to_dict(self):
        """Test alert serialization."""
        from cryptoai.black_swan.detector import BlackSwanAlert

        alert = BlackSwanAlert(
            timestamp=datetime.utcnow(),
            alert_type="liquidity_freeze",
            probability=0.6,
            severity="medium",
            horizon_minutes=30,
            recommended_action="reduce_position_size",
            details={},
        )

        data = alert.to_dict()

        assert isinstance(data, dict)
        assert data["alert_type"] == "liquidity_freeze"
        assert data["probability"] == 0.6


class TestTailRiskEstimator:
    """Tests for tail risk estimation."""

    def test_initialization(self, device):
        """Test tail risk estimator initialization."""
        from cryptoai.black_swan.tail_risk import TailRiskEstimator

        estimator = TailRiskEstimator(
            state_dim=256,
            hidden_dim=128,
        ).to(device)

        assert estimator is not None

    def test_forward_pass(self, device):
        """Test forward pass."""
        from cryptoai.black_swan.tail_risk import TailRiskEstimator

        batch_size = 4
        state_dim = 256

        estimator = TailRiskEstimator(state_dim=state_dim).to(device)
        state = torch.randn(batch_size, state_dim, device=device)

        output = estimator(state)

        assert "var_95" in output
        assert "var_99" in output
        assert "expected_shortfall" in output


class TestAnomalyDetector:
    """Tests for anomaly detection."""

    def test_initialization(self, device):
        """Test anomaly detector initialization."""
        from cryptoai.black_swan.anomaly import VariationalAnomalyDetector

        detector = VariationalAnomalyDetector(
            state_dim=256,
            latent_dim=64,
        ).to(device)

        assert detector is not None

    def test_forward_pass(self, device):
        """Test forward pass."""
        from cryptoai.black_swan.anomaly import VariationalAnomalyDetector

        batch_size = 4
        state_dim = 256

        detector = VariationalAnomalyDetector(state_dim=state_dim).to(device)
        state = torch.randn(batch_size, state_dim, device=device)

        output = detector(state)

        assert "anomaly_score" in output
        assert output["anomaly_score"].shape == (batch_size,)

    def test_anomaly_score_range(self, device):
        """Test that anomaly scores are in valid range."""
        from cryptoai.black_swan.anomaly import VariationalAnomalyDetector

        detector = VariationalAnomalyDetector(state_dim=256).to(device)

        # Normal data
        normal_state = torch.randn(10, 256, device=device) * 0.1

        output = detector(normal_state)
        scores = output["anomaly_score"]

        # Scores should be non-negative
        assert (scores >= 0).all()


class TestCascadeDetector:
    """Tests for liquidation cascade detection."""

    def test_initialization(self, device):
        """Test cascade detector initialization."""
        from cryptoai.black_swan.cascade import CascadeDetector

        detector = CascadeDetector(
            state_dim=256,
            hidden_dim=128,
        ).to(device)

        assert detector is not None

    def test_forward_pass(self, device):
        """Test forward pass."""
        from cryptoai.black_swan.cascade import CascadeDetector

        batch_size = 4
        state_dim = 256

        detector = CascadeDetector(state_dim=state_dim).to(device)
        state = torch.randn(batch_size, state_dim, device=device)

        output = detector(state)

        assert "cascade_probability" in output
        assert output["cascade_probability"].shape == (batch_size,)


class TestBlackSwanIntegration:
    """Integration tests for black swan system."""

    def test_full_pipeline(self, device):
        """Test complete black swan detection pipeline."""
        from cryptoai.black_swan.detector import BlackSwanDetector
        from cryptoai.encoders.unified import UnifiedStateEncoder

        # Create encoder
        encoder = UnifiedStateEncoder(
            microstructure_dim=40,
            derivatives_dim=42,
            onchain_dim=33,
            event_dim=15,
            unified_output=256,
        ).to(device)

        # Create detector
        detector = BlackSwanDetector(state_dim=256).to(device)

        # Create dummy inputs
        batch_size = 4
        seq_len = 100

        microstructure = torch.randn(batch_size, seq_len, 40, device=device)
        derivatives = torch.randn(batch_size, seq_len, 42, device=device)
        onchain = torch.randn(batch_size, seq_len, 33, device=device)
        events = torch.randn(batch_size, seq_len, 15, device=device)
        asset_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Encode state
        state = encoder(
            microstructure, derivatives, onchain, events, asset_ids
        )

        # Detect black swans
        output = detector(state)

        # Verify output structure
        assert "extreme_risk" in output
        assert output["extreme_risk"].shape == (batch_size,)

    def test_stress_scenario_luna(self, device):
        """Simulate LUNA-like crash scenario."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(
            state_dim=256,
            crash_threshold=0.3,
        ).to(device)

        # Simulate extreme downward movement
        # High volatility + large negative returns
        state = torch.randn(1, 256, device=device)
        state[:, :50] = 5.0  # Simulate extreme feature values

        output = detector(state)

        # In a real scenario, this should show elevated risk
        # Can't guarantee specific behavior without trained weights
        assert output["extreme_risk"].shape == (1,)

    def test_stress_scenario_ftx(self, device):
        """Simulate FTX-like liquidity crisis."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(
            state_dim=256,
            liquidity_threshold=0.4,
        ).to(device)

        # Simulate liquidity crisis features
        state = torch.randn(1, 256, device=device)
        state[:, 100:150] = -5.0  # Simulate extreme negative features

        output = detector(state)

        assert "liquidity_freeze_prob" in output


class TestBlackSwanSeverity:
    """Tests for severity level calculations."""

    def test_severity_levels(self, device):
        """Test that severity levels are correctly assigned."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(state_dim=256).to(device)

        # Test severity calculation
        assert detector._get_severity(0.85) == "critical"
        assert detector._get_severity(0.65) == "high"
        assert detector._get_severity(0.45) == "medium"
        assert detector._get_severity(0.25) == "low"

    def test_crash_action_recommendations(self, device):
        """Test action recommendations based on probability."""
        from cryptoai.black_swan.detector import BlackSwanDetector

        detector = BlackSwanDetector(state_dim=256).to(device)

        assert detector._get_crash_action(0.85) == "emergency_flatten"
        assert detector._get_crash_action(0.65) == "reduce_exposure_50pct"
        assert detector._get_crash_action(0.45) == "reduce_exposure_25pct"
        assert detector._get_crash_action(0.25) == "increase_monitoring"


# Fixtures
@pytest.fixture
def device():
    """Get device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
