"""Main Black Swan detection system."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cryptoai.black_swan.tail_risk import TailRiskEstimator
from cryptoai.black_swan.anomaly import AnomalyDetector, VariationalAnomalyDetector
from cryptoai.black_swan.cascade import CascadeDetector


@dataclass
class BlackSwanAlert:
    """Alert from black swan detection system."""

    timestamp: datetime
    alert_type: str  # crash, liquidity_freeze, cascade, regime_break
    probability: float
    severity: str  # low, medium, high, critical
    horizon_minutes: int
    recommended_action: str
    details: Dict[str, float]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type,
            "probability": self.probability,
            "severity": self.severity,
            "horizon_minutes": self.horizon_minutes,
            "recommended_action": self.recommended_action,
            "details": self.details,
        }


class BlackSwanDetector(nn.Module):
    """
    Main Black Swan Intelligence Layer.

    This layer NEVER trades - it ONLY controls risk.

    Monitors:
    - Crash probability (1h / 6h horizons)
    - Liquidity freeze probability
    - Liquidation cascade probability
    - Regime break score
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
        num_horizons: int = 2,  # 1h and 6h
        crash_threshold: float = 0.3,
        liquidity_threshold: float = 0.4,
        cascade_threshold: float = 0.35,
        regime_break_threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.crash_threshold = crash_threshold
        self.liquidity_threshold = liquidity_threshold
        self.cascade_threshold = cascade_threshold
        self.regime_break_threshold = regime_break_threshold

        # Horizons in minutes
        self.horizons = [60, 360]  # 1h and 6h

        # Sub-detectors
        self.tail_risk = TailRiskEstimator(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_horizons=num_horizons,
        )

        self.anomaly_detector = VariationalAnomalyDetector(
            state_dim=state_dim,
            latent_dim=64,
            hidden_dim=hidden_dim,
        )

        self.cascade_detector = CascadeDetector(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        # Combined assessment network
        self.risk_assessor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Output heads
        self.crash_head = nn.Linear(hidden_dim // 2, len(self.horizons))
        self.liquidity_head = nn.Linear(hidden_dim // 2, 1)
        self.regime_break_head = nn.Linear(hidden_dim // 2, 1)

        # Override authority flag
        self.risk_override_authority = True

    def forward(
        self,
        state: torch.Tensor,
        orderbook_state: Optional[torch.Tensor] = None,
        derivatives_state: Optional[torch.Tensor] = None,
        onchain_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Unified market state
            orderbook_state: Order book features (for liquidity analysis)
            derivatives_state: Derivatives features (for cascade detection)
            onchain_state: On-chain features (for panic flow detection)

        Returns:
            Dict with all risk probabilities
        """
        # Main risk assessment
        risk_features = self.risk_assessor(state)

        # Crash probability for each horizon
        crash_probs = torch.sigmoid(self.crash_head(risk_features))

        # Liquidity freeze probability
        liquidity_prob = torch.sigmoid(self.liquidity_head(risk_features))

        # Regime break probability
        regime_break = torch.sigmoid(self.regime_break_head(risk_features))

        # Anomaly score
        anomaly_output = self.anomaly_detector(state)
        anomaly_score = anomaly_output["anomaly_score"]

        # Cascade probability (if derivatives state available)
        if derivatives_state is not None:
            cascade_output = self.cascade_detector(derivatives_state)
            cascade_prob = cascade_output["cascade_probability"]
        else:
            cascade_prob = torch.zeros(state.shape[0], device=state.device)

        # Tail risk estimates
        tail_output = self.tail_risk(state)

        # Combined extreme risk indicator
        extreme_risk = torch.max(
            torch.stack([
                crash_probs.max(dim=-1)[0],
                liquidity_prob.squeeze(-1),
                cascade_prob,
                regime_break.squeeze(-1),
            ], dim=-1),
            dim=-1,
        )[0]

        return {
            "crash_prob_1h": crash_probs[:, 0],
            "crash_prob_6h": crash_probs[:, 1],
            "liquidity_freeze_prob": liquidity_prob.squeeze(-1),
            "cascade_prob": cascade_prob,
            "regime_break_prob": regime_break.squeeze(-1),
            "anomaly_score": anomaly_score,
            "tail_risk": tail_output["expected_shortfall"],
            "extreme_risk": extreme_risk,
            "var_95": tail_output["var_95"],
            "var_99": tail_output["var_99"],
        }

    def check_alerts(
        self,
        state: torch.Tensor,
        orderbook_state: Optional[torch.Tensor] = None,
        derivatives_state: Optional[torch.Tensor] = None,
        onchain_state: Optional[torch.Tensor] = None,
    ) -> List[BlackSwanAlert]:
        """
        Check for black swan alerts.

        Returns list of alerts if thresholds exceeded.
        """
        alerts = []
        now = datetime.utcnow()

        with torch.no_grad():
            output = self.forward(
                state, orderbook_state, derivatives_state, onchain_state
            )

        # Check each indicator (take first element for single state)
        crash_1h = output["crash_prob_1h"][0].item()
        crash_6h = output["crash_prob_6h"][0].item()
        liquidity = output["liquidity_freeze_prob"][0].item()
        cascade = output["cascade_prob"][0].item()
        regime_break = output["regime_break_prob"][0].item()

        # Generate alerts
        if crash_1h > self.crash_threshold:
            severity = self._get_severity(crash_1h)
            alerts.append(BlackSwanAlert(
                timestamp=now,
                alert_type="crash",
                probability=crash_1h,
                severity=severity,
                horizon_minutes=60,
                recommended_action=self._get_crash_action(crash_1h),
                details={"var_95": output["var_95"][0].item()},
            ))

        if crash_6h > self.crash_threshold:
            severity = self._get_severity(crash_6h)
            alerts.append(BlackSwanAlert(
                timestamp=now,
                alert_type="crash",
                probability=crash_6h,
                severity=severity,
                horizon_minutes=360,
                recommended_action=self._get_crash_action(crash_6h),
                details={"var_95": output["var_95"][0].item()},
            ))

        if liquidity > self.liquidity_threshold:
            severity = self._get_severity(liquidity)
            alerts.append(BlackSwanAlert(
                timestamp=now,
                alert_type="liquidity_freeze",
                probability=liquidity,
                severity=severity,
                horizon_minutes=60,
                recommended_action="reduce_position_size",
                details={},
            ))

        if cascade > self.cascade_threshold:
            severity = self._get_severity(cascade)
            alerts.append(BlackSwanAlert(
                timestamp=now,
                alert_type="cascade",
                probability=cascade,
                severity=severity,
                horizon_minutes=30,
                recommended_action="flatten_leveraged_positions",
                details={},
            ))

        if regime_break > self.regime_break_threshold:
            severity = self._get_severity(regime_break)
            alerts.append(BlackSwanAlert(
                timestamp=now,
                alert_type="regime_break",
                probability=regime_break,
                severity=severity,
                horizon_minutes=240,
                recommended_action="pause_strategy",
                details={},
            ))

        return alerts

    def _get_severity(self, prob: float) -> str:
        """Get severity level from probability."""
        if prob > 0.8:
            return "critical"
        elif prob > 0.6:
            return "high"
        elif prob > 0.4:
            return "medium"
        return "low"

    def _get_crash_action(self, prob: float) -> str:
        """Get recommended action for crash probability."""
        if prob > 0.8:
            return "emergency_flatten"
        elif prob > 0.6:
            return "reduce_exposure_50pct"
        elif prob > 0.4:
            return "reduce_exposure_25pct"
        return "increase_monitoring"

    def get_risk_multiplier(
        self,
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get position size multiplier based on risk.

        Returns value 0-1 that should multiply position sizes.
        High risk -> low multiplier.
        """
        extreme_risk = output["extreme_risk"]

        # Sigmoid-based reduction
        # risk=0.5 -> mult=0.5, risk=0.8 -> mult=0.1
        multiplier = 1 - extreme_risk ** 2

        return multiplier.clamp(0.0, 1.0)

    def should_emergency_stop(
        self,
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Determine if emergency stop should be triggered.

        Returns boolean tensor.
        """
        return output["extreme_risk"] > 0.9
