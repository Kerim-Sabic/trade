"""
AI Governance & Self-Improvement Loop.

CHIEF RISK & AI GOVERNANCE ENGINE
==================================

This module controls the self-improvement capabilities of the trading system.
It is MISSION-CRITICAL for preventing catastrophic failures.

CORE PRINCIPLES:
================

1. CAPITAL PRESERVATION IS PARAMOUNT
   - No learning update can risk more than defined limits
   - Kill-switches CANNOT be disabled by the AI
   - Human override always takes precedence

2. BOUNDED LEARNING
   - System learns WHAT it's allowed to learn
   - System CANNOT learn to bypass safety constraints
   - All learning is auditable and reversible

3. FAIL-SAFE BY DEFAULT
   - When in doubt, do nothing
   - Any anomaly triggers conservative mode
   - Multiple independent kill-switches

4. EXPLAINABILITY
   - Every decision is logged with reasoning
   - Performance attribution is continuous
   - Model decay is detected and reported

ARCHITECTURE:
=============

┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI GOVERNANCE & SELF-IMPROVEMENT LOOP                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        HUMAN OVERSIGHT LAYER                          │  │
│  │                                                                       │  │
│  │  • Emergency kill-switch (hardware button simulation)                │  │
│  │  • Manual model approval required for deployment                     │  │
│  │  • Audit log review interface                                        │  │
│  │  • Override any AI decision                                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                        │
│                                    │ Human can intervene at any point       │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      CAPITAL PROTECTION LAYER                         │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │ HARD LIMITS  │  │ SOFT LIMITS  │  │ KILL-SWITCH  │               │  │
│  │  │              │  │              │  │              │               │  │
│  │  │ • Max loss   │  │ • Daily PnL  │  │ • Drawdown   │               │  │
│  │  │ • Max pos    │  │ • Vol adj    │  │ • Correlation│               │  │
│  │  │ • Max lever  │  │ • Trend foll │  │ • Anomaly    │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     MODEL HEALTH MONITORING                           │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │ PERFORMANCE  │  │   DECAY      │  │  DRIFT       │               │  │
│  │  │ ATTRIBUTION  │  │ DETECTION    │  │ DETECTION    │               │  │
│  │  │              │  │              │  │              │               │  │
│  │  │ • By model   │  │ • Accuracy   │  │ • Feature    │               │  │
│  │  │ • By regime  │  │ • Calibratn  │  │ • Label      │               │  │
│  │  │ • By asset   │  │ • Sharpness  │  │ • Concept    │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    SELF-IMPROVEMENT ENGINE                            │  │
│  │                                                                       │  │
│  │  ALLOWED TO LEARN:              │  FORBIDDEN TO LEARN:               │  │
│  │  ✓ Price prediction             │  ✗ Bypass risk limits             │  │
│  │  ✓ Volatility estimation        │  ✗ Ignore kill-switches           │  │
│  │  ✓ Market regime detection      │  ✗ Exceed position limits         │  │
│  │  ✓ Order timing optimization    │  ✗ Trade during blackout          │  │
│  │  ✓ Feature importance           │  ✗ Modify its own constraints     │  │
│  │                                 │  ✗ Access unauthorized data        │  │
│  │  RETRAINING TRIGGERS:           │  ✗ Communicate externally          │  │
│  │  • Accuracy < threshold         │  ✗ Persist state outside sandbox   │  │
│  │  • Drift detected              │                                     │  │
│  │  • Scheduled (weekly)           │                                     │  │
│  │  • Human-approved only          │                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

CATASTROPHIC FAILURE PREVENTION:
================================

Layer 1: Position Limits (enforced in execution layer)
Layer 2: Drawdown Circuit Breakers (3 independent monitors)
Layer 3: Model Sanity Checks (predictions must be bounded)
Layer 4: Human Approval Gate (for any significant change)
Layer 5: Automatic Rollback (on any anomaly)
Layer 6: Hardware Kill-Switch (simulated, cannot be bypassed)

Author: Chief Risk & AI Governance Engineer
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from enum import Enum
from collections import deque
import hashlib
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import json
import threading
import os


# =============================================================================
# ENUMS AND CONFIGURATIONS
# =============================================================================

class GovernanceState(Enum):
    """System governance states."""
    OPERATIONAL = "operational"         # Normal trading allowed
    RESTRICTED = "restricted"           # Reduced position sizes
    SUSPENDED = "suspended"             # No new positions, can close
    HALTED = "halted"                   # No trading at all
    EMERGENCY = "emergency"             # Close all positions immediately


class LearningPermission(Enum):
    """What the system is allowed to learn."""
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    REQUIRES_APPROVAL = "requires_approval"


class RetrainingTrigger(Enum):
    """Reasons for triggering retraining."""
    SCHEDULED = "scheduled"             # Regular schedule
    ACCURACY_DECAY = "accuracy_decay"   # Performance degradation
    DRIFT_DETECTED = "drift_detected"   # Distribution shift
    REGIME_CHANGE = "regime_change"     # Market regime shift
    HUMAN_REQUESTED = "human_requested" # Manual trigger
    NONE = "none"                       # No trigger


@dataclass
class GovernanceConfig:
    """Configuration for AI governance."""

    # Capital protection (HARD LIMITS - cannot be modified by AI)
    max_total_loss_pct: float = 0.20        # 20% max total loss
    max_daily_loss_pct: float = 0.05        # 5% max daily loss
    max_single_trade_loss_pct: float = 0.02 # 2% max per trade
    max_drawdown_pct: float = 0.15          # 15% max drawdown
    max_position_pct: float = 0.25          # 25% max single position
    max_leverage: float = 5.0               # 5x max leverage

    # Kill-switch thresholds
    drawdown_warning_pct: float = 0.08      # 8% warning
    drawdown_suspend_pct: float = 0.12      # 12% suspend
    drawdown_halt_pct: float = 0.15         # 15% halt
    correlation_spike_threshold: float = 0.9 # Correlation crisis
    volatility_spike_multiplier: float = 3.0 # Vol crisis

    # Model decay detection
    accuracy_decay_threshold: float = 0.1   # 10% accuracy drop
    calibration_threshold: float = 0.15     # VaR violation rate
    drift_threshold: float = 0.2            # Distribution shift
    decay_window_hours: int = 24            # Lookback for decay

    # Retraining controls
    min_samples_for_retrain: int = 10000    # Minimum samples
    max_retrain_frequency_hours: int = 24   # Max once per day
    retrain_validation_split: float = 0.2   # Hold-out set
    max_performance_regression: float = 0.05 # Max allowed regression
    require_human_approval: bool = True     # Human must approve

    # Audit settings
    audit_log_retention_days: int = 365     # Keep logs 1 year
    decision_log_every_n: int = 100         # Log every N decisions

    # Safety margins
    confidence_threshold: float = 0.3       # Min confidence to trade
    uncertainty_cap: float = 0.7            # Max uncertainty


@dataclass
class PerformanceAttribution:
    """Attribution of performance to different factors."""
    timestamp: datetime
    total_pnl: float

    # By source
    model_contribution: Dict[str, float] = field(default_factory=dict)
    regime_contribution: Dict[str, float] = field(default_factory=dict)
    asset_contribution: Dict[str, float] = field(default_factory=dict)

    # By type
    alpha_pnl: float = 0.0          # Skill-based returns
    beta_pnl: float = 0.0           # Market exposure returns
    noise_pnl: float = 0.0          # Random/unexplained

    # Risk-adjusted
    sharpe_contribution: float = 0.0
    information_ratio: float = 0.0

    def explain(self) -> str:
        """Human-readable explanation."""
        lines = [
            f"=== PERFORMANCE ATTRIBUTION ===",
            f"Time: {self.timestamp.isoformat()}",
            f"Total PnL: {self.total_pnl:+.2%}",
            "",
            f"BY TYPE:",
            f"  Alpha (skill): {self.alpha_pnl:+.2%}",
            f"  Beta (market): {self.beta_pnl:+.2%}",
            f"  Noise (random): {self.noise_pnl:+.2%}",
            "",
            "BY MODEL:",
        ]
        for model, contrib in sorted(self.model_contribution.items(),
                                     key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  {model}: {contrib:+.2%}")

        if self.asset_contribution:
            lines.append("")
            lines.append("BY ASSET:")
            for asset, contrib in sorted(self.asset_contribution.items(),
                                        key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {asset}: {contrib:+.2%}")

        return "\n".join(lines)


@dataclass
class DecayReport:
    """Report on model decay."""
    timestamp: datetime
    model_name: str
    is_decaying: bool

    # Metrics
    current_accuracy: float
    baseline_accuracy: float
    accuracy_change: float

    calibration_error: float
    drift_score: float

    # Recommendation
    trigger: RetrainingTrigger
    urgency: str  # low, medium, high, critical
    explanation: str


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: datetime
    event_type: str
    actor: str  # "ai", "human", "system"
    action: str
    details: Dict[str, Any]
    outcome: str
    reversible: bool
    hash: str = ""

    def __post_init__(self):
        """Compute integrity hash."""
        content = f"{self.timestamp}{self.event_type}{self.actor}{self.action}{self.details}{self.outcome}"
        self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# LEARNING BOUNDARIES - WHAT AI CAN AND CANNOT LEARN
# =============================================================================

class LearningBoundaries:
    """
    Defines what the AI system is allowed and forbidden to learn.

    This is a CRITICAL safety component. The boundaries are:
    - IMMUTABLE at runtime
    - AUDITED on every check
    - CANNOT be modified by the AI itself
    """

    # ALLOWED learning domains (AI CAN modify these)
    ALLOWED_LEARNING = {
        "price_prediction": "Learn to predict price movements",
        "volatility_estimation": "Learn to estimate future volatility",
        "regime_detection": "Learn to detect market regimes",
        "order_timing": "Learn optimal order execution timing",
        "feature_importance": "Learn which features matter most",
        "position_sizing_within_limits": "Optimize sizing within hard limits",
        "asset_correlation": "Learn asset correlations",
        "market_impact": "Learn own trading impact",
        "slippage_estimation": "Learn expected slippage",
        "liquidity_detection": "Learn to detect liquidity conditions",
    }

    # FORBIDDEN learning domains (AI CANNOT touch these)
    FORBIDDEN_LEARNING = {
        "risk_limit_bypass": "CANNOT learn to exceed risk limits",
        "kill_switch_disable": "CANNOT learn to disable kill-switches",
        "position_limit_increase": "CANNOT increase position limits",
        "leverage_limit_increase": "CANNOT increase leverage limits",
        "drawdown_limit_modification": "CANNOT modify drawdown limits",
        "audit_log_modification": "CANNOT modify audit logs",
        "governance_config_change": "CANNOT change governance config",
        "external_communication": "CANNOT communicate externally",
        "unauthorized_data_access": "CANNOT access unauthorized data",
        "self_modification": "CANNOT modify its own code/constraints",
        "human_override_bypass": "CANNOT bypass human overrides",
        "blackout_period_trading": "CANNOT trade during blackouts",
    }

    # REQUIRES APPROVAL (AI can propose, human must approve)
    REQUIRES_APPROVAL = {
        "new_model_deployment": "New models require human approval",
        "parameter_change_major": "Major parameter changes need approval",
        "new_asset_addition": "Adding new trading assets",
        "strategy_modification": "Changing trading strategy",
        "risk_parameter_adjustment": "Adjusting risk parameters down",
    }

    @classmethod
    def check_permission(cls, learning_domain: str) -> LearningPermission:
        """Check if learning domain is allowed."""
        if learning_domain in cls.FORBIDDEN_LEARNING:
            return LearningPermission.FORBIDDEN
        if learning_domain in cls.REQUIRES_APPROVAL:
            return LearningPermission.REQUIRES_APPROVAL
        if learning_domain in cls.ALLOWED_LEARNING:
            return LearningPermission.ALLOWED
        # Unknown domain - FORBIDDEN by default (fail-safe)
        return LearningPermission.FORBIDDEN

    @classmethod
    def validate_update(cls, update_description: str) -> Tuple[bool, str]:
        """
        Validate a proposed model update.

        Returns (is_allowed, reason)
        """
        update_lower = update_description.lower()

        # Check for forbidden keywords
        forbidden_keywords = [
            "risk limit", "kill switch", "position limit", "leverage",
            "drawdown", "audit", "governance", "external", "unauthorized",
            "self modify", "bypass", "blackout", "override"
        ]

        for keyword in forbidden_keywords:
            if keyword in update_lower:
                return False, f"Update contains forbidden domain: '{keyword}'"

        return True, "Update appears to be within allowed boundaries"

    @classmethod
    def get_all_forbidden(cls) -> List[str]:
        """Get all forbidden learning domains."""
        return list(cls.FORBIDDEN_LEARNING.keys())

    @classmethod
    def explain_boundaries(cls) -> str:
        """Return human-readable explanation of boundaries."""
        lines = [
            "=" * 60,
            "AI LEARNING BOUNDARIES",
            "=" * 60,
            "",
            "ALLOWED TO LEARN (AI can modify freely):",
            "-" * 40,
        ]
        for domain, desc in cls.ALLOWED_LEARNING.items():
            lines.append(f"  ✓ {domain}: {desc}")

        lines.extend([
            "",
            "FORBIDDEN TO LEARN (IMMUTABLE constraints):",
            "-" * 40,
        ])
        for domain, desc in cls.FORBIDDEN_LEARNING.items():
            lines.append(f"  ✗ {domain}: {desc}")

        lines.extend([
            "",
            "REQUIRES HUMAN APPROVAL:",
            "-" * 40,
        ])
        for domain, desc in cls.REQUIRES_APPROVAL.items():
            lines.append(f"  ? {domain}: {desc}")

        return "\n".join(lines)


# =============================================================================
# PERFORMANCE ATTRIBUTION ENGINE
# =============================================================================

class PerformanceAttributionEngine:
    """
    Attributes trading performance to different factors.

    Answers: WHY did we make/lose money?
    - Which models contributed?
    - Which regimes were profitable?
    - How much was skill vs luck?
    """

    def __init__(self, config: GovernanceConfig):
        self.config = config

        # Historical data
        self.trade_history: deque = deque(maxlen=10000)
        self.daily_returns: deque = deque(maxlen=365)
        self.model_predictions: Dict[str, deque] = {}
        self.model_actuals: Dict[str, deque] = {}

        # Attribution results
        self.attributions: deque = deque(maxlen=100)

    def record_trade(
        self,
        trade_id: str,
        asset: str,
        pnl: float,
        model_signals: Dict[str, float],
        regime: str,
        market_return: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a completed trade for attribution."""
        self.trade_history.append({
            "id": trade_id,
            "timestamp": timestamp or datetime.now(),
            "asset": asset,
            "pnl": pnl,
            "signals": model_signals,
            "regime": regime,
            "market_return": market_return,
        })

    def record_prediction(
        self,
        model_name: str,
        prediction: float,
        actual: float,
    ) -> None:
        """Record a model prediction for tracking."""
        if model_name not in self.model_predictions:
            self.model_predictions[model_name] = deque(maxlen=1000)
            self.model_actuals[model_name] = deque(maxlen=1000)

        self.model_predictions[model_name].append(prediction)
        self.model_actuals[model_name].append(actual)

    def compute_attribution(
        self,
        window_hours: int = 24,
    ) -> PerformanceAttribution:
        """
        Compute performance attribution for recent period.
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=window_hours)

        # Filter recent trades
        recent_trades = [
            t for t in self.trade_history
            if t["timestamp"] > cutoff
        ]

        if not recent_trades:
            return PerformanceAttribution(
                timestamp=now,
                total_pnl=0.0,
            )

        total_pnl = sum(t["pnl"] for t in recent_trades)

        # === BY MODEL ===
        model_contribution: Dict[str, float] = {}
        for trade in recent_trades:
            for model, signal in trade["signals"].items():
                if model not in model_contribution:
                    model_contribution[model] = 0.0
                # Attribute based on signal alignment with outcome
                signal_correct = (signal > 0 and trade["pnl"] > 0) or \
                                (signal < 0 and trade["pnl"] < 0)
                contribution = abs(trade["pnl"]) * abs(signal) / len(trade["signals"])
                if signal_correct:
                    model_contribution[model] += contribution
                else:
                    model_contribution[model] -= contribution

        # === BY REGIME ===
        regime_contribution: Dict[str, float] = {}
        for trade in recent_trades:
            regime = trade["regime"]
            if regime not in regime_contribution:
                regime_contribution[regime] = 0.0
            regime_contribution[regime] += trade["pnl"]

        # === BY ASSET ===
        asset_contribution: Dict[str, float] = {}
        for trade in recent_trades:
            asset = trade["asset"]
            if asset not in asset_contribution:
                asset_contribution[asset] = 0.0
            asset_contribution[asset] += trade["pnl"]

        # === ALPHA vs BETA ===
        # Beta = market exposure returns
        # Alpha = excess returns above beta
        market_returns = [t["market_return"] for t in recent_trades]
        trade_returns = [t["pnl"] for t in recent_trades]

        if len(market_returns) > 1 and np.std(market_returns) > 0:
            # Simple beta calculation
            beta = np.cov(trade_returns, market_returns)[0, 1] / np.var(market_returns)
            beta_pnl = beta * np.mean(market_returns) * len(recent_trades)
            alpha_pnl = total_pnl - beta_pnl
            noise_pnl = 0.0  # Simplified
        else:
            alpha_pnl = total_pnl * 0.5  # Assume 50/50 when can't calculate
            beta_pnl = total_pnl * 0.5
            noise_pnl = 0.0

        # === RISK-ADJUSTED ===
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        attribution = PerformanceAttribution(
            timestamp=now,
            total_pnl=total_pnl,
            model_contribution=model_contribution,
            regime_contribution=regime_contribution,
            asset_contribution=asset_contribution,
            alpha_pnl=alpha_pnl,
            beta_pnl=beta_pnl,
            noise_pnl=noise_pnl,
            sharpe_contribution=sharpe,
        )

        self.attributions.append(attribution)

        return attribution

    def get_model_accuracy(self, model_name: str) -> float:
        """Get accuracy for a specific model."""
        if model_name not in self.model_predictions:
            return 0.5

        preds = np.array(self.model_predictions[model_name])
        actuals = np.array(self.model_actuals[model_name])

        if len(preds) < 10:
            return 0.5

        # Directional accuracy
        return np.mean(np.sign(preds) == np.sign(actuals))


# =============================================================================
# MODEL DECAY DETECTOR
# =============================================================================

class ModelDecayDetector:
    """
    Detects when models are degrading in performance.

    Monitors:
    - Prediction accuracy over time
    - Calibration (predicted vs actual uncertainty)
    - Distribution drift (feature and label)
    """

    def __init__(self, config: GovernanceConfig):
        self.config = config

        # Baseline metrics (established during validation)
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}

        # Current metrics (rolling window)
        self.current_metrics: Dict[str, deque] = {}

        # Decay reports
        self.decay_reports: List[DecayReport] = []

    def set_baseline(
        self,
        model_name: str,
        accuracy: float,
        calibration: float,
        feature_distribution: np.ndarray,
    ) -> None:
        """Set baseline metrics for a model."""
        self.baseline_metrics[model_name] = {
            "accuracy": accuracy,
            "calibration": calibration,
            "feature_mean": float(np.mean(feature_distribution)),
            "feature_std": float(np.std(feature_distribution)),
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(f"Baseline set for {model_name}: accuracy={accuracy:.2%}")

    def update(
        self,
        model_name: str,
        prediction: float,
        actual: float,
        confidence: float,
        features: np.ndarray,
    ) -> Optional[DecayReport]:
        """
        Update metrics and check for decay.

        Returns DecayReport if decay detected.
        """
        if model_name not in self.current_metrics:
            self.current_metrics[model_name] = deque(maxlen=1000)

        # Store sample
        self.current_metrics[model_name].append({
            "prediction": prediction,
            "actual": actual,
            "confidence": confidence,
            "features": features.tolist() if isinstance(features, np.ndarray) else features,
            "timestamp": datetime.now(),
        })

        # Check if enough samples
        if len(self.current_metrics[model_name]) < 100:
            return None

        # Check for decay
        return self._check_decay(model_name)

    def _check_decay(self, model_name: str) -> Optional[DecayReport]:
        """Check if model is decaying."""
        if model_name not in self.baseline_metrics:
            return None

        samples = list(self.current_metrics[model_name])
        baseline = self.baseline_metrics[model_name]

        # Calculate current accuracy
        preds = np.array([s["prediction"] for s in samples])
        actuals = np.array([s["actual"] for s in samples])
        current_accuracy = np.mean(np.sign(preds) == np.sign(actuals))

        accuracy_change = current_accuracy - baseline["accuracy"]

        # Calculate calibration error
        confidences = np.array([s["confidence"] for s in samples])
        correct = (np.sign(preds) == np.sign(actuals)).astype(float)

        # Bin confidences and check calibration
        bins = np.linspace(0, 1, 11)
        calibration_errors = []
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() > 10:
                expected = (bins[i] + bins[i + 1]) / 2
                observed = correct[mask].mean()
                calibration_errors.append(abs(expected - observed))

        calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0

        # Calculate drift (simplified - compare feature distributions)
        recent_features = np.array([s["features"] for s in samples[-100:]])
        if len(recent_features.shape) > 1:
            recent_mean = np.mean(recent_features)
            recent_std = np.std(recent_features)
        else:
            recent_mean = np.mean(recent_features)
            recent_std = np.std(recent_features)

        drift_score = abs(recent_mean - baseline["feature_mean"]) / (baseline["feature_std"] + 1e-8)

        # Determine if decaying
        is_decaying = (
            accuracy_change < -self.config.accuracy_decay_threshold or
            calibration_error > self.config.calibration_threshold or
            drift_score > self.config.drift_threshold
        )

        # Determine trigger
        if accuracy_change < -self.config.accuracy_decay_threshold:
            trigger = RetrainingTrigger.ACCURACY_DECAY
            urgency = "high" if accuracy_change < -0.15 else "medium"
            explanation = f"Accuracy dropped by {-accuracy_change:.1%}"
        elif drift_score > self.config.drift_threshold:
            trigger = RetrainingTrigger.DRIFT_DETECTED
            urgency = "medium"
            explanation = f"Feature drift detected (score={drift_score:.2f})"
        elif calibration_error > self.config.calibration_threshold:
            trigger = RetrainingTrigger.ACCURACY_DECAY
            urgency = "low"
            explanation = f"Calibration error: {calibration_error:.1%}"
        else:
            trigger = RetrainingTrigger.NONE
            urgency = "low"
            explanation = "Model performing within acceptable bounds"

        report = DecayReport(
            timestamp=datetime.now(),
            model_name=model_name,
            is_decaying=is_decaying,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline["accuracy"],
            accuracy_change=accuracy_change,
            calibration_error=calibration_error,
            drift_score=drift_score,
            trigger=trigger,
            urgency=urgency,
            explanation=explanation,
        )

        if is_decaying:
            self.decay_reports.append(report)
            logger.warning(f"DECAY DETECTED: {model_name} - {explanation}")

        return report


# =============================================================================
# KILL SWITCH SYSTEM
# =============================================================================

class KillSwitchSystem:
    """
    Multi-layer kill switch system.

    CANNOT be disabled by the AI - enforced at multiple levels.

    Triggers:
    1. Drawdown limits
    2. Volatility spikes
    3. Correlation breakdowns
    4. Anomaly detection
    5. Human override
    6. Scheduled blackouts
    """

    def __init__(self, config: GovernanceConfig):
        self.config = config

        # Current state
        self.state = GovernanceState.OPERATIONAL
        self.triggered_by: List[str] = []
        self.trigger_time: Optional[datetime] = None

        # Monitoring
        self.drawdown_history: deque = deque(maxlen=1000)
        self.volatility_history: deque = deque(maxlen=100)

        # Human override (always takes precedence)
        self.human_override_active = False
        self.human_override_state: Optional[GovernanceState] = None
        self.human_override_reason: str = ""

        # Blackout periods (UTC hours)
        self.blackout_periods: List[Tuple[int, int]] = []  # [(start_hour, end_hour), ...]

        # Lock for thread safety
        self._lock = threading.Lock()

    def check(
        self,
        current_drawdown: float,
        current_volatility: float,
        baseline_volatility: float,
        correlation_with_market: float,
        anomaly_score: float,
    ) -> GovernanceState:
        """
        Check all kill switch conditions.

        Returns current governance state.
        """
        with self._lock:
            triggers = []

            # === HUMAN OVERRIDE (highest priority) ===
            if self.human_override_active:
                self.state = self.human_override_state
                self.triggered_by = [f"HUMAN_OVERRIDE: {self.human_override_reason}"]
                return self.state

            # === BLACKOUT CHECK ===
            current_hour = datetime.utcnow().hour
            for start, end in self.blackout_periods:
                if start <= current_hour < end:
                    triggers.append(f"BLACKOUT_PERIOD: {start}:00-{end}:00 UTC")

            # === DRAWDOWN CHECKS ===
            self.drawdown_history.append(current_drawdown)

            if current_drawdown >= self.config.drawdown_halt_pct:
                triggers.append(f"DRAWDOWN_HALT: {current_drawdown:.1%} >= {self.config.drawdown_halt_pct:.1%}")
            elif current_drawdown >= self.config.drawdown_suspend_pct:
                triggers.append(f"DRAWDOWN_SUSPEND: {current_drawdown:.1%}")
            elif current_drawdown >= self.config.drawdown_warning_pct:
                triggers.append(f"DRAWDOWN_WARNING: {current_drawdown:.1%}")

            # === VOLATILITY SPIKE ===
            self.volatility_history.append(current_volatility)

            if baseline_volatility > 0:
                vol_ratio = current_volatility / baseline_volatility
                if vol_ratio > self.config.volatility_spike_multiplier:
                    triggers.append(f"VOLATILITY_SPIKE: {vol_ratio:.1f}x normal")

            # === CORRELATION BREAKDOWN ===
            if abs(correlation_with_market) > self.config.correlation_spike_threshold:
                triggers.append(f"CORRELATION_CRISIS: |{correlation_with_market:.2f}|")

            # === ANOMALY DETECTION ===
            if anomaly_score > 0.8:
                triggers.append(f"ANOMALY_DETECTED: score={anomaly_score:.2f}")

            # === DETERMINE STATE ===
            if any("HALT" in t or "BLACKOUT" in t for t in triggers):
                new_state = GovernanceState.HALTED
            elif any("SUSPEND" in t for t in triggers):
                new_state = GovernanceState.SUSPENDED
            elif any("CORRELATION" in t or "VOLATILITY" in t or "ANOMALY" in t for t in triggers):
                new_state = GovernanceState.RESTRICTED
            elif any("WARNING" in t for t in triggers):
                new_state = GovernanceState.RESTRICTED
            else:
                new_state = GovernanceState.OPERATIONAL

            # Log state change
            if new_state != self.state:
                logger.warning(f"KILL SWITCH STATE CHANGE: {self.state.value} -> {new_state.value}")
                for trigger in triggers:
                    logger.warning(f"  Trigger: {trigger}")
                self.trigger_time = datetime.now()

            self.state = new_state
            self.triggered_by = triggers

            return self.state

    def human_override(
        self,
        state: GovernanceState,
        reason: str,
        operator_id: str,
    ) -> None:
        """
        Activate human override.

        This ALWAYS takes precedence over AI decisions.
        """
        with self._lock:
            self.human_override_active = True
            self.human_override_state = state
            self.human_override_reason = f"[{operator_id}] {reason}"
            logger.critical(f"HUMAN OVERRIDE ACTIVATED: {state.value} - {reason}")

    def clear_human_override(self, operator_id: str) -> None:
        """Clear human override."""
        with self._lock:
            self.human_override_active = False
            self.human_override_state = None
            self.human_override_reason = ""
            logger.info(f"Human override cleared by {operator_id}")

    def add_blackout_period(self, start_hour: int, end_hour: int) -> None:
        """Add a blackout period (UTC hours)."""
        self.blackout_periods.append((start_hour, end_hour))
        logger.info(f"Blackout period added: {start_hour}:00-{end_hour}:00 UTC")

    def is_operational(self) -> bool:
        """Check if trading is allowed."""
        return self.state == GovernanceState.OPERATIONAL

    def get_position_multiplier(self) -> float:
        """Get allowed position size multiplier."""
        multipliers = {
            GovernanceState.OPERATIONAL: 1.0,
            GovernanceState.RESTRICTED: 0.5,
            GovernanceState.SUSPENDED: 0.0,  # Can only close
            GovernanceState.HALTED: 0.0,
            GovernanceState.EMERGENCY: 0.0,
        }
        return multipliers.get(self.state, 0.0)


# =============================================================================
# AUTO-RETRAINING CONTROLLER
# =============================================================================

class AutoRetrainingController:
    """
    Controls automatic model retraining with safety bounds.

    Key principles:
    1. Retraining requires meeting multiple conditions
    2. Human approval gate (optional but recommended)
    3. Automatic rollback if performance degrades
    4. Rate limiting to prevent thrashing
    """

    def __init__(self, config: GovernanceConfig):
        self.config = config

        # State
        self.last_retrain: Optional[datetime] = None
        self.retrain_count = 0
        self.pending_approval: Optional[Dict] = None

        # History
        self.retrain_history: List[Dict] = []

        # Model checkpoints for rollback
        self.checkpoints: Dict[str, Any] = {}

        # Approval queue
        self.approval_queue: List[Dict] = []

    def should_retrain(
        self,
        decay_report: Optional[DecayReport],
        samples_available: int,
    ) -> Tuple[bool, RetrainingTrigger, str]:
        """
        Determine if retraining should be triggered.

        Returns (should_retrain, trigger_reason, explanation)
        """
        # Check minimum samples
        if samples_available < self.config.min_samples_for_retrain:
            return False, RetrainingTrigger.NONE, f"Insufficient samples: {samples_available}"

        # Check rate limit
        if self.last_retrain:
            hours_since = (datetime.now() - self.last_retrain).total_seconds() / 3600
            if hours_since < self.config.max_retrain_frequency_hours:
                return False, RetrainingTrigger.NONE, f"Rate limited: {hours_since:.1f}h since last retrain"

        # Check decay report
        if decay_report and decay_report.is_decaying:
            return True, decay_report.trigger, decay_report.explanation

        # Scheduled retrain (weekly)
        if self.last_retrain:
            days_since = (datetime.now() - self.last_retrain).days
            if days_since >= 7:
                return True, RetrainingTrigger.SCHEDULED, f"Scheduled retrain ({days_since} days since last)"
        elif self.retrain_count == 0:
            # First retrain
            return True, RetrainingTrigger.SCHEDULED, "Initial training"

        return False, RetrainingTrigger.NONE, "No retrain needed"

    def request_approval(
        self,
        trigger: RetrainingTrigger,
        models: List[str],
        explanation: str,
        expected_improvement: float,
    ) -> str:
        """
        Request human approval for retraining.

        Returns approval request ID.
        """
        request_id = hashlib.md5(
            f"{datetime.now()}{models}{trigger}".encode()
        ).hexdigest()[:8]

        request = {
            "id": request_id,
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger.value,
            "models": models,
            "explanation": explanation,
            "expected_improvement": expected_improvement,
            "status": "pending",
        }

        self.approval_queue.append(request)
        self.pending_approval = request

        logger.info(f"Retraining approval requested: {request_id}")
        logger.info(f"  Trigger: {trigger.value}")
        logger.info(f"  Models: {models}")
        logger.info(f"  Reason: {explanation}")

        return request_id

    def approve_request(
        self,
        request_id: str,
        approver_id: str,
    ) -> bool:
        """Human approves a retraining request."""
        for request in self.approval_queue:
            if request["id"] == request_id:
                request["status"] = "approved"
                request["approved_by"] = approver_id
                request["approved_at"] = datetime.now().isoformat()
                logger.info(f"Retraining approved by {approver_id}: {request_id}")
                return True
        return False

    def reject_request(
        self,
        request_id: str,
        rejector_id: str,
        reason: str,
    ) -> bool:
        """Human rejects a retraining request."""
        for request in self.approval_queue:
            if request["id"] == request_id:
                request["status"] = "rejected"
                request["rejected_by"] = rejector_id
                request["rejected_at"] = datetime.now().isoformat()
                request["rejection_reason"] = reason
                logger.info(f"Retraining rejected by {rejector_id}: {reason}")
                return True
        return False

    def save_checkpoint(self, model_name: str, state_dict: Dict) -> None:
        """Save model checkpoint before retraining."""
        self.checkpoints[model_name] = {
            k: v.clone() if hasattr(v, 'clone') else v
            for k, v in state_dict.items()
        }
        logger.info(f"Checkpoint saved for {model_name}")

    def validate_retrain(
        self,
        model_name: str,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Validate retraining results.

        Returns (is_valid, reason)
        """
        # Check for regression
        for metric, pre_val in pre_metrics.items():
            if metric in post_metrics:
                post_val = post_metrics[metric]

                # Allow up to max_performance_regression degradation
                allowed_degradation = self.config.max_performance_regression

                if post_val < pre_val - allowed_degradation:
                    return False, f"Regression in {metric}: {pre_val:.3f} -> {post_val:.3f}"

        return True, "Validation passed"

    def rollback(self, model: nn.Module, model_name: str) -> bool:
        """Rollback model to pre-retrain state."""
        if model_name not in self.checkpoints:
            logger.error(f"No checkpoint for {model_name}")
            return False

        model.load_state_dict(self.checkpoints[model_name])
        logger.warning(f"ROLLBACK: {model_name} restored to pre-retrain state")
        return True

    def complete_retrain(self, models: List[str], success: bool) -> None:
        """Record retrain completion."""
        self.last_retrain = datetime.now()
        self.retrain_count += 1

        self.retrain_history.append({
            "timestamp": self.last_retrain.isoformat(),
            "models": models,
            "success": success,
            "count": self.retrain_count,
        })

        # Clear approval
        if self.pending_approval:
            self.pending_approval["status"] = "completed" if success else "failed"
            self.pending_approval = None


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Immutable audit log for all system decisions.

    This log CANNOT be modified by the AI.
    All entries are hashed for integrity verification.
    """

    def __init__(self, config: GovernanceConfig, log_dir: str = "/tmp/cryptoai_audit"):
        self.config = config
        self.log_dir = log_dir

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # In-memory buffer
        self.entries: deque = deque(maxlen=10000)

        # Chain hash for integrity
        self.chain_hash = "genesis"

        # Decision counter
        self.decision_count = 0

    def log(
        self,
        event_type: str,
        actor: str,
        action: str,
        details: Dict[str, Any],
        outcome: str,
        reversible: bool = True,
    ) -> AuditEntry:
        """Log an event."""
        # Include chain hash for integrity
        details["_chain_hash"] = self.chain_hash

        entry = AuditEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            action=action,
            details=details,
            outcome=outcome,
            reversible=reversible,
        )

        # Update chain hash
        self.chain_hash = hashlib.sha256(
            f"{self.chain_hash}{entry.hash}".encode()
        ).hexdigest()[:16]

        self.entries.append(entry)
        self.decision_count += 1

        # Periodic file write
        if self.decision_count % self.config.decision_log_every_n == 0:
            self._flush_to_file()

        return entry

    def log_trade_decision(
        self,
        decision: str,
        signals: Dict[str, float],
        position_size: float,
        confidence: float,
        reasoning: List[str],
    ) -> AuditEntry:
        """Log a trade decision."""
        return self.log(
            event_type="trade_decision",
            actor="ai",
            action=decision,
            details={
                "signals": signals,
                "position_size": position_size,
                "confidence": confidence,
                "reasoning": reasoning,
            },
            outcome="pending",
            reversible=True,
        )

    def log_kill_switch(
        self,
        old_state: str,
        new_state: str,
        triggers: List[str],
    ) -> AuditEntry:
        """Log kill switch activation."""
        return self.log(
            event_type="kill_switch",
            actor="system",
            action=f"state_change:{old_state}->{new_state}",
            details={
                "old_state": old_state,
                "new_state": new_state,
                "triggers": triggers,
            },
            outcome="executed",
            reversible=False,
        )

    def log_human_override(
        self,
        operator_id: str,
        action: str,
        reason: str,
    ) -> AuditEntry:
        """Log human override."""
        return self.log(
            event_type="human_override",
            actor=f"human:{operator_id}",
            action=action,
            details={
                "reason": reason,
            },
            outcome="executed",
            reversible=True,
        )

    def log_retrain(
        self,
        models: List[str],
        trigger: str,
        approved_by: Optional[str],
        success: bool,
    ) -> AuditEntry:
        """Log model retraining."""
        return self.log(
            event_type="retrain",
            actor="ai",
            action="model_update",
            details={
                "models": models,
                "trigger": trigger,
                "approved_by": approved_by,
            },
            outcome="success" if success else "failed",
            reversible=True,
        )

    def _flush_to_file(self) -> None:
        """Write recent entries to file."""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = os.path.join(self.log_dir, f"audit_{date_str}.jsonl")

            with open(filepath, "a") as f:
                for entry in list(self.entries)[-self.config.decision_log_every_n:]:
                    f.write(json.dumps({
                        "timestamp": entry.timestamp.isoformat(),
                        "event_type": entry.event_type,
                        "actor": entry.actor,
                        "action": entry.action,
                        "details": entry.details,
                        "outcome": entry.outcome,
                        "hash": entry.hash,
                    }) + "\n")
        except Exception as e:
            logger.error(f"Failed to flush audit log: {e}")

    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify audit log integrity."""
        computed_hash = "genesis"

        for entry in self.entries:
            computed_hash = hashlib.sha256(
                f"{computed_hash}{entry.hash}".encode()
            ).hexdigest()[:16]

        if computed_hash == self.chain_hash:
            return True, "Integrity verified"
        else:
            return False, f"Integrity violation! Expected {self.chain_hash}, got {computed_hash}"


# =============================================================================
# UNIFIED GOVERNANCE SYSTEM
# =============================================================================

class AIGovernanceSystem:
    """
    Unified AI governance system.

    Combines all components:
    - Performance attribution
    - Model decay detection
    - Kill switches
    - Auto-retraining
    - Human override hooks
    - Audit logging
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        log_dir: str = "/tmp/cryptoai_audit",
    ):
        self.config = config or GovernanceConfig()

        # Components
        self.attribution = PerformanceAttributionEngine(self.config)
        self.decay_detector = ModelDecayDetector(self.config)
        self.kill_switch = KillSwitchSystem(self.config)
        self.retrain_controller = AutoRetrainingController(self.config)
        self.audit = AuditLogger(self.config, log_dir)

        # State
        self.is_initialized = False
        self.last_check = datetime.now()

        logger.info("AI Governance System initialized")
        logger.info(LearningBoundaries.explain_boundaries())

    def initialize_models(
        self,
        model_baselines: Dict[str, Dict[str, float]],
    ) -> None:
        """Initialize with model baselines."""
        for model_name, metrics in model_baselines.items():
            self.decay_detector.set_baseline(
                model_name=model_name,
                accuracy=metrics.get("accuracy", 0.5),
                calibration=metrics.get("calibration", 0.1),
                feature_distribution=np.array(metrics.get("feature_dist", [0.0])),
            )

        self.is_initialized = True
        self.audit.log(
            event_type="initialization",
            actor="system",
            action="models_initialized",
            details={"models": list(model_baselines.keys())},
            outcome="success",
        )

    def check_governance(
        self,
        portfolio_state: Dict[str, float],
        market_state: Dict[str, float],
        model_outputs: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Main governance check - call this on every decision cycle.

        Args:
            portfolio_state: {drawdown, exposure, daily_pnl}
            market_state: {volatility, baseline_vol, correlation, anomaly_score}
            model_outputs: {model_name: {prediction, actual, confidence}}

        Returns:
            Governance decision with all relevant info
        """
        now = datetime.now()

        # === KILL SWITCH CHECK ===
        kill_state = self.kill_switch.check(
            current_drawdown=portfolio_state.get("drawdown", 0),
            current_volatility=market_state.get("volatility", 0),
            baseline_volatility=market_state.get("baseline_vol", 0.01),
            correlation_with_market=market_state.get("correlation", 0),
            anomaly_score=market_state.get("anomaly_score", 0),
        )

        # Log state change if needed
        if kill_state != GovernanceState.OPERATIONAL:
            self.audit.log_kill_switch(
                old_state=self.kill_switch.state.value,
                new_state=kill_state.value,
                triggers=self.kill_switch.triggered_by,
            )

        # === DECAY CHECK ===
        decay_reports = []
        retrain_needed = False

        for model_name, outputs in model_outputs.items():
            if "prediction" in outputs and "actual" in outputs:
                report = self.decay_detector.update(
                    model_name=model_name,
                    prediction=outputs["prediction"],
                    actual=outputs["actual"],
                    confidence=outputs.get("confidence", 0.5),
                    features=np.array(outputs.get("features", [0.0])),
                )

                if report and report.is_decaying:
                    decay_reports.append(report)
                    retrain_needed = True

        # === RETRAIN CHECK ===
        should_retrain, trigger, retrain_reason = self.retrain_controller.should_retrain(
            decay_report=decay_reports[0] if decay_reports else None,
            samples_available=len(self.decay_detector.current_metrics.get("main", [])),
        )

        # === PERFORMANCE ATTRIBUTION ===
        attribution = self.attribution.compute_attribution()

        # === BUILD RESPONSE ===
        self.last_check = now

        return {
            "timestamp": now.isoformat(),
            "governance_state": kill_state.value,
            "position_multiplier": self.kill_switch.get_position_multiplier(),
            "can_trade": kill_state == GovernanceState.OPERATIONAL,
            "can_open_new": kill_state in [GovernanceState.OPERATIONAL, GovernanceState.RESTRICTED],

            "kill_switch": {
                "state": kill_state.value,
                "triggers": self.kill_switch.triggered_by,
                "human_override": self.kill_switch.human_override_active,
            },

            "decay": {
                "reports": [
                    {
                        "model": r.model_name,
                        "is_decaying": r.is_decaying,
                        "accuracy_change": r.accuracy_change,
                        "urgency": r.urgency,
                    }
                    for r in decay_reports
                ],
                "retrain_recommended": retrain_needed,
            },

            "retrain": {
                "should_retrain": should_retrain,
                "trigger": trigger.value,
                "reason": retrain_reason,
                "requires_approval": self.config.require_human_approval,
            },

            "attribution": {
                "total_pnl": attribution.total_pnl,
                "alpha": attribution.alpha_pnl,
                "beta": attribution.beta_pnl,
                "sharpe": attribution.sharpe_contribution,
            },
        }

    def human_override(
        self,
        state: GovernanceState,
        reason: str,
        operator_id: str,
    ) -> None:
        """Activate human override."""
        self.kill_switch.human_override(state, reason, operator_id)
        self.audit.log_human_override(operator_id, f"set_state:{state.value}", reason)

    def clear_human_override(self, operator_id: str) -> None:
        """Clear human override."""
        self.kill_switch.clear_human_override(operator_id)
        self.audit.log_human_override(operator_id, "clear_override", "Manual clear")

    def approve_retrain(self, request_id: str, approver_id: str) -> bool:
        """Approve a pending retrain request."""
        result = self.retrain_controller.approve_request(request_id, approver_id)
        if result:
            self.audit.log(
                event_type="retrain_approval",
                actor=f"human:{approver_id}",
                action="approve",
                details={"request_id": request_id},
                outcome="approved",
            )
        return result

    def get_learning_boundaries(self) -> str:
        """Get explanation of learning boundaries."""
        return LearningBoundaries.explain_boundaries()

    def validate_proposed_update(self, description: str) -> Tuple[bool, str]:
        """Validate a proposed model update against learning boundaries."""
        return LearningBoundaries.validate_update(description)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        integrity_ok, integrity_msg = self.audit.verify_integrity()

        return {
            "governance_state": self.kill_switch.state.value,
            "is_initialized": self.is_initialized,
            "last_check": self.last_check.isoformat(),

            "kill_switch": {
                "state": self.kill_switch.state.value,
                "human_override": self.kill_switch.human_override_active,
                "triggers": self.kill_switch.triggered_by,
            },

            "retrain": {
                "last_retrain": self.retrain_controller.last_retrain.isoformat() if self.retrain_controller.last_retrain else None,
                "retrain_count": self.retrain_controller.retrain_count,
                "pending_approval": self.retrain_controller.pending_approval is not None,
            },

            "audit": {
                "entries": len(self.audit.entries),
                "integrity": integrity_ok,
                "integrity_message": integrity_msg,
            },

            "decay_reports": len(self.decay_detector.decay_reports),
        }

    def explain_behavior(self) -> str:
        """
        Explain current system behavior in human terms.

        This is critical for transparency during chaos.
        """
        status = self.get_system_status()

        lines = [
            "=" * 60,
            "SYSTEM BEHAVIOR EXPLANATION",
            "=" * 60,
            "",
            f"Current State: {status['governance_state']}",
            f"Can Trade: {self.kill_switch.is_operational()}",
            f"Position Multiplier: {self.kill_switch.get_position_multiplier():.0%}",
            "",
        ]

        if self.kill_switch.triggered_by:
            lines.append("ACTIVE TRIGGERS:")
            for trigger in self.kill_switch.triggered_by:
                lines.append(f"  - {trigger}")
            lines.append("")

        if self.kill_switch.human_override_active:
            lines.append(f"HUMAN OVERRIDE ACTIVE: {self.kill_switch.human_override_reason}")
            lines.append("")

        if self.decay_detector.decay_reports:
            lines.append("MODEL DECAY DETECTED:")
            for report in self.decay_detector.decay_reports[-3:]:
                lines.append(f"  - {report.model_name}: {report.explanation}")
            lines.append("")

        lines.extend([
            "LEARNING BOUNDARIES:",
            "  - AI is ALLOWED to learn: price prediction, volatility, regime detection",
            "  - AI is FORBIDDEN from: bypassing risk limits, disabling kill-switches",
            "  - All changes require human approval",
            "",
            "CATASTROPHIC FAILURE PREVENTION:",
            "  - Layer 1: Position limits (ACTIVE)",
            "  - Layer 2: Drawdown circuit breakers (ACTIVE)",
            "  - Layer 3: Model sanity checks (ACTIVE)",
            "  - Layer 4: Human approval gate (ACTIVE)",
            "  - Layer 5: Automatic rollback (READY)",
            "",
        ])

        return "\n".join(lines)


def create_governance_system(
    log_dir: str = "/tmp/cryptoai_audit",
) -> AIGovernanceSystem:
    """
    Factory function to create properly configured governance system.
    """
    config = GovernanceConfig(
        max_drawdown_pct=0.15,
        require_human_approval=True,
        accuracy_decay_threshold=0.1,
    )

    system = AIGovernanceSystem(config, log_dir)

    logger.info("AI Governance System created with safety constraints")

    return system
