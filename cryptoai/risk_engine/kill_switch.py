"""Kill switch and emergency controls."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional
import torch
from loguru import logger


class KillSwitchTrigger(str, Enum):
    """Kill switch trigger types."""

    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    BLACK_SWAN_ALERT = "black_swan_alert"
    EXCHANGE_CONNECTIVITY_LOST = "exchange_connectivity_lost"
    LIQUIDATION_RISK = "liquidation_risk"
    MANUAL_TRIGGER = "manual_trigger"
    ANOMALY_DETECTED = "anomaly_detected"
    CASCADE_DETECTED = "cascade_detected"


class KillSwitchAction(str, Enum):
    """Actions to take when kill switch triggered."""

    FLATTEN_ALL = "flatten_all_positions"
    SUSPEND_TRADING = "suspend_trading"
    REDUCE_EXPOSURE = "reduce_exposure"
    MAKER_ONLY = "maker_only_mode"
    ALERT_OPERATORS = "alert_operators"


@dataclass
class KillSwitchEvent:
    """Record of a kill switch event."""

    timestamp: datetime
    trigger: KillSwitchTrigger
    action: KillSwitchAction
    details: Dict
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class KillSwitch:
    """
    Emergency kill switch system.

    Monitors multiple risk conditions and takes emergency action.
    """

    def __init__(
        self,
        max_drawdown_threshold: float = 0.15,
        daily_loss_threshold: float = 0.05,
        black_swan_threshold: float = 0.5,
        liquidation_threshold: float = 0.3,
        cooldown_minutes: int = 60,
        auto_resume: bool = False,
    ):
        self.max_drawdown_threshold = max_drawdown_threshold
        self.daily_loss_threshold = daily_loss_threshold
        self.black_swan_threshold = black_swan_threshold
        self.liquidation_threshold = liquidation_threshold
        self.cooldown_minutes = cooldown_minutes
        self.auto_resume = auto_resume

        # State
        self._is_triggered = False
        self._current_trigger: Optional[KillSwitchTrigger] = None
        self._trigger_time: Optional[datetime] = None
        self._events: List[KillSwitchEvent] = []

        # Callbacks
        self._flatten_callback: Optional[Callable] = None
        self._alert_callback: Optional[Callable] = None

    def register_callbacks(
        self,
        flatten_callback: Optional[Callable] = None,
        alert_callback: Optional[Callable] = None,
    ):
        """Register callbacks for kill switch actions."""
        self._flatten_callback = flatten_callback
        self._alert_callback = alert_callback

    def check(
        self,
        current_drawdown: float,
        daily_pnl_pct: float,
        black_swan_prob: float = 0.0,
        liquidation_prob: float = 0.0,
        exchange_connected: bool = True,
        anomaly_score: float = 0.0,
        cascade_prob: float = 0.0,
    ) -> Optional[KillSwitchEvent]:
        """
        Check all kill switch conditions.

        Returns KillSwitchEvent if triggered, None otherwise.
        """
        now = datetime.utcnow()

        # Check cooldown
        if self._is_triggered and self.auto_resume:
            if self._trigger_time and (now - self._trigger_time) > timedelta(minutes=self.cooldown_minutes):
                self._resolve_current()

        # Check each condition
        trigger = None
        action = None
        details = {}

        if current_drawdown > self.max_drawdown_threshold:
            trigger = KillSwitchTrigger.MAX_DRAWDOWN_EXCEEDED
            action = KillSwitchAction.FLATTEN_ALL
            details = {"drawdown": current_drawdown, "threshold": self.max_drawdown_threshold}

        elif daily_pnl_pct < -self.daily_loss_threshold:
            trigger = KillSwitchTrigger.DAILY_LOSS_EXCEEDED
            action = KillSwitchAction.SUSPEND_TRADING
            details = {"daily_loss": daily_pnl_pct, "threshold": self.daily_loss_threshold}

        elif black_swan_prob > self.black_swan_threshold:
            trigger = KillSwitchTrigger.BLACK_SWAN_ALERT
            action = KillSwitchAction.FLATTEN_ALL
            details = {"probability": black_swan_prob, "threshold": self.black_swan_threshold}

        elif not exchange_connected:
            trigger = KillSwitchTrigger.EXCHANGE_CONNECTIVITY_LOST
            action = KillSwitchAction.SUSPEND_TRADING
            details = {"exchange_connected": False}

        elif liquidation_prob > self.liquidation_threshold:
            trigger = KillSwitchTrigger.LIQUIDATION_RISK
            action = KillSwitchAction.REDUCE_EXPOSURE
            details = {"probability": liquidation_prob, "threshold": self.liquidation_threshold}

        elif cascade_prob > 0.5:
            trigger = KillSwitchTrigger.CASCADE_DETECTED
            action = KillSwitchAction.FLATTEN_ALL
            details = {"probability": cascade_prob}

        elif anomaly_score > 0.9:
            trigger = KillSwitchTrigger.ANOMALY_DETECTED
            action = KillSwitchAction.REDUCE_EXPOSURE
            details = {"score": anomaly_score}

        if trigger is not None:
            event = self._trigger(trigger, action, details)
            return event

        return None

    def _trigger(
        self,
        trigger: KillSwitchTrigger,
        action: KillSwitchAction,
        details: Dict,
    ) -> KillSwitchEvent:
        """Trigger kill switch."""
        now = datetime.utcnow()

        self._is_triggered = True
        self._current_trigger = trigger
        self._trigger_time = now

        event = KillSwitchEvent(
            timestamp=now,
            trigger=trigger,
            action=action,
            details=details,
        )
        self._events.append(event)

        logger.critical(f"KILL SWITCH TRIGGERED: {trigger.value} -> {action.value}")
        logger.critical(f"Details: {details}")

        # Execute action
        self._execute_action(action)

        return event

    def _execute_action(self, action: KillSwitchAction):
        """Execute kill switch action."""
        if action == KillSwitchAction.FLATTEN_ALL:
            if self._flatten_callback:
                self._flatten_callback()
            logger.warning("Executing: FLATTEN ALL POSITIONS")

        elif action == KillSwitchAction.SUSPEND_TRADING:
            logger.warning("Executing: SUSPEND TRADING")

        elif action == KillSwitchAction.REDUCE_EXPOSURE:
            logger.warning("Executing: REDUCE EXPOSURE")

        elif action == KillSwitchAction.MAKER_ONLY:
            logger.warning("Executing: MAKER ONLY MODE")

        # Always alert operators
        if self._alert_callback:
            self._alert_callback(action)

    def _resolve_current(self):
        """Resolve current kill switch event."""
        if self._events:
            self._events[-1].resolved = True
            self._events[-1].resolved_at = datetime.utcnow()

        self._is_triggered = False
        self._current_trigger = None
        self._trigger_time = None

        logger.info("Kill switch resolved, resuming normal operation")

    def manual_trigger(self, reason: str = "Manual trigger"):
        """Manually trigger kill switch."""
        self._trigger(
            KillSwitchTrigger.MANUAL_TRIGGER,
            KillSwitchAction.FLATTEN_ALL,
            {"reason": reason},
        )

    def manual_reset(self):
        """Manually reset kill switch."""
        self._resolve_current()
        logger.info("Kill switch manually reset")

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch is currently triggered."""
        return self._is_triggered

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if not self._is_triggered:
            return True

        # Some actions still allow trading
        if self._events:
            last_action = self._events[-1].action
            return last_action in [
                KillSwitchAction.MAKER_ONLY,
                KillSwitchAction.REDUCE_EXPOSURE,
            ]

        return False

    def get_exposure_limit(self) -> float:
        """Get current exposure limit (0-1)."""
        if not self._is_triggered:
            return 1.0

        if self._events:
            last_action = self._events[-1].action
            if last_action == KillSwitchAction.REDUCE_EXPOSURE:
                return 0.5
            elif last_action == KillSwitchAction.MAKER_ONLY:
                return 0.3
            else:
                return 0.0

        return 1.0

    def get_events(self, limit: int = 10) -> List[KillSwitchEvent]:
        """Get recent kill switch events."""
        return self._events[-limit:]


class GradualRecovery:
    """
    Manages gradual recovery after kill switch events.

    Slowly increases exposure over time to avoid re-triggering.
    """

    def __init__(
        self,
        recovery_hours: int = 24,
        min_exposure: float = 0.1,
        max_exposure: float = 1.0,
    ):
        self.recovery_hours = recovery_hours
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

        self._recovery_start: Optional[datetime] = None
        self._is_recovering = False

    def start_recovery(self):
        """Start gradual recovery."""
        self._recovery_start = datetime.utcnow()
        self._is_recovering = True
        logger.info("Starting gradual recovery")

    def get_exposure_limit(self) -> float:
        """Get current allowed exposure during recovery."""
        if not self._is_recovering or self._recovery_start is None:
            return self.max_exposure

        elapsed = datetime.utcnow() - self._recovery_start
        elapsed_hours = elapsed.total_seconds() / 3600

        if elapsed_hours >= self.recovery_hours:
            self._is_recovering = False
            return self.max_exposure

        # Linear recovery
        progress = elapsed_hours / self.recovery_hours
        current_limit = self.min_exposure + progress * (self.max_exposure - self.min_exposure)

        return current_limit

    def is_recovering(self) -> bool:
        """Check if in recovery mode."""
        return self._is_recovering
