"""Alerting system for the trading AI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
import asyncio
from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Single alert."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    condition: Callable[[], bool]
    level: AlertLevel
    title: str
    message_template: str
    cooldown_minutes: int = 5  # Minimum time between alerts
    auto_resolve: bool = True


class AlertManager:
    """
    Manages alerts and notifications for the trading system.

    Handles:
    - Alert generation from rules
    - Deduplication and throttling
    - Multi-channel notification
    - Alert lifecycle (acknowledge, resolve)
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Alert storage
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []

        # Rules
        self._rules: Dict[str, AlertRule] = {}
        self._last_triggered: Dict[str, datetime] = {}

        # Notification channels
        self._channels: Dict[str, Callable[[Alert], None]] = {}
        self._channel_config: Dict[str, Dict] = {}

        # Register default channels
        self._register_default_channels()

        # Register default rules
        self._register_default_rules()

    def _register_default_channels(self) -> None:
        """Register default notification channels."""
        # Log channel (always available)
        self.register_channel("log", self._log_alert)

    def _log_alert(self, alert: Alert) -> None:
        """Log alert to console."""
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.EMERGENCY: logger.critical,
        }

        log_fn = level_map.get(alert.level, logger.info)
        log_fn(f"[{alert.level.value.upper()}] {alert.title}: {alert.message}")

    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        # These are placeholders - actual conditions would check metrics
        pass

    def register_channel(
        self,
        name: str,
        handler: Callable[[Alert], None],
        config: Optional[Dict] = None,
    ) -> None:
        """Register a notification channel."""
        with self._lock:
            self._channels[name] = handler
            self._channel_config[name] = config or {}

        logger.info(f"Registered alert channel: {name}")

    def register_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        with self._lock:
            self._rules[rule.name] = rule

        logger.debug(f"Registered alert rule: {rule.name}")

    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict] = None,
        channels: Optional[List[str]] = None,
    ) -> Alert:
        """
        Create and send a new alert.

        Args:
            level: Severity level
            title: Alert title
            message: Alert message
            source: Source component
            metadata: Additional metadata
            channels: Notification channels (default: all)

        Returns:
            Created alert
        """
        alert_id = f"{source}_{datetime.utcnow().timestamp()}"

        alert = Alert(
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {},
        )

        with self._lock:
            self._alerts[alert_id] = alert
            self._alert_history.append(alert)

            # Keep history bounded
            if len(self._alert_history) > 10000:
                self._alert_history = self._alert_history[-5000:]

        # Send notifications
        self._notify(alert, channels)

        return alert

    def _notify(
        self,
        alert: Alert,
        channels: Optional[List[str]] = None,
    ) -> None:
        """Send alert to notification channels."""
        channels = channels or list(self._channels.keys())

        for channel_name in channels:
            if channel_name not in self._channels:
                continue

            try:
                handler = self._channels[channel_name]
                handler(alert)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel_name}: {e}")

    def check_rules(self, context: Optional[Dict] = None) -> List[Alert]:
        """
        Check all alert rules and generate alerts.

        Args:
            context: Optional context for rule evaluation

        Returns:
            List of triggered alerts
        """
        triggered = []

        with self._lock:
            for rule_name, rule in self._rules.items():
                # Check cooldown
                last_time = self._last_triggered.get(rule_name)
                if last_time:
                    cooldown = timedelta(minutes=rule.cooldown_minutes)
                    if datetime.utcnow() - last_time < cooldown:
                        continue

                # Evaluate condition
                try:
                    if rule.condition():
                        alert = self.create_alert(
                            level=rule.level,
                            title=rule.title,
                            message=rule.message_template,
                            source=f"rule:{rule_name}",
                        )
                        triggered.append(alert)
                        self._last_triggered[rule_name] = datetime.utcnow()

                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_name}: {e}")

        return triggered

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            self._alerts[alert_id].acknowledged = True

        logger.info(f"Acknowledged alert: {alert_id}")
        return True

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            self._alerts[alert_id].resolved = True

        logger.info(f"Resolved alert: {alert_id}")
        return True

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        source: Optional[str] = None,
    ) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with self._lock:
            alerts = [
                a for a in self._alerts.values()
                if not a.resolved
            ]

            if level:
                alerts = [a for a in alerts if a.level == level]

            if source:
                alerts = [a for a in alerts if a.source == source]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(
        self,
        hours: int = 24,
        level: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """Get alert history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            history = [
                a for a in self._alert_history
                if a.timestamp > cutoff
            ]

            if level:
                history = [a for a in history if a.level == level]

        return sorted(history, key=lambda a: a.timestamp, reverse=True)

    def get_alert_counts(self, hours: int = 24) -> Dict[str, int]:
        """Get alert counts by level."""
        history = self.get_alert_history(hours)

        counts = {level.value: 0 for level in AlertLevel}
        for alert in history:
            counts[alert.level.value] += 1

        return counts

    def clear_resolved(self, older_than_hours: int = 24) -> int:
        """Clear resolved alerts older than threshold."""
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        cleared = 0

        with self._lock:
            to_remove = [
                alert_id for alert_id, alert in self._alerts.items()
                if alert.resolved and alert.timestamp < cutoff
            ]

            for alert_id in to_remove:
                del self._alerts[alert_id]
                cleared += 1

        return cleared


class SlackAlertChannel:
    """Slack notification channel."""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel

    def send(self, alert: Alert) -> None:
        """Send alert to Slack."""
        try:
            import requests

            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#f2c744",
                AlertLevel.CRITICAL: "#d00000",
                AlertLevel.EMERGENCY: "#000000",
            }

            payload = {
                "channel": self.channel,
                "attachments": [{
                    "color": color_map.get(alert.level, "#808080"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Level", "value": alert.level.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                    ],
                    "ts": alert.timestamp.timestamp(),
                }],
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class TelegramAlertChannel:
    """Telegram notification channel."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    def send(self, alert: Alert) -> None:
        """Send alert to Telegram."""
        try:
            import requests

            emoji_map = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🔥",
            }

            emoji = emoji_map.get(alert.level, "📢")
            message = (
                f"{emoji} *{alert.title}*\n\n"
                f"{alert.message}\n\n"
                f"_Level: {alert.level.value}_\n"
                f"_Source: {alert.source}_"
            )

            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


class EmailAlertChannel:
    """Email notification channel."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, alert: Alert) -> None:
        """Send alert via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"

            body = f"""
            Alert Details:

            Title: {alert.title}
            Level: {alert.level.value}
            Source: {alert.source}
            Time: {alert.timestamp.isoformat()}

            Message:
            {alert.message}

            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
