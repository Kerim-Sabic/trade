"""Metrics collection and aggregation for the trading system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
import time
import json
import numpy as np
from loguru import logger


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Single metric value."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)  # For histograms


class MetricsCollector:
    """
    Central metrics collection system.

    Collects and aggregates metrics from all system components.
    Compatible with Prometheus format for export.
    """

    def __init__(
        self,
        retention_hours: int = 24,
        aggregation_interval_seconds: float = 60.0,
    ):
        self.retention_hours = retention_hours
        self.aggregation_interval = aggregation_interval_seconds
        self._lock = threading.Lock()

        # Metric definitions
        self._definitions: Dict[str, MetricDefinition] = {}

        # Raw metric values
        self._metrics: Dict[str, deque] = {}

        # Aggregated values
        self._aggregates: Dict[str, Dict[str, float]] = {}

        # Register default trading metrics
        self._register_default_metrics()

        # Start aggregation thread
        self._stop_event = threading.Event()
        self._aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True,
        )
        self._aggregation_thread.start()

    def _register_default_metrics(self) -> None:
        """Register default trading metrics."""
        default_metrics = [
            MetricDefinition(
                name="trading_pnl",
                metric_type=MetricType.GAUGE,
                description="Current PnL",
                labels=["asset", "strategy"],
            ),
            MetricDefinition(
                name="trading_position_size",
                metric_type=MetricType.GAUGE,
                description="Current position size",
                labels=["asset", "exchange"],
            ),
            MetricDefinition(
                name="trading_orders_total",
                metric_type=MetricType.COUNTER,
                description="Total orders placed",
                labels=["asset", "side", "status"],
            ),
            MetricDefinition(
                name="inference_latency_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Inference latency in milliseconds",
                labels=["model"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            ),
            MetricDefinition(
                name="inference_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total inference requests",
                labels=["model", "status"],
            ),
            MetricDefinition(
                name="model_anomaly_score",
                metric_type=MetricType.GAUGE,
                description="Black swan anomaly score",
                labels=["asset"],
            ),
            MetricDefinition(
                name="risk_exposure",
                metric_type=MetricType.GAUGE,
                description="Current risk exposure",
                labels=["asset"],
            ),
            MetricDefinition(
                name="market_volatility",
                metric_type=MetricType.GAUGE,
                description="Market volatility estimate",
                labels=["asset", "timeframe"],
            ),
            MetricDefinition(
                name="data_latency_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Data feed latency",
                labels=["source", "asset"],
                buckets=[10, 50, 100, 500, 1000, 5000],
            ),
        ]

        for metric in default_metrics:
            self.register_metric(metric)

    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric definition."""
        with self._lock:
            self._definitions[definition.name] = definition
            max_len = int(self.retention_hours * 3600 / self.aggregation_interval)
            self._metrics[definition.name] = deque(maxlen=max_len * 100)
            self._aggregates[definition.name] = {}

        logger.debug(f"Registered metric: {definition.name}")

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        labels = labels or {}

        with self._lock:
            if name not in self._metrics:
                logger.warning(f"Unregistered metric: {name}")
                return

            metric_value = MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
            )

            self._metrics[name].append(metric_value)

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        self.record(name, value, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        self.record(name, value, labels)

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for histogram/summary."""
        self.record(name, value, labels)

    def _aggregation_loop(self) -> None:
        """Background thread for metric aggregation."""
        while not self._stop_event.is_set():
            try:
                self._aggregate_metrics()
            except Exception as e:
                logger.error(f"Aggregation error: {e}")

            time.sleep(self.aggregation_interval)

    def _aggregate_metrics(self) -> None:
        """Aggregate metrics over the interval."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.aggregation_interval)

        with self._lock:
            for name, definition in self._definitions.items():
                values = self._metrics.get(name, [])

                # Filter recent values
                recent = [v for v in values if v.timestamp > cutoff]

                if not recent:
                    continue

                # Group by labels
                label_groups: Dict[str, List[float]] = {}
                for v in recent:
                    label_key = json.dumps(v.labels, sort_keys=True)
                    if label_key not in label_groups:
                        label_groups[label_key] = []
                    label_groups[label_key].append(v.value)

                # Compute aggregates
                for label_key, group_values in label_groups.items():
                    agg_key = f"{name}:{label_key}"

                    if definition.metric_type == MetricType.COUNTER:
                        self._aggregates[name][agg_key] = sum(group_values)

                    elif definition.metric_type == MetricType.GAUGE:
                        self._aggregates[name][agg_key] = group_values[-1]

                    elif definition.metric_type == MetricType.HISTOGRAM:
                        self._aggregates[name][agg_key] = {
                            "count": len(group_values),
                            "sum": sum(group_values),
                            "min": min(group_values),
                            "max": max(group_values),
                            "mean": np.mean(group_values),
                            "p50": np.percentile(group_values, 50),
                            "p95": np.percentile(group_values, 95),
                            "p99": np.percentile(group_values, 99),
                        }

                    elif definition.metric_type == MetricType.SUMMARY:
                        self._aggregates[name][agg_key] = {
                            "count": len(group_values),
                            "sum": sum(group_values),
                            "mean": np.mean(group_values),
                            "std": np.std(group_values),
                        }

    def get_metric(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Any]:
        """Get current metric value."""
        with self._lock:
            if name not in self._aggregates:
                return None

            if labels:
                label_key = json.dumps(labels, sort_keys=True)
                agg_key = f"{name}:{label_key}"
                return self._aggregates[name].get(agg_key)

            return self._aggregates[name]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                name: dict(aggregates)
                for name, aggregates in self._aggregates.items()
            }

    def get_time_series(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        hours: int = 1,
    ) -> List[Dict]:
        """Get time series data for a metric."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            if name not in self._metrics:
                return []

            values = self._metrics[name]
            result = []

            for v in values:
                if v.timestamp < cutoff:
                    continue

                if labels and v.labels != labels:
                    continue

                result.append({
                    "timestamp": v.timestamp.isoformat(),
                    "value": v.value,
                    "labels": v.labels,
                })

            return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            for name, definition in self._definitions.items():
                # Header
                lines.append(f"# HELP {name} {definition.description}")
                lines.append(f"# TYPE {name} {definition.metric_type.value}")

                aggregates = self._aggregates.get(name, {})

                for agg_key, value in aggregates.items():
                    # Parse labels
                    _, label_str = agg_key.split(":", 1)
                    labels = json.loads(label_str)

                    label_parts = ",".join(
                        f'{k}="{v}"' for k, v in labels.items()
                    )
                    label_suffix = f"{{{label_parts}}}" if label_parts else ""

                    if definition.metric_type == MetricType.HISTOGRAM:
                        # Histogram format
                        lines.append(f"{name}_count{label_suffix} {value['count']}")
                        lines.append(f"{name}_sum{label_suffix} {value['sum']}")

                        for bucket in definition.buckets:
                            count = sum(1 for v in self._get_raw_values(name, labels) if v <= bucket)
                            lines.append(f'{name}_bucket{{le="{bucket}",{label_parts}}} {count}')

                        lines.append(f'{name}_bucket{{le="+Inf",{label_parts}}} {value["count"]}')

                    else:
                        # Simple value
                        val = value if isinstance(value, (int, float)) else value.get("mean", 0)
                        lines.append(f"{name}{label_suffix} {val}")

        return "\n".join(lines)

    def _get_raw_values(
        self,
        name: str,
        labels: Dict[str, str],
    ) -> List[float]:
        """Get raw values for a metric with specific labels."""
        values = self._metrics.get(name, [])
        return [v.value for v in values if v.labels == labels]

    def stop(self) -> None:
        """Stop the metrics collector."""
        self._stop_event.set()
        self._aggregation_thread.join(timeout=5.0)


class TradingMetrics:
    """
    Convenience class for recording trading-specific metrics.

    Provides typed methods for common trading metrics.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def record_pnl(self, asset: str, pnl: float, strategy: str = "default") -> None:
        """Record PnL."""
        self.collector.set_gauge(
            "trading_pnl",
            pnl,
            {"asset": asset, "strategy": strategy},
        )

    def record_position(self, asset: str, size: float, exchange: str) -> None:
        """Record position size."""
        self.collector.set_gauge(
            "trading_position_size",
            size,
            {"asset": asset, "exchange": exchange},
        )

    def record_order(
        self,
        asset: str,
        side: str,
        status: str,
    ) -> None:
        """Record order."""
        self.collector.increment(
            "trading_orders_total",
            1.0,
            {"asset": asset, "side": side, "status": status},
        )

    def record_inference(
        self,
        model: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record inference metrics."""
        self.collector.observe(
            "inference_latency_ms",
            latency_ms,
            {"model": model},
        )
        self.collector.increment(
            "inference_requests_total",
            1.0,
            {"model": model, "status": "success" if success else "error"},
        )

    def record_anomaly_score(self, asset: str, score: float) -> None:
        """Record anomaly score."""
        self.collector.set_gauge(
            "model_anomaly_score",
            score,
            {"asset": asset},
        )

    def record_risk_exposure(self, asset: str, exposure: float) -> None:
        """Record risk exposure."""
        self.collector.set_gauge(
            "risk_exposure",
            exposure,
            {"asset": asset},
        )

    def record_volatility(
        self,
        asset: str,
        volatility: float,
        timeframe: str = "1h",
    ) -> None:
        """Record market volatility."""
        self.collector.set_gauge(
            "market_volatility",
            volatility,
            {"asset": asset, "timeframe": timeframe},
        )

    def record_data_latency(
        self,
        source: str,
        asset: str,
        latency_ms: float,
    ) -> None:
        """Record data feed latency."""
        self.collector.observe(
            "data_latency_ms",
            latency_ms,
            {"source": source, "asset": asset},
        )
