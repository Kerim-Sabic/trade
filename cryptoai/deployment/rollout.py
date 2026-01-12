"""Rollout management for gradual deployment and A/B testing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import random
import json
import threading
from loguru import logger


class RolloutStrategy(Enum):
    """Rollout strategies."""
    IMMEDIATE = "immediate"  # Full rollout immediately
    CANARY = "canary"  # Gradual percentage rollout
    SHADOW = "shadow"  # Run alongside without affecting production
    AB_TEST = "ab_test"  # A/B test with statistical significance


@dataclass
class RolloutConfig:
    """Configuration for rollout."""

    strategy: RolloutStrategy = RolloutStrategy.SHADOW
    initial_percentage: float = 5.0  # Starting percentage
    max_percentage: float = 100.0  # Maximum percentage
    increment_percentage: float = 5.0  # Increase per step
    increment_interval_hours: float = 1.0  # Time between increments

    # Success criteria
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05
    max_latency_p99_ms: float = 100.0
    min_sample_size: int = 100

    # Rollback triggers
    auto_rollback: bool = True
    rollback_on_error_rate: float = 0.1
    rollback_on_latency_spike: float = 2.0  # multiplier

    # A/B test settings
    ab_significance_level: float = 0.05
    ab_min_effect_size: float = 0.01


@dataclass
class RolloutState:
    """Current rollout state."""

    model_id: str
    strategy: RolloutStrategy
    current_percentage: float
    start_time: datetime
    last_increment_time: Optional[datetime] = None
    is_active: bool = True
    is_completed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class RolloutManager:
    """
    Manages gradual rollout of new model versions.

    Features:
    - Canary deployments
    - A/B testing
    - Shadow mode
    - Automatic rollback
    - Statistical significance testing
    """

    def __init__(self, config: RolloutConfig):
        self.config = config
        self._lock = threading.Lock()

        # Active rollouts
        self._rollouts: Dict[str, RolloutState] = {}
        self._control_model: Optional[str] = None

        # Metrics per model
        self._model_metrics: Dict[str, Dict[str, List]] = {}

        # Callbacks
        self._on_complete: Optional[Callable] = None
        self._on_rollback: Optional[Callable] = None

    def start_rollout(
        self,
        model_id: str,
        strategy: Optional[RolloutStrategy] = None,
        control_model: Optional[str] = None,
    ) -> RolloutState:
        """
        Start a new rollout.

        Args:
            model_id: Model version to roll out
            strategy: Override default strategy
            control_model: Model to compare against (for A/B)

        Returns:
            RolloutState
        """
        strategy = strategy or self.config.strategy

        with self._lock:
            if model_id in self._rollouts:
                raise ValueError(f"Rollout already active for {model_id}")

            state = RolloutState(
                model_id=model_id,
                strategy=strategy,
                current_percentage=self.config.initial_percentage
                if strategy != RolloutStrategy.IMMEDIATE
                else 100.0,
                start_time=datetime.utcnow(),
            )

            self._rollouts[model_id] = state

            if control_model:
                self._control_model = control_model

            self._model_metrics[model_id] = {
                "successes": [],
                "errors": [],
                "latencies": [],
                "rewards": [],
            }

        logger.info(
            f"Started {strategy.value} rollout for {model_id} "
            f"at {state.current_percentage}%"
        )

        return state

    def should_use_model(self, model_id: str) -> bool:
        """
        Determine if a request should use the new model.

        Args:
            model_id: Model to check

        Returns:
            True if request should use this model
        """
        with self._lock:
            if model_id not in self._rollouts:
                return False

            state = self._rollouts[model_id]

            if not state.is_active or state.is_completed:
                return state.is_completed and state.current_percentage == 100.0

            # Random selection based on percentage
            return random.random() * 100 < state.current_percentage

    def record_result(
        self,
        model_id: str,
        success: bool,
        latency_ms: float,
        reward: Optional[float] = None,
    ) -> None:
        """Record a result for the model."""
        with self._lock:
            if model_id not in self._model_metrics:
                return

            metrics = self._model_metrics[model_id]

            if success:
                metrics["successes"].append(1)
                metrics["errors"].append(0)
            else:
                metrics["successes"].append(0)
                metrics["errors"].append(1)

            metrics["latencies"].append(latency_ms)

            if reward is not None:
                metrics["rewards"].append(reward)

        # Check for rollback triggers
        self._check_rollback_triggers(model_id)

    def _check_rollback_triggers(self, model_id: str) -> None:
        """Check if rollback should be triggered."""
        if not self.config.auto_rollback:
            return

        metrics = self._model_metrics.get(model_id, {})

        if len(metrics.get("errors", [])) < self.config.min_sample_size:
            return

        # Calculate error rate
        recent_errors = metrics["errors"][-100:]
        error_rate = sum(recent_errors) / len(recent_errors)

        if error_rate > self.config.rollback_on_error_rate:
            logger.warning(
                f"High error rate ({error_rate:.2%}) for {model_id}, triggering rollback"
            )
            self.rollback(model_id)
            return

        # Check latency
        recent_latencies = metrics["latencies"][-100:]
        baseline_latency = metrics["latencies"][:100]

        if baseline_latency:
            baseline_p99 = sorted(baseline_latency)[int(len(baseline_latency) * 0.99)]
            current_p99 = sorted(recent_latencies)[int(len(recent_latencies) * 0.99)]

            if current_p99 > baseline_p99 * self.config.rollback_on_latency_spike:
                logger.warning(
                    f"Latency spike ({current_p99:.1f}ms vs {baseline_p99:.1f}ms) "
                    f"for {model_id}, triggering rollback"
                )
                self.rollback(model_id)

    def increment_rollout(self, model_id: str) -> bool:
        """
        Increment rollout percentage.

        Returns:
            True if increment was successful
        """
        with self._lock:
            if model_id not in self._rollouts:
                return False

            state = self._rollouts[model_id]

            if not state.is_active or state.is_completed:
                return False

            # Check success criteria
            if not self._check_success_criteria(model_id):
                logger.warning(f"Success criteria not met for {model_id}")
                return False

            # Increment
            new_percentage = min(
                state.current_percentage + self.config.increment_percentage,
                self.config.max_percentage,
            )

            state.current_percentage = new_percentage
            state.last_increment_time = datetime.utcnow()

            logger.info(f"Incremented rollout for {model_id} to {new_percentage}%")

            # Check if complete
            if new_percentage >= self.config.max_percentage:
                state.is_completed = True
                logger.info(f"Rollout complete for {model_id}")

                if self._on_complete:
                    self._on_complete(model_id)

            return True

    def _check_success_criteria(self, model_id: str) -> bool:
        """Check if success criteria are met."""
        metrics = self._model_metrics.get(model_id, {})

        # Check sample size
        if len(metrics.get("successes", [])) < self.config.min_sample_size:
            return False

        recent = metrics["successes"][-self.config.min_sample_size:]
        success_rate = sum(recent) / len(recent)

        if success_rate < self.config.min_success_rate:
            return False

        # Check latency
        recent_latencies = metrics["latencies"][-self.config.min_sample_size:]
        p99_latency = sorted(recent_latencies)[int(len(recent_latencies) * 0.99)]

        if p99_latency > self.config.max_latency_p99_ms:
            return False

        return True

    def rollback(self, model_id: str) -> None:
        """Rollback a model deployment."""
        with self._lock:
            if model_id not in self._rollouts:
                return

            state = self._rollouts[model_id]
            state.is_active = False
            state.current_percentage = 0.0

        logger.warning(f"Rolled back model {model_id}")

        if self._on_rollback:
            self._on_rollback(model_id)

    def complete_rollout(self, model_id: str) -> None:
        """Force complete a rollout."""
        with self._lock:
            if model_id not in self._rollouts:
                return

            state = self._rollouts[model_id]
            state.current_percentage = 100.0
            state.is_completed = True
            state.is_active = False

        logger.info(f"Force completed rollout for {model_id}")

    def run_ab_test(
        self,
        model_a: str,
        model_b: str,
        metric: str = "rewards",
    ) -> Dict[str, Any]:
        """
        Run A/B test between two models.

        Returns statistical comparison.
        """
        from scipy import stats

        metrics_a = self._model_metrics.get(model_a, {}).get(metric, [])
        metrics_b = self._model_metrics.get(model_b, {}).get(metric, [])

        if len(metrics_a) < self.config.min_sample_size:
            return {"error": f"Insufficient samples for {model_a}"}
        if len(metrics_b) < self.config.min_sample_size:
            return {"error": f"Insufficient samples for {model_b}"}

        # T-test
        t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)

        # Effect size (Cohen's d)
        pooled_std = (
            (len(metrics_a) - 1) * (sum((x - sum(metrics_a)/len(metrics_a))**2 for x in metrics_a) / (len(metrics_a) - 1)) +
            (len(metrics_b) - 1) * (sum((x - sum(metrics_b)/len(metrics_b))**2 for x in metrics_b) / (len(metrics_b) - 1))
        ) / (len(metrics_a) + len(metrics_b) - 2)
        pooled_std = pooled_std ** 0.5

        effect_size = (sum(metrics_a)/len(metrics_a) - sum(metrics_b)/len(metrics_b)) / pooled_std if pooled_std > 0 else 0

        # Determine winner
        significant = p_value < self.config.ab_significance_level
        meaningful = abs(effect_size) > self.config.ab_min_effect_size

        if significant and meaningful:
            winner = model_a if sum(metrics_a)/len(metrics_a) > sum(metrics_b)/len(metrics_b) else model_b
        else:
            winner = None

        return {
            "model_a": {
                "id": model_a,
                "mean": sum(metrics_a) / len(metrics_a),
                "std": (sum((x - sum(metrics_a)/len(metrics_a))**2 for x in metrics_a) / len(metrics_a)) ** 0.5,
                "n": len(metrics_a),
            },
            "model_b": {
                "id": model_b,
                "mean": sum(metrics_b) / len(metrics_b),
                "std": (sum((x - sum(metrics_b)/len(metrics_b))**2 for x in metrics_b) / len(metrics_b)) ** 0.5,
                "n": len(metrics_b),
            },
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "significant": significant,
            "meaningful": meaningful,
            "winner": winner,
        }

    def get_rollout_status(self, model_id: str) -> Optional[Dict]:
        """Get current rollout status."""
        with self._lock:
            if model_id not in self._rollouts:
                return None

            state = self._rollouts[model_id]
            metrics = self._model_metrics.get(model_id, {})

            success_rate = 0.0
            if metrics.get("successes"):
                success_rate = sum(metrics["successes"]) / len(metrics["successes"])

            avg_latency = 0.0
            if metrics.get("latencies"):
                avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])

            return {
                "model_id": model_id,
                "strategy": state.strategy.value,
                "current_percentage": state.current_percentage,
                "is_active": state.is_active,
                "is_completed": state.is_completed,
                "start_time": state.start_time.isoformat(),
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "sample_count": len(metrics.get("successes", [])),
            }

    def set_callbacks(
        self,
        on_complete: Optional[Callable] = None,
        on_rollback: Optional[Callable] = None,
    ) -> None:
        """Set rollout callbacks."""
        self._on_complete = on_complete
        self._on_rollback = on_rollback

    def export_results(self, path: str) -> None:
        """Export rollout results."""
        results = {
            "rollouts": {
                model_id: self.get_rollout_status(model_id)
                for model_id in self._rollouts
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Exported rollout results to {path}")
