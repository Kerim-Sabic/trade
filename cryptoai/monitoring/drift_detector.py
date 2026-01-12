"""Drift detection for monitoring model and data distribution changes."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
import numpy as np
from scipy import stats
from loguru import logger


class DriftType(Enum):
    """Types of drift."""
    DATA_DRIFT = "data_drift"  # Input distribution shift
    CONCEPT_DRIFT = "concept_drift"  # Target distribution shift
    PREDICTION_DRIFT = "prediction_drift"  # Model output shift
    PERFORMANCE_DRIFT = "performance_drift"  # Performance degradation


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_type: DriftType
    detected: bool
    score: float  # Drift magnitude (0-1)
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Reference window
    reference_window_size: int = 10000
    detection_window_size: int = 1000

    # Thresholds
    ks_test_threshold: float = 0.05  # p-value threshold
    psi_threshold: float = 0.1  # Population Stability Index threshold
    performance_threshold: float = 0.1  # 10% performance drop

    # Detection frequency
    detection_interval_seconds: float = 300.0  # 5 minutes

    # Alert settings
    alert_on_drift: bool = True
    consecutive_drift_threshold: int = 3


class DriftDetector:
    """
    Detects distribution drift in model inputs and outputs.

    Monitors for:
    1. Data drift - Changes in input feature distributions
    2. Concept drift - Changes in the relationship between features and target
    3. Prediction drift - Changes in model output distributions
    4. Performance drift - Degradation in model performance metrics
    """

    def __init__(self, config: DriftConfig):
        self.config = config
        self._lock = threading.Lock()

        # Reference distributions (baseline)
        self._reference_data: Dict[str, np.ndarray] = {}
        self._reference_predictions: Optional[np.ndarray] = None
        self._reference_performance: Dict[str, float] = {}

        # Current windows
        self._current_data: Dict[str, deque] = {}
        self._current_predictions: deque = deque(maxlen=config.detection_window_size)
        self._current_performance: Dict[str, deque] = {}

        # Drift history
        self._drift_history: List[DriftResult] = []
        self._consecutive_drifts: Dict[DriftType, int] = {dt: 0 for dt in DriftType}

        # Feature names for tracking
        self._feature_names: List[str] = []

    def set_reference(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference distribution from training/baseline data.

        Args:
            data: Reference data array (N, F)
            predictions: Reference predictions (N,)
            feature_names: Names of features
        """
        with self._lock:
            n_features = data.shape[1] if data.ndim > 1 else 1

            if feature_names:
                self._feature_names = feature_names
            else:
                self._feature_names = [f"feature_{i}" for i in range(n_features)]

            # Store reference for each feature
            for i, name in enumerate(self._feature_names):
                feature_data = data[:, i] if data.ndim > 1 else data
                self._reference_data[name] = feature_data
                self._current_data[name] = deque(maxlen=self.config.detection_window_size)

            if predictions is not None:
                self._reference_predictions = predictions

        logger.info(f"Set reference distribution with {len(data)} samples, {len(self._feature_names)} features")

    def add_observation(
        self,
        data: np.ndarray,
        prediction: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Add new observation for drift monitoring.

        Args:
            data: Feature values (F,)
            prediction: Model prediction
            metrics: Performance metrics
        """
        with self._lock:
            # Add feature values
            for i, name in enumerate(self._feature_names):
                value = data[i] if len(data) > i else 0
                self._current_data[name].append(value)

            # Add prediction
            if prediction is not None:
                self._current_predictions.append(prediction)

            # Add metrics
            if metrics:
                for metric_name, value in metrics.items():
                    if metric_name not in self._current_performance:
                        self._current_performance[metric_name] = deque(
                            maxlen=self.config.detection_window_size
                        )
                    self._current_performance[metric_name].append(value)

    def detect_data_drift(self) -> DriftResult:
        """
        Detect drift in input feature distributions.

        Uses Kolmogorov-Smirnov test and Population Stability Index.
        """
        drifted_features = []
        overall_psi = 0.0
        min_p_value = 1.0

        with self._lock:
            for name in self._feature_names:
                if name not in self._reference_data or name not in self._current_data:
                    continue

                reference = self._reference_data[name]
                current = np.array(self._current_data[name])

                if len(current) < self.config.detection_window_size // 2:
                    continue

                # KS test
                ks_stat, p_value = stats.ks_2samp(reference, current)
                min_p_value = min(min_p_value, p_value)

                # PSI calculation
                psi = self._calculate_psi(reference, current)
                overall_psi += psi

                if p_value < self.config.ks_test_threshold or psi > self.config.psi_threshold:
                    drifted_features.append({
                        "feature": name,
                        "ks_statistic": ks_stat,
                        "p_value": p_value,
                        "psi": psi,
                    })

        overall_psi /= max(len(self._feature_names), 1)
        drift_detected = len(drifted_features) > 0

        result = DriftResult(
            drift_type=DriftType.DATA_DRIFT,
            detected=drift_detected,
            score=overall_psi,
            p_value=min_p_value,
            details={
                "drifted_features": drifted_features,
                "total_features": len(self._feature_names),
                "drift_ratio": len(drifted_features) / max(len(self._feature_names), 1),
            },
        )

        self._record_drift(result)
        return result

    def detect_prediction_drift(self) -> DriftResult:
        """Detect drift in model predictions."""
        with self._lock:
            if self._reference_predictions is None:
                return DriftResult(
                    drift_type=DriftType.PREDICTION_DRIFT,
                    detected=False,
                    score=0.0,
                    details={"error": "No reference predictions set"},
                )

            current = np.array(self._current_predictions)

            if len(current) < self.config.detection_window_size // 2:
                return DriftResult(
                    drift_type=DriftType.PREDICTION_DRIFT,
                    detected=False,
                    score=0.0,
                    details={"error": "Insufficient current predictions"},
                )

            # KS test
            ks_stat, p_value = stats.ks_2samp(self._reference_predictions, current)

            # PSI
            psi = self._calculate_psi(self._reference_predictions, current)

            drift_detected = p_value < self.config.ks_test_threshold or psi > self.config.psi_threshold

        result = DriftResult(
            drift_type=DriftType.PREDICTION_DRIFT,
            detected=drift_detected,
            score=psi,
            p_value=p_value,
            details={
                "ks_statistic": ks_stat,
                "reference_mean": float(self._reference_predictions.mean()),
                "current_mean": float(current.mean()),
                "reference_std": float(self._reference_predictions.std()),
                "current_std": float(current.std()),
            },
        )

        self._record_drift(result)
        return result

    def detect_performance_drift(
        self,
        metric_name: str = "reward",
    ) -> DriftResult:
        """Detect degradation in model performance."""
        with self._lock:
            if metric_name not in self._current_performance:
                return DriftResult(
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    detected=False,
                    score=0.0,
                    details={"error": f"Metric not tracked: {metric_name}"},
                )

            current_values = list(self._current_performance[metric_name])

            if len(current_values) < self.config.detection_window_size // 2:
                return DriftResult(
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    detected=False,
                    score=0.0,
                    details={"error": "Insufficient performance data"},
                )

            # Split into early and recent
            mid = len(current_values) // 2
            early = current_values[:mid]
            recent = current_values[mid:]

            early_mean = np.mean(early)
            recent_mean = np.mean(recent)

            # Relative change
            if abs(early_mean) > 1e-6:
                relative_change = (recent_mean - early_mean) / abs(early_mean)
            else:
                relative_change = 0.0

            # Statistical test
            t_stat, p_value = stats.ttest_ind(early, recent)

            drift_detected = (
                abs(relative_change) > self.config.performance_threshold and
                p_value < self.config.ks_test_threshold
            )

        result = DriftResult(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            detected=drift_detected,
            score=abs(relative_change),
            p_value=p_value,
            details={
                "metric": metric_name,
                "early_mean": float(early_mean),
                "recent_mean": float(recent_mean),
                "relative_change": float(relative_change),
                "t_statistic": float(t_stat),
            },
        )

        self._record_drift(result)
        return result

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI measures the shift between two distributions.
        PSI < 0.1: No significant shift
        PSI 0.1-0.25: Moderate shift
        PSI > 0.25: Significant shift
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        ref_props = ref_counts / len(reference) + 1e-6
        curr_props = curr_counts / len(current) + 1e-6

        # PSI formula
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return float(psi)

    def _record_drift(self, result: DriftResult) -> None:
        """Record drift result and update consecutive count."""
        self._drift_history.append(result)

        # Keep history bounded
        if len(self._drift_history) > 1000:
            self._drift_history = self._drift_history[-500:]

        # Update consecutive drift count
        if result.detected:
            self._consecutive_drifts[result.drift_type] += 1
        else:
            self._consecutive_drifts[result.drift_type] = 0

    def detect_all(self) -> Dict[DriftType, DriftResult]:
        """Run all drift detection methods."""
        results = {}

        results[DriftType.DATA_DRIFT] = self.detect_data_drift()
        results[DriftType.PREDICTION_DRIFT] = self.detect_prediction_drift()
        results[DriftType.PERFORMANCE_DRIFT] = self.detect_performance_drift()

        return results

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        results = self.detect_all()

        return {
            "data_drift": {
                "detected": results[DriftType.DATA_DRIFT].detected,
                "score": results[DriftType.DATA_DRIFT].score,
                "consecutive": self._consecutive_drifts[DriftType.DATA_DRIFT],
            },
            "prediction_drift": {
                "detected": results[DriftType.PREDICTION_DRIFT].detected,
                "score": results[DriftType.PREDICTION_DRIFT].score,
                "consecutive": self._consecutive_drifts[DriftType.PREDICTION_DRIFT],
            },
            "performance_drift": {
                "detected": results[DriftType.PERFORMANCE_DRIFT].detected,
                "score": results[DriftType.PERFORMANCE_DRIFT].score,
                "consecutive": self._consecutive_drifts[DriftType.PERFORMANCE_DRIFT],
            },
            "requires_retraining": self._should_retrain(),
        }

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained based on drift."""
        for drift_type, count in self._consecutive_drifts.items():
            if count >= self.config.consecutive_drift_threshold:
                return True
        return False

    def get_drift_history(
        self,
        drift_type: Optional[DriftType] = None,
        hours: int = 24,
    ) -> List[DriftResult]:
        """Get drift detection history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            history = [
                r for r in self._drift_history
                if r.timestamp > cutoff
            ]

            if drift_type:
                history = [r for r in history if r.drift_type == drift_type]

        return history

    def reset_reference(self) -> None:
        """Reset reference distribution using current data."""
        with self._lock:
            for name, current in self._current_data.items():
                if len(current) > 0:
                    self._reference_data[name] = np.array(current)

            if len(self._current_predictions) > 0:
                self._reference_predictions = np.array(self._current_predictions)

            # Reset consecutive counts
            self._consecutive_drifts = {dt: 0 for dt in DriftType}

        logger.info("Reset reference distributions from current data")
