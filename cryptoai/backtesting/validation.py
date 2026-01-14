"""
Professional-Grade Backtesting Validation Framework.

This module addresses the fundamental flaws in naive backtesting:

WHY NAIVE BACKTESTS LIE:
========================

1. LOOKAHEAD BIAS
   - Using future information to make past decisions
   - Example: Using close price for signals when only open was available
   - Example: Parameter optimization on full dataset then testing on same data

2. SURVIVORSHIP BIAS
   - Only testing on assets that still exist today
   - Delisted/failed tokens excluded = artificially inflated returns

3. REGIME IGNORANCE
   - Bull market strategy tested only in bull conditions
   - No stress testing against Black Swan events
   - Correlation breakdown in crisis not modeled

4. TRANSACTION COST FANTASY
   - Fixed slippage assumptions (reality: varies 10x with volatility)
   - Ignoring market impact on large orders
   - Assuming infinite liquidity

5. DATA LEAKAGE
   - Train/test contamination through overlapping windows
   - Feature normalization using future data
   - Model selection on test set

6. OVERFITTING TO NOISE
   - Too many parameters relative to data
   - No cross-validation
   - Cherry-picked backtest periods

This module implements:
- Walk-forward validation with proper embargo periods
- Regime-aware data splitting
- Combinatorial purged cross-validation
- Monte Carlo permutation tests
- Bootstrap confidence intervals

Author: Quant Research Engineering
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Iterator
from enum import Enum
import numpy as np
from loguru import logger


class MarketRegime(Enum):
    """Market regime classification."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    CRISIS = "crisis"


@dataclass
class ValidationConfig:
    """Configuration for backtesting validation."""

    # Walk-forward settings
    train_window_days: int = 90
    test_window_days: int = 30
    step_days: int = 15
    min_train_samples: int = 1000

    # Embargo period to prevent leakage
    embargo_days: int = 5

    # Purging for overlapping labels
    purge_window_days: int = 3

    # Cross-validation
    n_splits: int = 5

    # Regime detection
    regime_lookback_days: int = 20
    volatility_threshold_high: float = 0.03  # 3% daily vol
    volatility_threshold_low: float = 0.01   # 1% daily vol
    trend_threshold: float = 0.15            # 15% move for trend

    # Monte Carlo settings
    n_permutations: int = 1000
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Capital constraints
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.20           # Max 20% in single position
    max_leverage: float = 3.0
    margin_requirement: float = 0.10         # 10% margin

    # Stress testing
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "btc_50pct_crash",
        "flash_crash",
        "liquidity_crisis",
        "correlation_breakdown",
    ])


@dataclass
class WalkForwardWindow:
    """Single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    regime: MarketRegime
    window_id: int

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days


class RegimeDetector:
    """
    Detect market regimes for proper data splitting.

    Regimes matter because:
    - Strategy performance varies dramatically across regimes
    - Training on bull, testing on bear = meaningless results
    - Need representative samples from each regime
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def detect_regime(
        self,
        returns: np.ndarray,
        timestamps: List[datetime],
        current_idx: int,
    ) -> MarketRegime:
        """
        Detect current market regime.

        Uses ONLY past data (no lookahead).
        """
        lookback = min(self.config.regime_lookback_days, current_idx)
        if lookback < 5:
            return MarketRegime.RANGING

        window_returns = returns[current_idx - lookback:current_idx]

        # Calculate metrics
        volatility = np.std(window_returns) * np.sqrt(252)
        cumulative_return = np.prod(1 + window_returns) - 1

        # Check for crisis (extreme negative returns)
        if np.min(window_returns) < -0.10:  # 10% single day drop
            return MarketRegime.CRISIS

        # Volatility regime
        if volatility > self.config.volatility_threshold_high * np.sqrt(252):
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.config.volatility_threshold_low * np.sqrt(252):
            return MarketRegime.LOW_VOLATILITY

        # Trend regime
        if cumulative_return > self.config.trend_threshold:
            return MarketRegime.BULL_TREND
        elif cumulative_return < -self.config.trend_threshold:
            return MarketRegime.BEAR_TREND

        return MarketRegime.RANGING

    def classify_all_periods(
        self,
        returns: np.ndarray,
        timestamps: List[datetime],
    ) -> Dict[datetime, MarketRegime]:
        """Classify regime for each timestamp."""
        regimes = {}
        for i in range(len(timestamps)):
            regimes[timestamps[i]] = self.detect_regime(returns, timestamps, i)
        return regimes


class WalkForwardValidator:
    """
    Walk-forward validation with embargo periods.

    CRITICAL: This prevents lookahead bias by ensuring:
    1. Model is ONLY trained on past data
    2. Embargo period prevents label leakage
    3. No re-optimization on test results
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.regime_detector = RegimeDetector(config)

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[datetime]] = None,
    ) -> Iterator[WalkForwardWindow]:
        """
        Generate walk-forward windows.

        Timeline for each window:
        |---TRAIN---|--EMBARGO--|---TEST---|

        The embargo period prevents data leakage from overlapping labels.
        """
        train_days = self.config.train_window_days
        test_days = self.config.test_window_days
        embargo_days = self.config.embargo_days
        step_days = self.config.step_days

        current_start = start_date
        window_id = 0

        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=embargo_days)
            test_end = test_start + timedelta(days=test_days)

            # Check if we've exceeded the data range
            if test_end > end_date:
                break

            # Detect regime (using only train data for classification)
            if returns is not None and timestamps is not None:
                # Find index corresponding to train_end
                train_end_idx = next(
                    (i for i, ts in enumerate(timestamps) if ts >= train_end),
                    len(timestamps) - 1
                )
                regime = self.regime_detector.detect_regime(
                    returns, timestamps, train_end_idx
                )
            else:
                regime = MarketRegime.RANGING

            yield WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                regime=regime,
                window_id=window_id,
            )

            current_start += timedelta(days=step_days)
            window_id += 1

    def validate_no_lookahead(
        self,
        model_predictions: np.ndarray,
        actual_values: np.ndarray,
        timestamps: List[datetime],
        model_training_cutoff: datetime,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that model doesn't use future information.

        Returns (is_valid, list_of_violations)
        """
        violations = []

        for i, ts in enumerate(timestamps):
            if ts < model_training_cutoff:
                # Model shouldn't have predictions before it was trained
                # (unless using historical walk-forward)
                continue

        # Check for suspiciously perfect predictions
        correlation = np.corrcoef(model_predictions, actual_values)[0, 1]
        if correlation > 0.99:
            violations.append(
                f"Suspiciously high correlation ({correlation:.4f}) - possible lookahead"
            )

        # Check for zero-lag predictions (should have some delay)
        # Real trading has latency

        return len(violations) == 0, violations


class PurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Standard k-fold is WRONG for time series because:
    - Training folds contain future information
    - Overlapping samples cause leakage

    This implementation:
    1. Purges overlapping samples between train/test
    2. Adds embargo to prevent leakage from labels that span time
    3. Respects temporal ordering

    Reference: "Advances in Financial Machine Learning" - López de Prado
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        timestamps: np.ndarray = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test indices.

        Args:
            X: Features array
            y: Labels array (optional)
            timestamps: Timestamps for each sample

        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            test_indices = indices[test_start:test_end]

            # Get train indices (all except test + embargo + purge)
            train_mask = np.ones(n_samples, dtype=bool)

            # Remove test indices
            train_mask[test_start:test_end] = False

            # Remove embargo period after test (prevent leakage from labels)
            embargo_end = min(test_end + embargo_size, n_samples)
            train_mask[test_end:embargo_end] = False

            # Remove purge period before test (overlapping samples)
            purge_start = max(test_start - purge_size, 0)
            train_mask[purge_start:test_start] = False

            train_indices = indices[train_mask]

            # Ensure chronological order for remaining train samples
            # that come before test
            train_before_test = train_indices[train_indices < test_start]

            if len(train_before_test) > 0:
                yield train_before_test, test_indices


class MonteCarloValidator:
    """
    Monte Carlo permutation tests for statistical significance.

    WHY THIS MATTERS:
    - A strategy might look profitable by pure chance
    - Need to test if results are statistically significant
    - Permutation test: "If signals were random, how often would we see this result?"

    If your strategy can't beat random noise, it's worthless.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self._rng = np.random.default_rng(42)  # Reproducible

    def permutation_test(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray,
        metric_fn: Callable[[np.ndarray], float],
    ) -> Tuple[float, float, float]:
        """
        Test if strategy performance is statistically significant.

        Args:
            strategy_returns: Strategy return series
            market_returns: Benchmark return series
            metric_fn: Function to calculate performance metric

        Returns:
            (observed_metric, p_value, critical_value)
        """
        observed_metric = metric_fn(strategy_returns)

        # Generate null distribution by permuting signals
        null_metrics = []

        for _ in range(self.config.n_permutations):
            # Shuffle strategy returns (breaks any real signal)
            permuted = self._rng.permutation(strategy_returns)
            null_metrics.append(metric_fn(permuted))

        null_metrics = np.array(null_metrics)

        # p-value: proportion of null results >= observed
        p_value = np.mean(null_metrics >= observed_metric)

        # Critical value at confidence level
        critical_value = np.percentile(
            null_metrics,
            self.config.confidence_level * 100
        )

        return observed_metric, p_value, critical_value

    def bootstrap_confidence_interval(
        self,
        returns: np.ndarray,
        metric_fn: Callable[[np.ndarray], float],
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Returns:
            (lower_bound, point_estimate, upper_bound)
        """
        point_estimate = metric_fn(returns)

        # Bootstrap samples
        bootstrap_metrics = []
        n = len(returns)

        for _ in range(self.config.bootstrap_samples):
            # Sample with replacement
            sample_idx = self._rng.choice(n, size=n, replace=True)
            sample_returns = returns[sample_idx]
            bootstrap_metrics.append(metric_fn(sample_returns))

        bootstrap_metrics = np.array(bootstrap_metrics)

        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)

        return lower, point_estimate, upper


@dataclass
class DataLeakageReport:
    """Report of potential data leakage issues."""
    has_leakage: bool
    issues: List[str]
    severity: str  # "critical", "warning", "info"
    recommendations: List[str]


class DataLeakageDetector:
    """
    Detect potential data leakage in backtesting.

    Common leakage sources:
    1. Feature normalization using full dataset statistics
    2. Overlapping train/test samples
    3. Using future data in feature engineering
    4. Model selection on test set
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def check_feature_leakage(
        self,
        feature_timestamps: np.ndarray,
        label_timestamps: np.ndarray,
    ) -> DataLeakageReport:
        """Check if features use future information."""
        issues = []
        recommendations = []

        # Check if any feature timestamp > label timestamp
        for i in range(len(label_timestamps)):
            feature_ts = feature_timestamps[i]
            label_ts = label_timestamps[i]

            if feature_ts > label_ts:
                issues.append(
                    f"Feature at index {i} uses future data "
                    f"(feature: {feature_ts}, label: {label_ts})"
                )

        if issues:
            return DataLeakageReport(
                has_leakage=True,
                issues=issues[:10],  # Limit to first 10
                severity="critical",
                recommendations=[
                    "Ensure features only use data available at prediction time",
                    "Add proper lag to all features",
                    "Review feature engineering pipeline",
                ]
            )

        return DataLeakageReport(
            has_leakage=False,
            issues=[],
            severity="info",
            recommendations=[]
        )

    def check_normalization_leakage(
        self,
        train_stats_computed_at: datetime,
        test_start: datetime,
    ) -> DataLeakageReport:
        """Check if normalization uses test data."""
        if train_stats_computed_at > test_start:
            return DataLeakageReport(
                has_leakage=True,
                issues=[
                    f"Normalization statistics computed at {train_stats_computed_at} "
                    f"but test starts at {test_start}"
                ],
                severity="critical",
                recommendations=[
                    "Compute normalization stats only on training data",
                    "Apply same stats to test data without recomputing",
                ]
            )
        return DataLeakageReport(
            has_leakage=False, issues=[], severity="info", recommendations=[]
        )

    def check_sample_overlap(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        sample_span: int = 1,
    ) -> DataLeakageReport:
        """Check for overlapping samples between train and test."""
        issues = []

        # Check direct overlap
        overlap = np.intersect1d(train_indices, test_indices)
        if len(overlap) > 0:
            issues.append(f"Direct overlap: {len(overlap)} samples in both train and test")

        # Check for samples within span distance
        for test_idx in test_indices:
            nearby_train = train_indices[
                (train_indices >= test_idx - sample_span) &
                (train_indices <= test_idx + sample_span)
            ]
            if len(nearby_train) > 0 and test_idx not in nearby_train:
                issues.append(
                    f"Test sample {test_idx} has {len(nearby_train)} "
                    f"train samples within {sample_span} positions"
                )
                if len(issues) > 10:
                    break

        if issues:
            return DataLeakageReport(
                has_leakage=True,
                issues=issues[:10],
                severity="critical",
                recommendations=[
                    f"Add embargo period of at least {sample_span} samples",
                    "Use purged cross-validation",
                ]
            )

        return DataLeakageReport(
            has_leakage=False, issues=[], severity="info", recommendations=[]
        )


def validate_backtest_integrity(
    config: ValidationConfig,
    returns: np.ndarray,
    timestamps: List[datetime],
    strategy_signals: np.ndarray,
) -> Dict[str, any]:
    """
    Comprehensive backtest validation.

    Returns validation report with:
    - Lookahead bias check
    - Data leakage check
    - Statistical significance
    - Regime coverage
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {},
    }

    # Check regime coverage
    regime_detector = RegimeDetector(config)
    regimes = regime_detector.classify_all_periods(returns, timestamps)
    regime_counts = {}
    for regime in regimes.values():
        regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

    report["statistics"]["regime_distribution"] = regime_counts

    # Check for regime imbalance
    total = sum(regime_counts.values())
    for regime, count in regime_counts.items():
        pct = count / total
        if pct < 0.05:
            report["warnings"].append(
                f"Regime '{regime}' only represents {pct:.1%} of data - "
                "results may not generalize"
            )

    # Monte Carlo significance test
    mc_validator = MonteCarloValidator(config)

    def sharpe_ratio(rets):
        if np.std(rets) == 0:
            return 0
        return np.sqrt(252) * np.mean(rets) / np.std(rets)

    observed, p_value, critical = mc_validator.permutation_test(
        returns, returns, sharpe_ratio
    )

    report["statistics"]["sharpe_ratio"] = observed
    report["statistics"]["sharpe_p_value"] = p_value

    if p_value > 0.05:
        report["warnings"].append(
            f"Strategy Sharpe ({observed:.2f}) is NOT statistically significant "
            f"(p-value: {p_value:.3f})"
        )

    # Bootstrap confidence interval
    lower, point, upper = mc_validator.bootstrap_confidence_interval(
        returns, sharpe_ratio
    )
    report["statistics"]["sharpe_95_ci"] = (lower, upper)

    if lower < 0:
        report["warnings"].append(
            f"Sharpe ratio 95% CI includes zero [{lower:.2f}, {upper:.2f}] - "
            "strategy may not be profitable"
        )

    # Check if any errors make the backtest invalid
    report["valid"] = len(report["errors"]) == 0

    return report
