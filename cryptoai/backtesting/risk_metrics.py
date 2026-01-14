"""
Professional Risk-Adjusted Performance Metrics.

This module implements institutional-grade risk metrics that account for:
- Non-normal return distributions (fat tails)
- Autocorrelation in returns
- Drawdown dynamics
- Tail risk measures

WHY STANDARD METRICS LIE:
========================

1. SHARPE RATIO PROBLEMS:
   - Assumes normally distributed returns (crypto has fat tails)
   - Penalizes upside volatility same as downside
   - Doesn't account for autocorrelation
   - Can be gamed by taking tail risk

2. STANDARD DEVIATION PROBLEMS:
   - Symmetric measure (upside ≠ downside)
   - Underestimates tail risk
   - Historical vol ≠ future vol

3. VAR PROBLEMS:
   - Tells you nothing about losses beyond VaR
   - Not subadditive (can increase with diversification!)
   - Historical VaR misses regime changes

This module provides:
- Autocorrelation-adjusted Sharpe
- Probabilistic Sharpe Ratio (is it statistically significant?)
- Sortino with proper downside deviation
- CVaR (Expected Shortfall) - coherent risk measure
- Tail risk metrics (kurtosis, skewness)
- Drawdown-based metrics

Author: Quant Research Engineering
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class TailRiskMetrics:
    """Tail risk analysis results."""
    var_95: float           # 95% Value at Risk (daily)
    var_99: float           # 99% Value at Risk (daily)
    cvar_95: float          # 95% Conditional VaR (Expected Shortfall)
    cvar_99: float          # 99% Conditional VaR
    skewness: float         # Return distribution skewness
    kurtosis: float         # Excess kurtosis (0 = normal)
    tail_ratio: float       # Right tail / Left tail (>1 = positive skew)
    max_loss_1d: float      # Maximum single-day loss
    max_loss_5d: float      # Maximum 5-day loss
    extreme_loss_prob: float  # P(loss > 3 std)


@dataclass
class DrawdownMetrics:
    """Drawdown analysis results."""
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown_duration_days: float
    current_drawdown: float
    underwater_pct: float  # % of time in drawdown
    recovery_factor: float  # Total return / Max drawdown
    ulcer_index: float     # Quadratic mean of drawdowns


@dataclass
class RiskAdjustedMetrics:
    """Complete risk-adjusted performance analysis."""

    # Return metrics
    total_return: float
    cagr: float  # Compound Annual Growth Rate
    daily_return_mean: float
    daily_return_std: float

    # Risk-adjusted returns (corrected)
    sharpe_ratio: float
    sharpe_ratio_adjusted: float  # Autocorrelation adjusted
    probabilistic_sharpe: float   # Statistical significance
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Tail risk
    tail_risk: TailRiskMetrics

    # Drawdown
    drawdown: DrawdownMetrics

    # Statistical tests
    is_significant: bool
    p_value: float
    confidence_interval_95: Tuple[float, float]


class RiskMetricsCalculator:
    """
    Calculate professional-grade risk metrics.

    All calculations are:
    - Statistically rigorous
    - Appropriate for non-normal distributions
    - Account for autocorrelation
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
    ):
        self.rf = risk_free_rate
        self.ann_factor = annualization_factor

    def calculate_all(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> RiskAdjustedMetrics:
        """Calculate all risk-adjusted metrics."""

        if len(returns) < 30:
            logger.warning("Less than 30 observations - metrics may be unreliable")

        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        n_days = len(returns)
        n_years = n_days / self.ann_factor
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        daily_mean = np.mean(returns)
        daily_std = np.std(returns, ddof=1)

        # Risk-adjusted metrics
        sharpe = self._sharpe_ratio(returns)
        sharpe_adj = self._autocorr_adjusted_sharpe(returns)
        prob_sharpe = self._probabilistic_sharpe(returns, benchmark_sharpe=0.0)
        sortino = self._sortino_ratio(returns)
        calmar = self._calmar_ratio(returns)
        omega = self._omega_ratio(returns)

        # Tail risk
        tail_risk = self._tail_risk_metrics(returns)

        # Drawdown metrics
        drawdown = self._drawdown_metrics(returns)

        # Statistical significance
        is_sig, p_val = self._test_significance(returns)
        ci_low, ci_high = self._bootstrap_sharpe_ci(returns)

        return RiskAdjustedMetrics(
            total_return=total_return,
            cagr=cagr,
            daily_return_mean=daily_mean,
            daily_return_std=daily_std,
            sharpe_ratio=sharpe,
            sharpe_ratio_adjusted=sharpe_adj,
            probabilistic_sharpe=prob_sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            tail_risk=tail_risk,
            drawdown=drawdown,
            is_significant=is_sig,
            p_value=p_val,
            confidence_interval_95=(ci_low, ci_high),
        )

    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Standard Sharpe ratio."""
        excess = returns - self.rf / self.ann_factor
        if np.std(excess) == 0:
            return 0.0
        return np.sqrt(self.ann_factor) * np.mean(excess) / np.std(excess, ddof=1)

    def _autocorr_adjusted_sharpe(self, returns: np.ndarray) -> float:
        """
        Sharpe ratio adjusted for autocorrelation.

        Standard Sharpe overstates significance when returns are autocorrelated.
        This adjustment accounts for serial correlation in returns.

        Reference: Lo (2002) "The Statistics of Sharpe Ratios"
        """
        n = len(returns)
        if n < 10:
            return self._sharpe_ratio(returns)

        excess = returns - self.rf / self.ann_factor
        mean_ret = np.mean(excess)
        std_ret = np.std(excess, ddof=1)

        if std_ret == 0:
            return 0.0

        # Calculate autocorrelations
        max_lag = min(n // 4, 10)
        autocorrs = []

        for lag in range(1, max_lag + 1):
            if lag < n:
                corr = np.corrcoef(excess[:-lag], excess[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(corr)

        if not autocorrs:
            return self._sharpe_ratio(returns)

        # Adjustment factor
        rho_sum = sum(
            (1 - k / (max_lag + 1)) * autocorrs[k - 1]
            for k in range(1, len(autocorrs) + 1)
        )
        adjustment = 1 + 2 * rho_sum

        if adjustment <= 0:
            adjustment = 1  # Fallback

        # Adjusted Sharpe
        unadjusted_sharpe = np.sqrt(self.ann_factor) * mean_ret / std_ret
        adjusted_sharpe = unadjusted_sharpe / np.sqrt(adjustment)

        return adjusted_sharpe

    def _probabilistic_sharpe(
        self,
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0,
    ) -> float:
        """
        Probabilistic Sharpe Ratio.

        Answers: "What's the probability that the true Sharpe > benchmark?"

        This accounts for:
        - Estimation error in Sharpe
        - Non-normality (skew/kurtosis)
        - Sample size

        Reference: Bailey & López de Prado (2012)
        """
        n = len(returns)
        if n < 10:
            return 0.5

        sharpe = self._sharpe_ratio(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis

        # Standard error of Sharpe ratio
        se = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4) * sharpe**2) / (n - 1)
        )

        if se == 0:
            return 0.5

        # Z-score
        z = (sharpe - benchmark_sharpe) / se

        # Probability (one-sided)
        prob = stats.norm.cdf(z)

        return prob

    def _sortino_ratio(self, returns: np.ndarray, mar: float = 0.0) -> float:
        """
        Sortino ratio with proper downside deviation.

        Unlike Sharpe, only penalizes downside volatility.
        MAR = Minimum Acceptable Return (usually 0 or risk-free rate)
        """
        excess = returns - mar

        # Downside returns only
        downside = np.minimum(excess, 0)
        downside_std = np.sqrt(np.mean(downside**2))

        if downside_std == 0:
            return 0.0 if np.mean(excess) <= 0 else np.inf

        return np.sqrt(self.ann_factor) * np.mean(excess) / downside_std

    def _calmar_ratio(self, returns: np.ndarray) -> float:
        """
        Calmar ratio: CAGR / Max Drawdown.

        Better than Sharpe for strategies with large drawdowns.
        """
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / running_max
        max_dd = np.max(drawdowns)

        if max_dd == 0:
            return np.inf

        # CAGR
        total_return = cum_returns[-1] - 1
        n_years = len(returns) / self.ann_factor
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        return cagr / max_dd

    def _omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """
        Omega ratio: Probability-weighted gain/loss ratio.

        Omega = ∫[threshold,∞] (1-F(x))dx / ∫[-∞,threshold] F(x)dx

        Where F is the CDF of returns.

        Better than Sharpe because:
        - Uses entire return distribution
        - No normality assumption
        - Accounts for higher moments
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        total_gain = np.sum(gains) if len(gains) > 0 else 0
        total_loss = np.sum(losses) if len(losses) > 0 else 0

        if total_loss == 0:
            return np.inf if total_gain > 0 else 1.0

        return total_gain / total_loss

    def _tail_risk_metrics(self, returns: np.ndarray) -> TailRiskMetrics:
        """Calculate comprehensive tail risk metrics."""
        n = len(returns)

        # VaR (Historical)
        var_95 = -np.percentile(returns, 5)
        var_99 = -np.percentile(returns, 1)

        # CVaR (Expected Shortfall)
        var_95_idx = int(0.05 * n)
        var_99_idx = int(0.01 * n)
        sorted_returns = np.sort(returns)

        cvar_95 = -np.mean(sorted_returns[:max(var_95_idx, 1)])
        cvar_99 = -np.mean(sorted_returns[:max(var_99_idx, 1)])

        # Distribution shape
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess (normal = 0)

        # Tail ratio (right tail / left tail)
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        median = np.median(returns)

        if abs(median - p5) > 1e-10:
            tail_ratio = (p95 - median) / abs(median - p5)
        else:
            tail_ratio = 1.0

        # Maximum losses
        max_loss_1d = -np.min(returns)

        # 5-day rolling returns
        if n >= 5:
            rolling_5d = np.array([
                np.prod(1 + returns[i:i+5]) - 1
                for i in range(n - 4)
            ])
            max_loss_5d = -np.min(rolling_5d)
        else:
            max_loss_5d = max_loss_1d

        # Extreme loss probability
        std = np.std(returns)
        threshold = -3 * std
        extreme_loss_prob = np.mean(returns < threshold)

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            max_loss_1d=max_loss_1d,
            max_loss_5d=max_loss_5d,
            extreme_loss_prob=extreme_loss_prob,
        )

    def _drawdown_metrics(self, returns: np.ndarray) -> DrawdownMetrics:
        """Calculate comprehensive drawdown metrics."""
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / running_max

        max_dd = np.max(drawdowns)
        avg_dd = np.mean(drawdowns)
        current_dd = drawdowns[-1] if len(drawdowns) > 0 else 0

        # Underwater percentage
        underwater_pct = np.mean(drawdowns > 0)

        # Drawdown durations
        dd_durations = []
        current_duration = 0

        for dd in drawdowns:
            if dd > 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            dd_durations.append(current_duration)

        max_dd_duration = max(dd_durations) if dd_durations else 0
        avg_dd_duration = np.mean(dd_durations) if dd_durations else 0

        # Recovery factor
        total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0
        recovery_factor = total_return / max_dd if max_dd > 0 else np.inf

        # Ulcer Index (quadratic mean of drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdowns**2))

        return DrawdownMetrics(
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown_duration_days=avg_dd_duration,
            current_drawdown=current_dd,
            underwater_pct=underwater_pct,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
        )

    def _test_significance(self, returns: np.ndarray) -> Tuple[bool, float]:
        """Test if returns are statistically significant from zero."""
        if len(returns) < 10:
            return False, 1.0

        # t-test against zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        # One-sided test (we care if returns > 0)
        p_value_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        is_significant = p_value_one_sided < 0.05

        return is_significant, p_value_one_sided

    def _bootstrap_sharpe_ci(
        self,
        returns: np.ndarray,
        n_samples: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for Sharpe ratio."""
        if len(returns) < 10:
            return (-np.inf, np.inf)

        rng = np.random.default_rng(42)
        bootstrap_sharpes = []

        for _ in range(n_samples):
            sample_idx = rng.choice(len(returns), size=len(returns), replace=True)
            sample_returns = returns[sample_idx]
            bootstrap_sharpes.append(self._sharpe_ratio(sample_returns))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
        upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

        return lower, upper


def compare_strategies(
    strategy_returns: Dict[str, np.ndarray],
    benchmark_returns: Optional[np.ndarray] = None,
) -> Dict[str, RiskAdjustedMetrics]:
    """
    Compare multiple strategies with comprehensive metrics.

    Args:
        strategy_returns: Dict of strategy_name -> return series
        benchmark_returns: Optional benchmark for comparison

    Returns:
        Dict of strategy_name -> RiskAdjustedMetrics
    """
    calculator = RiskMetricsCalculator()
    results = {}

    for name, returns in strategy_returns.items():
        results[name] = calculator.calculate_all(returns, benchmark_returns)

    # Rank strategies
    logger.info("\n=== Strategy Comparison ===")
    logger.info(f"{'Strategy':<20} {'Sharpe':<10} {'Sortino':<10} "
                f"{'MaxDD':<10} {'Calmar':<10} {'Sig?':<8}")
    logger.info("-" * 68)

    for name, metrics in sorted(
        results.items(),
        key=lambda x: x[1].sharpe_ratio_adjusted,
        reverse=True
    ):
        sig_str = "Yes" if metrics.is_significant else "No"
        logger.info(
            f"{name:<20} {metrics.sharpe_ratio_adjusted:<10.2f} "
            f"{metrics.sortino_ratio:<10.2f} "
            f"{metrics.drawdown.max_drawdown:<10.2%} "
            f"{metrics.calmar_ratio:<10.2f} {sig_str:<8}"
        )

    return results
