"""
Crypto-Realistic Backtesting Engine for CryptoAI.

PROFESSIONAL-GRADE BACKTESTING
==============================

This module provides institutional-quality backtesting that addresses
the fundamental flaws in naive backtesting implementations.

KEY PRINCIPLES:
- NO LOOKAHEAD: Never use future data to make past decisions
- NO LEAKAGE: Strict train/test separation with embargo periods
- REALISTIC EXECUTION: Slippage, fees, latency, partial fills
- STATISTICAL RIGOR: Monte Carlo significance testing

MAIN COMPONENTS:
- ProfessionalBacktester: Walk-forward backtesting engine
- RiskMetricsCalculator: Sharpe, Sortino, CVaR, drawdown analysis
- WalkForwardValidator: Proper time-series cross-validation
- RegimeDetector: Market regime classification
- MonteCarloValidator: Statistical significance testing

USAGE:
------
```python
from cryptoai.backtesting import (
    create_professional_backtester,
    RiskMetricsCalculator,
)

# Create backtester
backtester = create_professional_backtester(
    initial_capital=100_000,
    max_leverage=3.0,
    execution_mode="realistic",
)

# Run walk-forward backtest
result = backtester.run_walk_forward(
    strategy=my_strategy,
    market_data=data,
    timestamps=timestamps,
    start_date=start,
    end_date=end,
)

# Check validity
if result.is_valid:
    print(f"Sharpe: {result.metrics.sharpe_ratio_adjusted:.2f}")
    print(f"p-value: {result.metrics.p_value:.4f}")
else:
    print("WARNINGS:", result.warnings)
```
"""

# Core engine
from cryptoai.backtesting.engine import BacktestEngine, BacktestConfig

# Professional engine
from cryptoai.backtesting.professional_engine import (
    ProfessionalBacktester,
    AdvancedSimConfig,
    ExecutionMode,
    BacktestResult,
    create_professional_backtester,
)

# Simulation
from cryptoai.backtesting.simulator import (
    MarketSimulator,
    SimulationConfig,
    OrderBookSimulator,
)

# Metrics
from cryptoai.backtesting.metrics import BacktestMetrics, PerformanceReport
from cryptoai.backtesting.risk_metrics import (
    RiskMetricsCalculator,
    RiskAdjustedMetrics,
    TailRiskMetrics,
    DrawdownMetrics,
    compare_strategies,
)

# Validation
from cryptoai.backtesting.validation import (
    ValidationConfig,
    WalkForwardValidator,
    WalkForwardWindow,
    RegimeDetector,
    MarketRegime,
    MonteCarloValidator,
    PurgedKFold,
    DataLeakageDetector,
    validate_backtest_integrity,
)

__all__ = [
    # Core
    "BacktestEngine",
    "BacktestConfig",
    # Professional
    "ProfessionalBacktester",
    "AdvancedSimConfig",
    "ExecutionMode",
    "BacktestResult",
    "create_professional_backtester",
    # Simulation
    "MarketSimulator",
    "SimulationConfig",
    "OrderBookSimulator",
    # Metrics
    "BacktestMetrics",
    "PerformanceReport",
    "RiskMetricsCalculator",
    "RiskAdjustedMetrics",
    "TailRiskMetrics",
    "DrawdownMetrics",
    "compare_strategies",
    # Validation
    "ValidationConfig",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "RegimeDetector",
    "MarketRegime",
    "MonteCarloValidator",
    "PurgedKFold",
    "DataLeakageDetector",
    "validate_backtest_integrity",
]
