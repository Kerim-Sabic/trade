"""Backtest performance metrics and analysis."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    # Returns
    total_return: float
    annualized_return: float
    daily_returns_mean: float
    daily_returns_std: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    var_95: float
    cvar_95: float

    # Trading
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float

    # Costs
    total_fees: float
    total_slippage: float
    total_funding: float

    # Recovery
    time_to_recovery_days: Optional[int]
    capital_survival_rate: float

    # Time analysis
    best_month_return: float
    worst_month_return: float
    profitable_months_pct: float


@dataclass
class DrawdownPeriod:
    """Drawdown period information."""

    start_date: datetime
    end_date: Optional[datetime]
    trough_date: datetime
    depth: float
    duration_days: int
    recovered: bool


class BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Follows crypto-realistic evaluation:
    - Sharpe / Sortino
    - Max drawdown
    - CVaR
    - Time-to-recovery
    - Capital survival rate
    """

    def __init__(
        self,
        equity_history: List[Tuple[datetime, float]],
        trade_history: List[Dict],
        initial_capital: float,
        risk_free_rate: float = 0.0,
    ):
        self.equity_history = equity_history
        self.trade_history = trade_history
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Calculate derived data
        self._calculate_returns()
        self._calculate_drawdowns()

    def _calculate_returns(self):
        """Calculate return series."""
        equities = [e[1] for e in self.equity_history]

        if len(equities) < 2:
            self.returns = np.array([])
            return

        self.returns = np.diff(equities) / equities[:-1]

        # Daily returns (assuming minute data)
        # Group by day and calculate daily return
        daily_equities = {}
        for ts, eq in self.equity_history:
            day = ts.date()
            daily_equities[day] = eq

        days = sorted(daily_equities.keys())
        if len(days) < 2:
            self.daily_returns = np.array([])
            return

        daily_eq = [daily_equities[d] for d in days]
        self.daily_returns = np.diff(daily_eq) / daily_eq[:-1]

    def _calculate_drawdowns(self):
        """Calculate drawdown series and periods."""
        equities = np.array([e[1] for e in self.equity_history])
        timestamps = [e[0] for e in self.equity_history]

        if len(equities) == 0:
            self.drawdowns = np.array([])
            self.drawdown_periods = []
            return

        # Running maximum
        running_max = np.maximum.accumulate(equities)
        self.drawdowns = (running_max - equities) / running_max

        # Identify drawdown periods
        self.drawdown_periods: List[DrawdownPeriod] = []
        in_drawdown = False
        current_period = None

        for i, dd in enumerate(self.drawdowns):
            if dd > 0 and not in_drawdown:
                # Start new drawdown
                in_drawdown = True
                current_period = {
                    "start_idx": i,
                    "start_date": timestamps[i],
                    "max_dd": dd,
                    "trough_idx": i,
                }
            elif in_drawdown:
                if dd > current_period["max_dd"]:
                    current_period["max_dd"] = dd
                    current_period["trough_idx"] = i

                if dd == 0:
                    # Recovered
                    in_drawdown = False
                    trough_date = timestamps[current_period["trough_idx"]]
                    self.drawdown_periods.append(DrawdownPeriod(
                        start_date=current_period["start_date"],
                        end_date=timestamps[i],
                        trough_date=trough_date,
                        depth=current_period["max_dd"],
                        duration_days=(timestamps[i] - current_period["start_date"]).days,
                        recovered=True,
                    ))

        # Handle ongoing drawdown
        if in_drawdown and current_period:
            trough_date = timestamps[current_period["trough_idx"]]
            self.drawdown_periods.append(DrawdownPeriod(
                start_date=current_period["start_date"],
                end_date=None,
                trough_date=trough_date,
                depth=current_period["max_dd"],
                duration_days=(timestamps[-1] - current_period["start_date"]).days,
                recovered=False,
            ))

    @classmethod
    def from_equity_history(
        cls,
        equity_history: List[Tuple[datetime, float]],
        trade_history: List[Dict],
        initial_capital: float,
    ) -> "BacktestMetrics":
        """Create metrics from equity history."""
        return cls(equity_history, trade_history, initial_capital)

    def get_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        # Basic returns
        if len(self.equity_history) < 2:
            return self._empty_report()

        final_equity = self.equity_history[-1][1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (self.equity_history[-1][0] - self.equity_history[0][0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0

        # Daily return stats
        if len(self.daily_returns) > 0:
            daily_mean = np.mean(self.daily_returns)
            daily_std = np.std(self.daily_returns)
        else:
            daily_mean = 0.0
            daily_std = 1.0

        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe()
        sortino = self._calculate_sortino()
        calmar = self._calculate_calmar()

        # Risk metrics
        max_dd = np.max(self.drawdowns) if len(self.drawdowns) > 0 else 0.0
        avg_dd = np.mean(self.drawdowns) if len(self.drawdowns) > 0 else 0.0
        max_dd_duration = max(
            (p.duration_days for p in self.drawdown_periods), default=0
        )
        var_95, cvar_95 = self._calculate_var_cvar()

        # Trading metrics
        trade_metrics = self._calculate_trade_metrics()

        # Cost analysis
        total_fees = sum(t.get("fees", 0) for t in self.trade_history)
        total_slippage = sum(t.get("slippage", 0) for t in self.trade_history)
        total_funding = 0.0  # Would need to be tracked separately

        # Recovery
        ttr = self._calculate_time_to_recovery()
        survival = self._calculate_survival_rate()

        # Monthly analysis
        monthly = self._calculate_monthly_metrics()

        return PerformanceReport(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns_mean=daily_mean,
            daily_returns_std=daily_std,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=trade_metrics["total_trades"],
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            avg_trade_pnl=trade_metrics["avg_pnl"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
            total_fees=total_fees,
            total_slippage=total_slippage,
            total_funding=total_funding,
            time_to_recovery_days=ttr,
            capital_survival_rate=survival,
            best_month_return=monthly["best"],
            worst_month_return=monthly["worst"],
            profitable_months_pct=monthly["profitable_pct"],
        )

    def _empty_report(self) -> PerformanceReport:
        """Return empty report."""
        return PerformanceReport(
            total_return=0, annualized_return=0, daily_returns_mean=0,
            daily_returns_std=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, avg_drawdown=0, max_drawdown_duration_days=0,
            var_95=0, cvar_95=0, total_trades=0, win_rate=0, profit_factor=0,
            avg_trade_pnl=0, avg_win=0, avg_loss=0, total_fees=0, total_slippage=0,
            total_funding=0, time_to_recovery_days=None, capital_survival_rate=1.0,
            best_month_return=0, worst_month_return=0, profitable_months_pct=0,
        )

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_returns) < 2:
            return 0.0

        excess_returns = self.daily_returns - self.risk_free_rate / 252
        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_sortino(self) -> float:
        """Calculate Sortino ratio."""
        if len(self.daily_returns) < 2:
            return 0.0

        excess_returns = self.daily_returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    def _calculate_calmar(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        max_dd = np.max(self.drawdowns) if len(self.drawdowns) > 0 else 0.0
        if max_dd == 0:
            return 0.0

        final_equity = self.equity_history[-1][1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        days = (self.equity_history[-1][0] - self.equity_history[0][0]).days
        annualized = (1 + total_return) ** (365 / max(days, 1)) - 1

        return annualized / max_dd

    def _calculate_var_cvar(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate VaR and CVaR (Expected Shortfall)."""
        if len(self.daily_returns) < 10:
            return 0.0, 0.0

        sorted_returns = np.sort(self.daily_returns)
        var_idx = int((1 - confidence) * len(sorted_returns))

        var = -sorted_returns[var_idx]
        cvar = -np.mean(sorted_returns[:var_idx + 1])

        return var, cvar

    def _calculate_trade_metrics(self) -> Dict:
        """Calculate trading metrics."""
        if len(self.trade_history) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        # This would need actual P&L per trade tracked
        total_trades = len(self.trade_history)

        # Placeholder - would need actual trade P&L
        return {
            "total_trades": total_trades,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    def _calculate_time_to_recovery(self) -> Optional[int]:
        """Calculate time to recover from max drawdown."""
        for period in self.drawdown_periods:
            if period.depth == np.max(self.drawdowns) and period.recovered:
                return period.duration_days
        return None

    def _calculate_survival_rate(self) -> float:
        """Calculate capital survival rate (final capital / initial)."""
        if len(self.equity_history) == 0:
            return 1.0
        return self.equity_history[-1][1] / self.initial_capital

    def _calculate_monthly_metrics(self) -> Dict:
        """Calculate monthly return metrics."""
        if len(self.equity_history) < 30:
            return {"best": 0.0, "worst": 0.0, "profitable_pct": 0.0}

        # Group by month
        monthly_equities = {}
        for ts, eq in self.equity_history:
            key = (ts.year, ts.month)
            if key not in monthly_equities:
                monthly_equities[key] = []
            monthly_equities[key].append(eq)

        # Calculate monthly returns
        months = sorted(monthly_equities.keys())
        monthly_returns = []

        for i in range(1, len(months)):
            prev_eq = monthly_equities[months[i-1]][-1]
            curr_eq = monthly_equities[months[i]][-1]
            ret = (curr_eq - prev_eq) / prev_eq
            monthly_returns.append(ret)

        if len(monthly_returns) == 0:
            return {"best": 0.0, "worst": 0.0, "profitable_pct": 0.0}

        return {
            "best": max(monthly_returns),
            "worst": min(monthly_returns),
            "profitable_pct": sum(1 for r in monthly_returns if r > 0) / len(monthly_returns),
        }
