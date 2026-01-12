"""
Crypto-Realistic Backtesting Engine for CryptoAI.

Event-driven simulation with:
- Exchange-specific fees
- Funding rates
- Partial fills
- Latency simulation
- Slippage from order book depth
- Liquidation mechanics

Rules:
- No lookahead
- No retraining in test windows
- Strict walk-forward
"""

from cryptoai.backtesting.engine import BacktestEngine
from cryptoai.backtesting.simulator import MarketSimulator
from cryptoai.backtesting.metrics import BacktestMetrics

__all__ = [
    "BacktestEngine",
    "MarketSimulator",
    "BacktestMetrics",
]
