"""
Risk Engine for CryptoAI.

Capital preservation > profit.

Dynamic controls:
- Volatility-scaled exposure
- Event-aware leverage caps
- Liquidity-aware sizing
- Kill-switch logic
"""

from cryptoai.risk_engine.controller import RiskController
from cryptoai.risk_engine.position_sizer import PositionSizer
from cryptoai.risk_engine.kill_switch import KillSwitch

__all__ = [
    "RiskController",
    "PositionSizer",
    "KillSwitch",
]
