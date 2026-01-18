"""
Risk Engine for CryptoAI.

Capital preservation > profit.

Dynamic controls:
- Volatility-scaled exposure
- CVaR-aware position sizing
- Dynamic drawdown management
- Event-aware leverage caps
- Liquidity-aware sizing
- Kill-switch logic
- Explicit "do nothing" conditions
"""

from cryptoai.risk_engine.controller import RiskController
from cryptoai.risk_engine.position_sizer import PositionSizer
from cryptoai.risk_engine.kill_switch import KillSwitch
from cryptoai.risk_engine.cvar_position_sizer import (
    CVaRPositionSizer,
    CVaRCalculator,
    DynamicDrawdownManager,
    InactionThreshold,
    ActionDecision,
    create_cvar_position_sizer,
)

__all__ = [
    "RiskController",
    "PositionSizer",
    "KillSwitch",
    "CVaRPositionSizer",
    "CVaRCalculator",
    "DynamicDrawdownManager",
    "InactionThreshold",
    "ActionDecision",
    "create_cvar_position_sizer",
]
