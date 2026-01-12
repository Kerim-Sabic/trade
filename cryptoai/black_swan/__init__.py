"""
Black Swan Intelligence Layer for CryptoAI.

Separate AI subsystem that NEVER trades - only controls risk.

Purpose:
- Detect tail risk buildup
- Detect liquidity collapse
- Detect forced liquidation cascades
- Detect regime breaks
"""

from cryptoai.black_swan.detector import BlackSwanDetector
from cryptoai.black_swan.tail_risk import TailRiskEstimator
from cryptoai.black_swan.anomaly import AnomalyDetector
from cryptoai.black_swan.cascade import CascadeDetector

__all__ = [
    "BlackSwanDetector",
    "TailRiskEstimator",
    "AnomalyDetector",
    "CascadeDetector",
]
