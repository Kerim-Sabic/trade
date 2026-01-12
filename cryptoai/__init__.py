"""
CryptoAI - Autonomous Crypto-Native AI Trading Intelligence

A general trading intelligence that:
- Understands the specific coin it is trading
- Understands the current market regime
- Understands participant behavior
- Adapts continuously
- Survives extreme events
- Optimizes long-term risk-adjusted growth
"""

__version__ = "0.1.0"
__author__ = "CryptoAI Team"

from cryptoai.utils.logging import setup_logging
from cryptoai.utils.config import load_config

__all__ = [
    "__version__",
    "setup_logging",
    "load_config",
]
