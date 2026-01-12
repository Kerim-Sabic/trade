"""
Neural State Encoders for CryptoAI.

All inputs are encoded via neural networks - no classical indicators.

Encoders:
- Order-flow encoder (Transformer)
- Liquidity dynamics encoder (CNN)
- Derivatives encoder (LSTM)
- On-chain temporal encoder (Transformer)
- Event & narrative encoder (Crypto-BERT)
- Asset identity encoder (Embedding)
- Volatility & regime encoder
"""

from cryptoai.encoders.order_flow import OrderFlowEncoder
from cryptoai.encoders.liquidity import LiquidityEncoder
from cryptoai.encoders.derivatives import DerivativesEncoder
from cryptoai.encoders.onchain import OnChainEncoder
from cryptoai.encoders.events import EventEncoder
from cryptoai.encoders.asset import AssetEncoder
from cryptoai.encoders.regime import RegimeEncoder
from cryptoai.encoders.unified import UnifiedStateEncoder

__all__ = [
    "OrderFlowEncoder",
    "LiquidityEncoder",
    "DerivativesEncoder",
    "OnChainEncoder",
    "EventEncoder",
    "AssetEncoder",
    "RegimeEncoder",
    "UnifiedStateEncoder",
]
