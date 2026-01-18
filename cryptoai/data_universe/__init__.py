"""
Data Universe Module for CryptoAI.

Handles all data ingestion and processing:
- Market Microstructure (tick data, order books)
- Derivatives Intelligence (funding, OI, liquidations)
- On-Chain Intelligence (flows, whale activity)
- Event & Narrative Data (news, announcements)
- Asset Registry (coin-specific profiles)
"""

from cryptoai.data_universe.base import DataSource, DataPoint, DataStream
from cryptoai.data_universe.market_microstructure.orderbook import OrderBookProcessor
from cryptoai.data_universe.market_microstructure.trades import TradeProcessor
from cryptoai.data_universe.derivatives.funding import FundingProcessor
from cryptoai.data_universe.derivatives.open_interest import OpenInterestProcessor
from cryptoai.data_universe.onchain.flows import OnChainFlowProcessor
from cryptoai.data_universe.events.processor import EventProcessor
from cryptoai.data_universe.asset_registry.registry import AssetRegistry
from cryptoai.data_universe.aggregator import DataAggregator
from cryptoai.data_universe.multi_resolution import (
    MultiResolutionProcessor,
    MultiResolutionEncoder,
    MultiResolutionConfig,
    create_multi_resolution_processor,
)

__all__ = [
    "DataSource",
    "DataPoint",
    "DataStream",
    "OrderBookProcessor",
    "TradeProcessor",
    "FundingProcessor",
    "OpenInterestProcessor",
    "OnChainFlowProcessor",
    "EventProcessor",
    "AssetRegistry",
    "DataAggregator",
    # Multi-resolution processing
    "MultiResolutionProcessor",
    "MultiResolutionEncoder",
    "MultiResolutionConfig",
    "create_multi_resolution_processor",
]
