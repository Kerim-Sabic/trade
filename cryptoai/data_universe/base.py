"""Base classes for data universe."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, Generic, List, Optional, TypeVar
import numpy as np
from pydantic import BaseModel, Field


class DataSourceType(str, Enum):
    """Types of data sources."""

    MARKET_MICROSTRUCTURE = "market_microstructure"
    DERIVATIVES = "derivatives"
    ONCHAIN = "onchain"
    EVENTS = "events"
    ASSET_INFO = "asset_info"


class ReliabilityScore(float, Enum):
    """Data source reliability scores."""

    EXCHANGE_DIRECT = 1.0
    AGGREGATOR_VERIFIED = 0.9
    ONCHAIN_VERIFIED = 0.95
    NEWS_VERIFIED = 0.8
    SOCIAL_VERIFIED = 0.6
    UNVERIFIED = 0.3


@dataclass
class DataPoint:
    """Base class for all data points with metadata."""

    timestamp: datetime
    source: str
    source_type: DataSourceType
    reliability_score: float
    asset: Optional[str] = None
    exchange: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "source_type": self.source_type.value,
            "reliability_score": self.reliability_score,
            "asset": self.asset,
            "exchange": self.exchange,
        }


@dataclass
class TradeData(DataPoint):
    """Tick-level trade data."""

    price: float = 0.0
    quantity: float = 0.0
    side: str = "buy"  # buy, sell
    trade_id: Optional[str] = None
    is_maker: bool = False

    def __post_init__(self):
        self.source_type = DataSourceType.MARKET_MICROSTRUCTURE


@dataclass
class OrderBookSnapshot(DataPoint):
    """Order book L2 snapshot."""

    bids: np.ndarray = field(default_factory=lambda: np.array([]))  # [[price, qty], ...]
    asks: np.ndarray = field(default_factory=lambda: np.array([]))
    sequence_id: Optional[int] = None

    def __post_init__(self):
        self.source_type = DataSourceType.MARKET_MICROSTRUCTURE

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if len(self.bids) == 0 or len(self.asks) == 0:
            return 0.0
        return (self.bids[0, 0] + self.asks[0, 0]) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if len(self.bids) == 0 or len(self.asks) == 0:
            return 0.0
        return self.asks[0, 0] - self.bids[0, 0]

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000

    def depth_at_level(self, level: int = 5) -> Dict[str, float]:
        """Calculate depth at specified level."""
        bid_depth = np.sum(self.bids[:level, 1]) if len(self.bids) >= level else np.sum(self.bids[:, 1])
        ask_depth = np.sum(self.asks[:level, 1]) if len(self.asks) >= level else np.sum(self.asks[:, 1])
        return {"bid_depth": float(bid_depth), "ask_depth": float(ask_depth)}

    def imbalance(self, level: int = 5) -> float:
        """Calculate order book imbalance."""
        depths = self.depth_at_level(level)
        total = depths["bid_depth"] + depths["ask_depth"]
        if total == 0:
            return 0.0
        return (depths["bid_depth"] - depths["ask_depth"]) / total


@dataclass
class DerivativesData(DataPoint):
    """Derivatives market data."""

    funding_rate: Optional[float] = None
    predicted_funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    open_interest_usd: Optional[float] = None
    long_short_ratio: Optional[float] = None
    liquidations_long: Optional[float] = None
    liquidations_short: Optional[float] = None
    basis: Optional[float] = None

    def __post_init__(self):
        self.source_type = DataSourceType.DERIVATIVES


@dataclass
class OnChainData(DataPoint):
    """On-chain flow data."""

    flow_type: str = ""  # inflow, outflow, transfer
    amount: float = 0.0
    amount_usd: float = 0.0
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    from_label: Optional[str] = None  # exchange, whale, contract
    to_label: Optional[str] = None
    tx_hash: Optional[str] = None
    chain: str = "ethereum"

    def __post_init__(self):
        self.source_type = DataSourceType.ONCHAIN


@dataclass
class EventData(DataPoint):
    """Event and narrative data."""

    event_type: str = ""  # announcement, upgrade, exploit, regulatory, etc.
    title: str = ""
    content: str = ""
    url: Optional[str] = None
    affected_assets: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1
    impact_magnitude: Optional[float] = None  # 0 to 1
    credibility_score: float = 0.5
    is_verified: bool = False

    def __post_init__(self):
        self.source_type = DataSourceType.EVENTS


T = TypeVar("T", bound=DataPoint)


class DataSource(ABC, Generic[T]):
    """Abstract base class for data sources."""

    def __init__(
        self,
        name: str,
        source_type: DataSourceType,
        reliability_score: float = 1.0,
    ):
        self.name = name
        self.source_type = source_type
        self.reliability_score = reliability_score
        self._is_connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[T]:
        """Stream data points from the source."""
        pass

    @abstractmethod
    async def fetch_historical(
        self,
        start: datetime,
        end: datetime,
        **kwargs,
    ) -> List[T]:
        """Fetch historical data."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._is_connected


class DataStream:
    """Manages multiple data sources and provides unified stream."""

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self._is_running = False

    def add_source(self, source: DataSource) -> None:
        """Add a data source."""
        self.sources[source.name] = source

    def remove_source(self, name: str) -> None:
        """Remove a data source."""
        if name in self.sources:
            del self.sources[name]

    async def connect_all(self) -> None:
        """Connect all data sources."""
        for source in self.sources.values():
            await source.connect()

    async def disconnect_all(self) -> None:
        """Disconnect all data sources."""
        for source in self.sources.values():
            await source.disconnect()

    async def stream_all(self) -> AsyncIterator[DataPoint]:
        """Stream from all sources."""
        import asyncio

        queues = []
        for source in self.sources.values():
            queue = asyncio.Queue()
            queues.append(queue)

            async def producer(s, q):
                async for data_point in s.stream():
                    await q.put(data_point)

            asyncio.create_task(producer(source, queue))

        self._is_running = True
        while self._is_running:
            for queue in queues:
                try:
                    data_point = queue.get_nowait()
                    yield data_point
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(0.001)

    def stop(self) -> None:
        """Stop streaming."""
        self._is_running = False
