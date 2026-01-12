"""Order book processing and feature extraction."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from cryptoai.data_universe.base import OrderBookSnapshot, DataSourceType


@dataclass
class OrderBookFeatures:
    """Extracted order book features."""

    timestamp: datetime
    asset: str
    exchange: str

    # Basic features
    mid_price: float
    spread: float
    spread_bps: float

    # Depth features
    bid_depth_5: float
    ask_depth_5: float
    bid_depth_10: float
    ask_depth_10: float
    bid_depth_20: float
    ask_depth_20: float

    # Imbalance features
    imbalance_5: float
    imbalance_10: float
    imbalance_20: float

    # Liquidity features
    bid_liquidity_usd: float
    ask_liquidity_usd: float
    total_liquidity_usd: float

    # Pressure features
    bid_pressure: float  # Weighted by proximity to mid
    ask_pressure: float

    # Elasticity (price impact per unit)
    bid_elasticity: float
    ask_elasticity: float

    # Cancellation metrics (requires historical)
    cancellation_rate: Optional[float] = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.mid_price,
            self.spread,
            self.spread_bps,
            self.bid_depth_5,
            self.ask_depth_5,
            self.bid_depth_10,
            self.ask_depth_10,
            self.bid_depth_20,
            self.ask_depth_20,
            self.imbalance_5,
            self.imbalance_10,
            self.imbalance_20,
            self.bid_liquidity_usd,
            self.ask_liquidity_usd,
            self.total_liquidity_usd,
            self.bid_pressure,
            self.ask_pressure,
            self.bid_elasticity,
            self.ask_elasticity,
            self.cancellation_rate or 0.0,
        ], dtype=np.float32)


class OrderBookProcessor:
    """Process order book data and extract features."""

    def __init__(
        self,
        depth: int = 20,
        buffer_size: int = 1000,
        track_cancellations: bool = True,
    ):
        self.depth = depth
        self.buffer_size = buffer_size
        self.track_cancellations = track_cancellations

        # Historical snapshots for feature calculation
        self._snapshot_buffer: Dict[str, deque] = {}
        self._previous_snapshots: Dict[str, OrderBookSnapshot] = {}

        # Cancellation tracking
        self._order_changes: Dict[str, deque] = {}

    def _get_key(self, asset: str, exchange: str) -> str:
        """Get unique key for asset-exchange pair."""
        return f"{asset}:{exchange}"

    def process(self, snapshot: OrderBookSnapshot) -> OrderBookFeatures:
        """
        Process order book snapshot and extract features.

        Args:
            snapshot: Order book snapshot

        Returns:
            Extracted features
        """
        key = self._get_key(snapshot.asset, snapshot.exchange)

        # Initialize buffers if needed
        if key not in self._snapshot_buffer:
            self._snapshot_buffer[key] = deque(maxlen=self.buffer_size)
            self._order_changes[key] = deque(maxlen=self.buffer_size)

        # Calculate features
        features = self._extract_features(snapshot, key)

        # Update buffers
        self._snapshot_buffer[key].append(snapshot)

        # Track cancellations if enabled
        if self.track_cancellations and key in self._previous_snapshots:
            cancellation_rate = self._calculate_cancellation_rate(
                self._previous_snapshots[key],
                snapshot,
            )
            features.cancellation_rate = cancellation_rate

        self._previous_snapshots[key] = snapshot

        return features

    def _extract_features(
        self,
        snapshot: OrderBookSnapshot,
        key: str,
    ) -> OrderBookFeatures:
        """Extract all features from snapshot."""
        bids = snapshot.bids
        asks = snapshot.asks
        mid_price = snapshot.mid_price

        # Basic features
        spread = snapshot.spread
        spread_bps = snapshot.spread_bps

        # Depth at different levels
        depth_5 = snapshot.depth_at_level(5)
        depth_10 = snapshot.depth_at_level(10)
        depth_20 = snapshot.depth_at_level(20)

        # Imbalance at different levels
        imbalance_5 = snapshot.imbalance(5)
        imbalance_10 = snapshot.imbalance(10)
        imbalance_20 = snapshot.imbalance(20)

        # Liquidity in USD
        bid_liquidity_usd = self._calculate_liquidity_usd(bids, mid_price, self.depth)
        ask_liquidity_usd = self._calculate_liquidity_usd(asks, mid_price, self.depth)
        total_liquidity_usd = bid_liquidity_usd + ask_liquidity_usd

        # Pressure (weighted by distance from mid)
        bid_pressure = self._calculate_pressure(bids, mid_price)
        ask_pressure = self._calculate_pressure(asks, mid_price)

        # Elasticity
        bid_elasticity = self._calculate_elasticity(bids, mid_price)
        ask_elasticity = self._calculate_elasticity(asks, mid_price)

        return OrderBookFeatures(
            timestamp=snapshot.timestamp,
            asset=snapshot.asset,
            exchange=snapshot.exchange,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            bid_depth_5=depth_5["bid_depth"],
            ask_depth_5=depth_5["ask_depth"],
            bid_depth_10=depth_10["bid_depth"],
            ask_depth_10=depth_10["ask_depth"],
            bid_depth_20=depth_20["bid_depth"],
            ask_depth_20=depth_20["ask_depth"],
            imbalance_5=imbalance_5,
            imbalance_10=imbalance_10,
            imbalance_20=imbalance_20,
            bid_liquidity_usd=bid_liquidity_usd,
            ask_liquidity_usd=ask_liquidity_usd,
            total_liquidity_usd=total_liquidity_usd,
            bid_pressure=bid_pressure,
            ask_pressure=ask_pressure,
            bid_elasticity=bid_elasticity,
            ask_elasticity=ask_elasticity,
        )

    def _calculate_liquidity_usd(
        self,
        orders: np.ndarray,
        mid_price: float,
        depth: int,
    ) -> float:
        """Calculate liquidity in USD within depth levels."""
        if len(orders) == 0:
            return 0.0

        orders_limited = orders[:depth]
        return float(np.sum(orders_limited[:, 0] * orders_limited[:, 1]))

    def _calculate_pressure(
        self,
        orders: np.ndarray,
        mid_price: float,
        decay: float = 0.1,
    ) -> float:
        """
        Calculate pressure weighted by distance from mid.

        Closer orders have more weight.
        """
        if len(orders) == 0 or mid_price == 0:
            return 0.0

        prices = orders[:, 0]
        quantities = orders[:, 1]

        # Distance from mid price (normalized)
        distances = np.abs(prices - mid_price) / mid_price

        # Exponential decay weights
        weights = np.exp(-decay * distances * 100)

        return float(np.sum(quantities * weights))

    def _calculate_elasticity(
        self,
        orders: np.ndarray,
        mid_price: float,
        volume_threshold: float = 100.0,
    ) -> float:
        """
        Calculate price elasticity (price impact per unit volume).

        Lower elasticity = more liquid market.
        """
        if len(orders) == 0 or mid_price == 0:
            return float("inf")

        cumulative_volume = np.cumsum(orders[:, 1])

        # Find price level at volume threshold
        idx = np.searchsorted(cumulative_volume, volume_threshold)
        if idx >= len(orders):
            idx = len(orders) - 1

        price_impact = abs(orders[idx, 0] - mid_price) / mid_price

        return price_impact / volume_threshold if volume_threshold > 0 else float("inf")

    def _calculate_cancellation_rate(
        self,
        prev_snapshot: OrderBookSnapshot,
        curr_snapshot: OrderBookSnapshot,
    ) -> float:
        """
        Calculate order cancellation rate between snapshots.

        Returns ratio of cancelled volume to total change.
        """
        # Compare bid sides
        bid_cancelled = self._compare_sides(prev_snapshot.bids, curr_snapshot.bids)
        ask_cancelled = self._compare_sides(prev_snapshot.asks, curr_snapshot.asks)

        total_cancelled = bid_cancelled + ask_cancelled
        total_volume = (
            np.sum(prev_snapshot.bids[:, 1]) if len(prev_snapshot.bids) > 0 else 0
        ) + (np.sum(prev_snapshot.asks[:, 1]) if len(prev_snapshot.asks) > 0 else 0)

        if total_volume == 0:
            return 0.0

        return total_cancelled / total_volume

    def _compare_sides(
        self,
        prev_orders: np.ndarray,
        curr_orders: np.ndarray,
    ) -> float:
        """Compare two order book sides and estimate cancellations."""
        if len(prev_orders) == 0:
            return 0.0

        # Create price-quantity dictionaries
        prev_dict = {row[0]: row[1] for row in prev_orders}
        curr_dict = {row[0]: row[1] for row in curr_orders}

        cancelled_volume = 0.0

        for price, prev_qty in prev_dict.items():
            curr_qty = curr_dict.get(price, 0.0)
            if curr_qty < prev_qty:
                # Some quantity was removed (cancelled or filled)
                cancelled_volume += prev_qty - curr_qty

        return cancelled_volume

    def get_historical_features(
        self,
        asset: str,
        exchange: str,
        n_samples: int = 100,
    ) -> List[OrderBookFeatures]:
        """Get historical features from buffer."""
        key = self._get_key(asset, exchange)
        if key not in self._snapshot_buffer:
            return []

        snapshots = list(self._snapshot_buffer[key])[-n_samples:]
        return [self._extract_features(s, key) for s in snapshots]
