"""Trade data processing and feature extraction."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from cryptoai.data_universe.base import TradeData, DataSourceType


@dataclass
class TradeFeatures:
    """Extracted trade features."""

    timestamp: datetime
    asset: str
    exchange: str

    # Volume features
    total_volume: float
    buy_volume: float
    sell_volume: float
    volume_imbalance: float  # (buy - sell) / total

    # Trade count features
    trade_count: int
    buy_count: int
    sell_count: int
    count_imbalance: float

    # Price features
    vwap: float
    price_range: float
    price_volatility: float

    # Aggressor analysis
    aggressor_ratio: float  # Proportion of taker trades
    large_trade_ratio: float  # Proportion from large trades

    # Flow features
    net_flow: float  # buy_volume - sell_volume
    cumulative_delta: float

    # Intensity features
    trade_intensity: float  # Trades per second
    volume_intensity: float  # Volume per second

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.total_volume,
            self.buy_volume,
            self.sell_volume,
            self.volume_imbalance,
            self.trade_count,
            self.buy_count,
            self.sell_count,
            self.count_imbalance,
            self.vwap,
            self.price_range,
            self.price_volatility,
            self.aggressor_ratio,
            self.large_trade_ratio,
            self.net_flow,
            self.cumulative_delta,
            self.trade_intensity,
            self.volume_intensity,
        ], dtype=np.float32)


class TradeProcessor:
    """Process trade data and extract features."""

    def __init__(
        self,
        buffer_size: int = 100000,
        aggregation_window_ms: int = 1000,
        large_trade_threshold_percentile: float = 95.0,
    ):
        self.buffer_size = buffer_size
        self.aggregation_window_ms = aggregation_window_ms
        self.large_trade_threshold_percentile = large_trade_threshold_percentile

        # Trade buffers per asset-exchange
        self._trade_buffers: Dict[str, deque] = {}
        self._cumulative_delta: Dict[str, float] = {}
        self._large_trade_thresholds: Dict[str, float] = {}

    def _get_key(self, asset: str, exchange: str) -> str:
        """Get unique key for asset-exchange pair."""
        return f"{asset}:{exchange}"

    def add_trade(self, trade: TradeData) -> None:
        """Add a trade to the buffer."""
        key = self._get_key(trade.asset, trade.exchange)

        if key not in self._trade_buffers:
            self._trade_buffers[key] = deque(maxlen=self.buffer_size)
            self._cumulative_delta[key] = 0.0

        self._trade_buffers[key].append(trade)

        # Update cumulative delta
        delta = trade.quantity if trade.side == "buy" else -trade.quantity
        self._cumulative_delta[key] += delta

        # Update large trade threshold periodically
        if len(self._trade_buffers[key]) % 1000 == 0:
            self._update_large_trade_threshold(key)

    def _update_large_trade_threshold(self, key: str) -> None:
        """Update the large trade threshold based on recent data."""
        trades = list(self._trade_buffers[key])
        if len(trades) < 100:
            return

        quantities = [t.quantity for t in trades]
        self._large_trade_thresholds[key] = np.percentile(
            quantities, self.large_trade_threshold_percentile
        )

    def get_features(
        self,
        asset: str,
        exchange: str,
        window_ms: Optional[int] = None,
    ) -> Optional[TradeFeatures]:
        """
        Get aggregated trade features for a time window.

        Args:
            asset: Asset symbol
            exchange: Exchange name
            window_ms: Time window in milliseconds (default: aggregation_window_ms)

        Returns:
            Aggregated trade features
        """
        key = self._get_key(asset, exchange)

        if key not in self._trade_buffers or len(self._trade_buffers[key]) == 0:
            return None

        window_ms = window_ms or self.aggregation_window_ms
        trades = self._get_trades_in_window(key, window_ms)

        if len(trades) == 0:
            return None

        return self._extract_features(trades, asset, exchange, key)

    def _get_trades_in_window(
        self,
        key: str,
        window_ms: int,
    ) -> List[TradeData]:
        """Get trades within the specified time window."""
        all_trades = list(self._trade_buffers[key])
        if len(all_trades) == 0:
            return []

        cutoff = datetime.utcnow() - timedelta(milliseconds=window_ms)
        return [t for t in all_trades if t.timestamp >= cutoff]

    def _extract_features(
        self,
        trades: List[TradeData],
        asset: str,
        exchange: str,
        key: str,
    ) -> TradeFeatures:
        """Extract features from a list of trades."""
        # Volume metrics
        buy_trades = [t for t in trades if t.side == "buy"]
        sell_trades = [t for t in trades if t.side == "sell"]

        buy_volume = sum(t.quantity for t in buy_trades)
        sell_volume = sum(t.quantity for t in sell_trades)
        total_volume = buy_volume + sell_volume

        volume_imbalance = (
            (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
        )

        # Count metrics
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        trade_count = buy_count + sell_count

        count_imbalance = (
            (buy_count - sell_count) / trade_count if trade_count > 0 else 0.0
        )

        # Price metrics
        prices = [t.price for t in trades]
        quantities = [t.quantity for t in trades]

        vwap = (
            np.average(prices, weights=quantities) if total_volume > 0 else prices[-1]
        )
        price_range = max(prices) - min(prices) if len(prices) > 0 else 0.0
        price_volatility = np.std(prices) if len(prices) > 1 else 0.0

        # Aggressor analysis
        taker_trades = [t for t in trades if not t.is_maker]
        aggressor_ratio = len(taker_trades) / trade_count if trade_count > 0 else 0.0

        # Large trade analysis
        large_threshold = self._large_trade_thresholds.get(key, float("inf"))
        large_trades = [t for t in trades if t.quantity >= large_threshold]
        large_trade_volume = sum(t.quantity for t in large_trades)
        large_trade_ratio = (
            large_trade_volume / total_volume if total_volume > 0 else 0.0
        )

        # Flow metrics
        net_flow = buy_volume - sell_volume
        cumulative_delta = self._cumulative_delta.get(key, 0.0)

        # Intensity metrics
        if len(trades) >= 2:
            time_span = (
                trades[-1].timestamp - trades[0].timestamp
            ).total_seconds()
            time_span = max(time_span, 0.001)  # Avoid division by zero
        else:
            time_span = self.aggregation_window_ms / 1000.0

        trade_intensity = trade_count / time_span
        volume_intensity = total_volume / time_span

        return TradeFeatures(
            timestamp=trades[-1].timestamp,
            asset=asset,
            exchange=exchange,
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_imbalance=volume_imbalance,
            trade_count=trade_count,
            buy_count=buy_count,
            sell_count=sell_count,
            count_imbalance=count_imbalance,
            vwap=vwap,
            price_range=price_range,
            price_volatility=price_volatility,
            aggressor_ratio=aggressor_ratio,
            large_trade_ratio=large_trade_ratio,
            net_flow=net_flow,
            cumulative_delta=cumulative_delta,
            trade_intensity=trade_intensity,
            volume_intensity=volume_intensity,
        )

    def get_volume_profile(
        self,
        asset: str,
        exchange: str,
        n_bins: int = 50,
        window_ms: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get volume profile (volume at price levels).

        Returns:
            Dictionary with price_levels, buy_volume, sell_volume arrays
        """
        key = self._get_key(asset, exchange)

        if key not in self._trade_buffers:
            return {
                "price_levels": np.array([]),
                "buy_volume": np.array([]),
                "sell_volume": np.array([]),
            }

        window_ms = window_ms or self.aggregation_window_ms * 60  # Default 1 minute
        trades = self._get_trades_in_window(key, window_ms)

        if len(trades) == 0:
            return {
                "price_levels": np.array([]),
                "buy_volume": np.array([]),
                "sell_volume": np.array([]),
            }

        prices = np.array([t.price for t in trades])
        quantities = np.array([t.quantity for t in trades])
        sides = np.array([1 if t.side == "buy" else 0 for t in trades])

        # Create bins
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_indices = np.digitize(prices, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate volume at each level
        buy_volume = np.zeros(n_bins)
        sell_volume = np.zeros(n_bins)

        for i, (idx, qty, side) in enumerate(zip(bin_indices, quantities, sides)):
            if side == 1:
                buy_volume[idx] += qty
            else:
                sell_volume[idx] += qty

        price_levels = (bins[:-1] + bins[1:]) / 2

        return {
            "price_levels": price_levels,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
        }

    def reset_cumulative_delta(self, asset: str, exchange: str) -> None:
        """Reset cumulative delta for an asset-exchange pair."""
        key = self._get_key(asset, exchange)
        self._cumulative_delta[key] = 0.0
