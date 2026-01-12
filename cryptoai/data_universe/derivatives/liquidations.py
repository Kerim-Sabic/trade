"""Liquidation data processing and analysis."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class LiquidationData:
    """Single liquidation event."""

    timestamp: datetime
    asset: str
    exchange: str
    side: str  # long, short
    quantity: float
    price: float
    value_usd: float
    order_id: Optional[str] = None


@dataclass
class LiquidationFeatures:
    """Extracted liquidation features."""

    timestamp: datetime
    asset: str

    # Volume metrics
    total_liquidations_usd: float
    long_liquidations_usd: float
    short_liquidations_usd: float
    liquidation_imbalance: float  # (long - short) / total

    # Count metrics
    liquidation_count: int
    long_count: int
    short_count: int

    # Size metrics
    avg_liquidation_size: float
    max_liquidation_size: float
    large_liquidation_count: int  # > $100k

    # Rate metrics
    liquidations_per_hour: float

    # Trend
    liquidation_trend_1h: float  # Change vs previous hour
    liquidation_trend_4h: float

    # Cascade detection
    cascade_score: float  # 0-1, higher = cascade likely

    # By exchange
    liquidations_by_exchange: Dict[str, float]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            np.log1p(self.total_liquidations_usd / 1e6),  # Log scale
            np.log1p(self.long_liquidations_usd / 1e6),
            np.log1p(self.short_liquidations_usd / 1e6),
            self.liquidation_imbalance,
            self.liquidation_count,
            self.long_count,
            self.short_count,
            np.log1p(self.avg_liquidation_size / 1e4),
            np.log1p(self.max_liquidation_size / 1e5),
            self.large_liquidation_count,
            self.liquidations_per_hour,
            self.liquidation_trend_1h,
            self.liquidation_trend_4h,
            self.cascade_score,
        ], dtype=np.float32)


class LiquidationProcessor:
    """Process liquidation data and extract features."""

    def __init__(
        self,
        history_hours: int = 72,
        large_threshold_usd: float = 100000.0,
        cascade_window_minutes: int = 5,
        cascade_threshold: int = 10,  # Liquidations in window
    ):
        self.history_hours = history_hours
        self.large_threshold_usd = large_threshold_usd
        self.cascade_window_minutes = cascade_window_minutes
        self.cascade_threshold = cascade_threshold

        # Historical data per asset
        self._history: Dict[str, deque] = {}

    def _get_asset_key(self, asset: str) -> str:
        """Get key for asset."""
        return asset.upper()

    def add_liquidation(self, data: LiquidationData) -> None:
        """Add a liquidation event."""
        asset_key = self._get_asset_key(data.asset)

        if asset_key not in self._history:
            # Store individual liquidations
            max_entries = self.history_hours * 1000  # Assume max 1000/hour
            self._history[asset_key] = deque(maxlen=max_entries)

        self._history[asset_key].append(data)

    def get_features(
        self,
        asset: str,
        window_hours: float = 1.0,
    ) -> Optional[LiquidationFeatures]:
        """
        Get liquidation features for an asset.

        Args:
            asset: Asset symbol
            window_hours: Time window for aggregation

        Returns:
            Liquidation features or None if no data
        """
        asset_key = self._get_asset_key(asset)

        if asset_key not in self._history:
            return None

        history = list(self._history[asset_key])
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [h for h in history if h.timestamp >= cutoff]

        if len(recent) == 0:
            return self._empty_features(asset)

        # Volume metrics
        long_liqs = [h for h in recent if h.side == "long"]
        short_liqs = [h for h in recent if h.side == "short"]

        long_usd = sum(h.value_usd for h in long_liqs)
        short_usd = sum(h.value_usd for h in short_liqs)
        total_usd = long_usd + short_usd

        imbalance = (long_usd - short_usd) / total_usd if total_usd > 0 else 0.0

        # Count metrics
        long_count = len(long_liqs)
        short_count = len(short_liqs)
        total_count = long_count + short_count

        # Size metrics
        sizes = [h.value_usd for h in recent]
        avg_size = np.mean(sizes) if sizes else 0.0
        max_size = np.max(sizes) if sizes else 0.0
        large_count = sum(1 for s in sizes if s > self.large_threshold_usd)

        # Rate metrics
        liqs_per_hour = total_count / window_hours

        # Trend metrics
        trend_1h = self._calculate_trend(history, compare_hours=1, window_hours=1)
        trend_4h = self._calculate_trend(history, compare_hours=4, window_hours=4)

        # Cascade detection
        cascade_score = self._detect_cascade(recent)

        # By exchange
        by_exchange: Dict[str, float] = {}
        for h in recent:
            by_exchange[h.exchange] = by_exchange.get(h.exchange, 0) + h.value_usd

        return LiquidationFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            total_liquidations_usd=total_usd,
            long_liquidations_usd=long_usd,
            short_liquidations_usd=short_usd,
            liquidation_imbalance=imbalance,
            liquidation_count=total_count,
            long_count=long_count,
            short_count=short_count,
            avg_liquidation_size=avg_size,
            max_liquidation_size=max_size,
            large_liquidation_count=large_count,
            liquidations_per_hour=liqs_per_hour,
            liquidation_trend_1h=trend_1h,
            liquidation_trend_4h=trend_4h,
            cascade_score=cascade_score,
            liquidations_by_exchange=by_exchange,
        )

    def _empty_features(self, asset: str) -> LiquidationFeatures:
        """Return empty features when no data."""
        return LiquidationFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            total_liquidations_usd=0.0,
            long_liquidations_usd=0.0,
            short_liquidations_usd=0.0,
            liquidation_imbalance=0.0,
            liquidation_count=0,
            long_count=0,
            short_count=0,
            avg_liquidation_size=0.0,
            max_liquidation_size=0.0,
            large_liquidation_count=0,
            liquidations_per_hour=0.0,
            liquidation_trend_1h=0.0,
            liquidation_trend_4h=0.0,
            cascade_score=0.0,
            liquidations_by_exchange={},
        )

    def _calculate_trend(
        self,
        history: List[LiquidationData],
        compare_hours: float,
        window_hours: float,
    ) -> float:
        """Calculate trend in liquidations vs previous period."""
        now = datetime.utcnow()

        # Current window
        current_start = now - timedelta(hours=window_hours)
        current = [h for h in history if h.timestamp >= current_start]
        current_vol = sum(h.value_usd for h in current)

        # Previous window
        prev_start = current_start - timedelta(hours=compare_hours)
        prev_end = current_start
        previous = [h for h in history if prev_start <= h.timestamp < prev_end]
        prev_vol = sum(h.value_usd for h in previous)

        if prev_vol == 0:
            return 0.0 if current_vol == 0 else 1.0

        return (current_vol - prev_vol) / prev_vol

    def _detect_cascade(self, liquidations: List[LiquidationData]) -> float:
        """
        Detect liquidation cascade.

        Returns score 0-1, higher = more likely cascade.
        """
        if len(liquidations) < 3:
            return 0.0

        # Sort by timestamp
        sorted_liqs = sorted(liquidations, key=lambda x: x.timestamp)

        # Find maximum concentration in any cascade window
        max_concentration = 0
        window_delta = timedelta(minutes=self.cascade_window_minutes)

        for i, liq in enumerate(sorted_liqs):
            window_end = liq.timestamp + window_delta
            window_liqs = [
                l for l in sorted_liqs[i:]
                if l.timestamp <= window_end
            ]
            max_concentration = max(max_concentration, len(window_liqs))

        # Normalize to 0-1
        cascade_score = min(1.0, max_concentration / self.cascade_threshold)

        return cascade_score

    def get_liquidation_heatmap(
        self,
        asset: str,
        hours: int = 24,
        price_bins: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Get liquidation heatmap (liquidations by price level).

        Returns:
            Dict with price_levels, long_volume, short_volume arrays
        """
        asset_key = self._get_asset_key(asset)

        if asset_key not in self._history:
            return {
                "price_levels": np.array([]),
                "long_volume": np.array([]),
                "short_volume": np.array([]),
            }

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        history = [h for h in self._history[asset_key] if h.timestamp >= cutoff]

        if len(history) == 0:
            return {
                "price_levels": np.array([]),
                "long_volume": np.array([]),
                "short_volume": np.array([]),
            }

        prices = np.array([h.price for h in history])
        values = np.array([h.value_usd for h in history])
        sides = np.array([1 if h.side == "long" else 0 for h in history])

        # Create bins
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, price_bins + 1)
        bin_indices = np.digitize(prices, bins) - 1
        bin_indices = np.clip(bin_indices, 0, price_bins - 1)

        # Aggregate
        long_volume = np.zeros(price_bins)
        short_volume = np.zeros(price_bins)

        for idx, val, side in zip(bin_indices, values, sides):
            if side == 1:
                long_volume[idx] += val
            else:
                short_volume[idx] += val

        price_levels = (bins[:-1] + bins[1:]) / 2

        return {
            "price_levels": price_levels,
            "long_volume": long_volume,
            "short_volume": short_volume,
        }
