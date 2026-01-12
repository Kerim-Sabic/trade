"""Open interest processing and analysis."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from loguru import logger


@dataclass
class OpenInterestData:
    """Open interest data point."""

    timestamp: datetime
    asset: str
    exchange: str
    open_interest: float  # In contracts or coins
    open_interest_usd: float
    long_short_ratio: Optional[float] = None
    top_trader_long_ratio: Optional[float] = None
    top_trader_short_ratio: Optional[float] = None


@dataclass
class OpenInterestFeatures:
    """Extracted open interest features."""

    timestamp: datetime
    asset: str

    # Current OI
    total_oi_usd: float
    oi_by_exchange: Dict[str, float]

    # Change metrics
    oi_change_1h_pct: float
    oi_change_4h_pct: float
    oi_change_24h_pct: float

    # Positioning
    avg_long_short_ratio: float
    long_short_ratio_by_exchange: Dict[str, float]

    # Concentration
    oi_concentration: float  # Herfindahl index across exchanges

    # Top trader positioning
    top_trader_long_ratio: Optional[float]
    top_trader_short_ratio: Optional[float]

    # Derived signals
    is_oi_building: bool
    is_oi_unwinding: bool
    positioning_imbalance: float  # -1 to 1, positive = more longs

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            np.log1p(self.total_oi_usd / 1e6),  # Log scale, in millions
            self.oi_change_1h_pct,
            self.oi_change_4h_pct,
            self.oi_change_24h_pct,
            self.avg_long_short_ratio,
            self.oi_concentration,
            self.top_trader_long_ratio or 0.5,
            self.top_trader_short_ratio or 0.5,
            float(self.is_oi_building),
            float(self.is_oi_unwinding),
            self.positioning_imbalance,
        ], dtype=np.float32)


class OpenInterestProcessor:
    """Process open interest data and extract features."""

    def __init__(
        self,
        exchanges: List[str] = None,
        history_hours: int = 168,  # 7 days
        building_threshold: float = 0.05,  # 5% increase
        unwinding_threshold: float = -0.05,  # 5% decrease
    ):
        self.exchanges = exchanges or ["binance", "okx", "bybit"]
        self.history_hours = history_hours
        self.building_threshold = building_threshold
        self.unwinding_threshold = unwinding_threshold

        # Historical data per asset
        self._history: Dict[str, deque] = {}
        self._latest: Dict[str, Dict[str, OpenInterestData]] = {}

    def _get_asset_key(self, asset: str) -> str:
        """Get key for asset."""
        return asset.upper()

    def add_open_interest(self, data: OpenInterestData) -> None:
        """Add an open interest observation."""
        asset_key = self._get_asset_key(data.asset)

        # Initialize if needed
        if asset_key not in self._history:
            # Assuming updates every minute, 7 days of history
            max_entries = self.history_hours * 60
            self._history[asset_key] = deque(maxlen=max_entries)
            self._latest[asset_key] = {}

        self._history[asset_key].append(data)
        self._latest[asset_key][data.exchange] = data

    def get_features(self, asset: str) -> Optional[OpenInterestFeatures]:
        """
        Get open interest features for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Open interest features or None if no data
        """
        asset_key = self._get_asset_key(asset)

        if asset_key not in self._history or len(self._history[asset_key]) == 0:
            return None

        history = list(self._history[asset_key])
        latest_by_exchange = self._latest.get(asset_key, {})

        if not latest_by_exchange:
            return None

        # Current OI by exchange
        oi_by_exchange = {
            ex: data.open_interest_usd for ex, data in latest_by_exchange.items()
        }
        total_oi_usd = sum(oi_by_exchange.values())

        # Change metrics
        oi_change_1h = self._calculate_oi_change(history, hours=1)
        oi_change_4h = self._calculate_oi_change(history, hours=4)
        oi_change_24h = self._calculate_oi_change(history, hours=24)

        # Long/short ratios
        long_short_ratios = {
            ex: data.long_short_ratio
            for ex, data in latest_by_exchange.items()
            if data.long_short_ratio is not None
        }
        avg_ls_ratio = np.mean(list(long_short_ratios.values())) if long_short_ratios else 1.0

        # OI concentration (Herfindahl index)
        oi_concentration = self._calculate_concentration(list(oi_by_exchange.values()))

        # Top trader positioning
        top_long_ratios = [
            data.top_trader_long_ratio
            for data in latest_by_exchange.values()
            if data.top_trader_long_ratio is not None
        ]
        top_short_ratios = [
            data.top_trader_short_ratio
            for data in latest_by_exchange.values()
            if data.top_trader_short_ratio is not None
        ]

        top_trader_long = np.mean(top_long_ratios) if top_long_ratios else None
        top_trader_short = np.mean(top_short_ratios) if top_short_ratios else None

        # Derived signals
        is_building = oi_change_4h > self.building_threshold
        is_unwinding = oi_change_4h < self.unwinding_threshold

        # Positioning imbalance from long/short ratio
        # L/S ratio of 1.5 means 60% long, 40% short -> imbalance = 0.2
        positioning_imbalance = (avg_ls_ratio - 1) / (avg_ls_ratio + 1) if avg_ls_ratio > 0 else 0

        return OpenInterestFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            total_oi_usd=total_oi_usd,
            oi_by_exchange=oi_by_exchange,
            oi_change_1h_pct=oi_change_1h,
            oi_change_4h_pct=oi_change_4h,
            oi_change_24h_pct=oi_change_24h,
            avg_long_short_ratio=avg_ls_ratio,
            long_short_ratio_by_exchange=long_short_ratios,
            oi_concentration=oi_concentration,
            top_trader_long_ratio=top_trader_long,
            top_trader_short_ratio=top_trader_short,
            is_oi_building=is_building,
            is_oi_unwinding=is_unwinding,
            positioning_imbalance=positioning_imbalance,
        )

    def _calculate_oi_change(
        self,
        history: List[OpenInterestData],
        hours: int,
    ) -> float:
        """Calculate OI change percentage over specified hours."""
        if len(history) < 2:
            return 0.0

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Get earliest OI after cutoff
        earlier_oi = None
        for h in history:
            if h.timestamp >= cutoff:
                earlier_oi = h.open_interest_usd
                break

        if earlier_oi is None or earlier_oi == 0:
            return 0.0

        # Get latest OI
        latest_oi = history[-1].open_interest_usd

        return (latest_oi - earlier_oi) / earlier_oi

    def _calculate_concentration(self, values: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        if len(values) == 0:
            return 0.0

        total = sum(values)
        if total == 0:
            return 0.0

        shares = [v / total for v in values]
        hhi = sum(s ** 2 for s in shares)

        return hhi

    def get_oi_price_divergence(
        self,
        asset: str,
        price_changes: Dict[str, float],  # {timeframe: pct_change}
    ) -> Dict[str, float]:
        """
        Detect OI-price divergences.

        Returns divergence scores for each timeframe.
        Positive = OI up, price down (bearish divergence)
        Negative = OI down, price up (bullish divergence)
        """
        asset_key = self._get_asset_key(asset)
        history = list(self._history.get(asset_key, []))

        divergences = {}

        for timeframe, price_change in price_changes.items():
            if timeframe == "1h":
                oi_change = self._calculate_oi_change(history, hours=1)
            elif timeframe == "4h":
                oi_change = self._calculate_oi_change(history, hours=4)
            elif timeframe == "24h":
                oi_change = self._calculate_oi_change(history, hours=24)
            else:
                continue

            # Divergence: OI and price moving in opposite directions
            if oi_change * price_change < 0:  # Opposite signs
                divergences[timeframe] = oi_change - price_change
            else:
                divergences[timeframe] = 0.0

        return divergences
