"""Funding rate processing and analysis."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class FundingRateData:
    """Funding rate data point."""

    timestamp: datetime
    asset: str
    exchange: str
    funding_rate: float
    predicted_funding_rate: Optional[float] = None
    funding_interval_hours: int = 8
    next_funding_time: Optional[datetime] = None


@dataclass
class FundingFeatures:
    """Extracted funding rate features."""

    timestamp: datetime
    asset: str

    # Current rates
    current_rate: float
    predicted_rate: float
    rate_by_exchange: Dict[str, float]

    # Aggregate metrics
    mean_rate: float
    max_rate: float
    min_rate: float
    rate_dispersion: float  # Std across exchanges

    # Trend features
    rate_change_1h: float
    rate_change_8h: float
    rate_change_24h: float

    # Regime features
    is_positive: bool
    is_extreme: bool  # > 0.1% or < -0.1%
    consecutive_positive: int
    consecutive_negative: int

    # Annualized
    annualized_rate: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.current_rate * 10000,  # Scale to bps
            self.predicted_rate * 10000,
            self.mean_rate * 10000,
            self.max_rate * 10000,
            self.min_rate * 10000,
            self.rate_dispersion * 10000,
            self.rate_change_1h * 10000,
            self.rate_change_8h * 10000,
            self.rate_change_24h * 10000,
            float(self.is_positive),
            float(self.is_extreme),
            self.consecutive_positive,
            self.consecutive_negative,
            self.annualized_rate * 100,  # Percentage
        ], dtype=np.float32)


class FundingProcessor:
    """Process funding rate data and extract features."""

    def __init__(
        self,
        exchanges: List[str] = None,
        history_hours: int = 168,  # 7 days
        extreme_threshold: float = 0.001,  # 0.1%
    ):
        self.exchanges = exchanges or ["binance", "okx", "bybit"]
        self.history_hours = history_hours
        self.extreme_threshold = extreme_threshold

        # Historical data per asset
        self._history: Dict[str, deque] = {}
        self._latest: Dict[str, Dict[str, FundingRateData]] = {}

    def _get_asset_key(self, asset: str) -> str:
        """Get key for asset."""
        return asset.upper()

    def add_funding_rate(self, data: FundingRateData) -> None:
        """Add a funding rate observation."""
        asset_key = self._get_asset_key(data.asset)

        # Initialize if needed
        if asset_key not in self._history:
            max_entries = (self.history_hours // 8) * len(self.exchanges) * 2
            self._history[asset_key] = deque(maxlen=max_entries)
            self._latest[asset_key] = {}

        self._history[asset_key].append(data)
        self._latest[asset_key][data.exchange] = data

    def get_features(self, asset: str) -> Optional[FundingFeatures]:
        """
        Get funding rate features for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Funding features or None if no data
        """
        asset_key = self._get_asset_key(asset)

        if asset_key not in self._history or len(self._history[asset_key]) == 0:
            return None

        history = list(self._history[asset_key])
        latest_by_exchange = self._latest.get(asset_key, {})

        if not latest_by_exchange:
            return None

        # Current rates by exchange
        rate_by_exchange = {
            ex: data.funding_rate for ex, data in latest_by_exchange.items()
        }

        rates = list(rate_by_exchange.values())
        current_rate = np.mean(rates)
        predicted_rates = [
            data.predicted_funding_rate
            for data in latest_by_exchange.values()
            if data.predicted_funding_rate is not None
        ]
        predicted_rate = np.mean(predicted_rates) if predicted_rates else current_rate

        # Aggregate metrics
        mean_rate = np.mean(rates)
        max_rate = np.max(rates)
        min_rate = np.min(rates)
        rate_dispersion = np.std(rates) if len(rates) > 1 else 0.0

        # Calculate rate changes
        rate_change_1h = self._calculate_rate_change(history, hours=1)
        rate_change_8h = self._calculate_rate_change(history, hours=8)
        rate_change_24h = self._calculate_rate_change(history, hours=24)

        # Regime features
        is_positive = current_rate > 0
        is_extreme = abs(current_rate) > self.extreme_threshold

        consecutive_positive, consecutive_negative = self._count_consecutive(history)

        # Annualized rate (assuming 3 funding periods per day)
        annualized_rate = current_rate * 3 * 365

        return FundingFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            current_rate=current_rate,
            predicted_rate=predicted_rate,
            rate_by_exchange=rate_by_exchange,
            mean_rate=mean_rate,
            max_rate=max_rate,
            min_rate=min_rate,
            rate_dispersion=rate_dispersion,
            rate_change_1h=rate_change_1h,
            rate_change_8h=rate_change_8h,
            rate_change_24h=rate_change_24h,
            is_positive=is_positive,
            is_extreme=is_extreme,
            consecutive_positive=consecutive_positive,
            consecutive_negative=consecutive_negative,
            annualized_rate=annualized_rate,
        )

    def _calculate_rate_change(
        self,
        history: List[FundingRateData],
        hours: int,
    ) -> float:
        """Calculate funding rate change over specified hours."""
        if len(history) < 2:
            return 0.0

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [h for h in history if h.timestamp >= cutoff]

        if len(recent) < 2:
            return 0.0

        # Get earliest and latest rates
        earliest = recent[0].funding_rate
        latest = recent[-1].funding_rate

        return latest - earliest

    def _count_consecutive(
        self,
        history: List[FundingRateData],
    ) -> Tuple[int, int]:
        """Count consecutive positive and negative funding periods."""
        if len(history) == 0:
            return 0, 0

        # Get unique funding periods (by timestamp, roughly)
        periods = []
        last_time = None
        for h in history:
            if last_time is None or (h.timestamp - last_time).total_seconds() > 3600:
                periods.append(h.funding_rate > 0)
                last_time = h.timestamp

        if len(periods) == 0:
            return 0, 0

        # Count consecutive from the end
        consecutive_positive = 0
        consecutive_negative = 0

        current_sign = periods[-1]
        count = 0

        for is_positive in reversed(periods):
            if is_positive == current_sign:
                count += 1
            else:
                break

        if current_sign:
            consecutive_positive = count
        else:
            consecutive_negative = count

        return consecutive_positive, consecutive_negative

    def get_funding_arbitrage(self, asset: str) -> Dict[str, float]:
        """
        Get funding arbitrage opportunities across exchanges.

        Returns dict with exchange pair -> rate difference.
        """
        asset_key = self._get_asset_key(asset)
        latest = self._latest.get(asset_key, {})

        if len(latest) < 2:
            return {}

        arbitrage = {}
        exchanges = list(latest.keys())

        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i + 1:]:
                diff = latest[ex1].funding_rate - latest[ex2].funding_rate
                arbitrage[f"{ex1}-{ex2}"] = diff

        return arbitrage

    def get_historical_rates(
        self,
        asset: str,
        hours: int = 24,
    ) -> List[Tuple[datetime, float]]:
        """Get historical funding rates as time series."""
        asset_key = self._get_asset_key(asset)

        if asset_key not in self._history:
            return []

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        history = list(self._history[asset_key])

        return [
            (h.timestamp, h.funding_rate)
            for h in history
            if h.timestamp >= cutoff
        ]
