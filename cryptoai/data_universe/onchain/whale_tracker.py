"""Whale wallet tracking and analysis."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import numpy as np
from loguru import logger


@dataclass
class WalletProfile:
    """Profile of a tracked whale wallet."""

    address: str
    label: Optional[str] = None  # Known entity name
    first_seen: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    total_volume_usd: float = 0.0
    transaction_count: int = 0

    # Behavioral metrics
    avg_transaction_size: float = 0.0
    activity_score: float = 0.0  # How active recently
    influence_score: float = 0.0  # Market impact correlation

    # Asset holdings (asset -> amount)
    holdings: Dict[str, float] = field(default_factory=dict)

    # Historical positions
    is_accumulating: bool = False
    is_distributing: bool = False


@dataclass
class WhaleActivity:
    """Single whale activity event."""

    timestamp: datetime
    wallet_address: str
    asset: str
    action: str  # buy, sell, transfer_in, transfer_out
    amount: float
    amount_usd: float
    price: Optional[float] = None
    destination: Optional[str] = None
    tx_hash: Optional[str] = None


@dataclass
class WhaleFeatures:
    """Aggregated whale activity features."""

    timestamp: datetime
    asset: str

    # Activity counts
    active_whale_count: int
    buy_whale_count: int
    sell_whale_count: int

    # Volume metrics
    whale_buy_volume_usd: float
    whale_sell_volume_usd: float
    whale_net_volume_usd: float

    # Sentiment
    whale_sentiment: float  # -1 to 1

    # Concentration
    top_whale_share: float  # Top 10 whales % of activity

    # Behavioral
    accumulation_score: float  # 0-1
    distribution_score: float  # 0-1

    # Smart money signal
    smart_money_signal: float  # -1 to 1

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.active_whale_count,
            self.buy_whale_count,
            self.sell_whale_count,
            np.log1p(self.whale_buy_volume_usd / 1e6),
            np.log1p(self.whale_sell_volume_usd / 1e6),
            np.sign(self.whale_net_volume_usd) * np.log1p(abs(self.whale_net_volume_usd) / 1e6),
            self.whale_sentiment,
            self.top_whale_share,
            self.accumulation_score,
            self.distribution_score,
            self.smart_money_signal,
        ], dtype=np.float32)


class WhaleTracker:
    """Track and analyze whale wallet activity."""

    def __init__(
        self,
        whale_threshold_usd: float = 1_000_000.0,
        history_hours: int = 168,  # 7 days
        smart_money_lookback_hours: int = 24,
    ):
        self.whale_threshold_usd = whale_threshold_usd
        self.history_hours = history_hours
        self.smart_money_lookback_hours = smart_money_lookback_hours

        # Tracked whale profiles
        self._profiles: Dict[str, WalletProfile] = {}

        # Activity history per asset
        self._activity_history: Dict[str, deque] = {}

        # Smart money wallets (historically profitable)
        self._smart_money_wallets: Set[str] = set()

    def add_wallet_profile(
        self,
        address: str,
        label: Optional[str] = None,
        is_smart_money: bool = False,
    ) -> None:
        """Add or update a whale wallet profile."""
        address_lower = address.lower()

        if address_lower not in self._profiles:
            self._profiles[address_lower] = WalletProfile(
                address=address_lower,
                label=label,
                first_seen=datetime.utcnow(),
            )
        else:
            if label:
                self._profiles[address_lower].label = label

        if is_smart_money:
            self._smart_money_wallets.add(address_lower)

    def add_activity(self, activity: WhaleActivity) -> None:
        """Add a whale activity event."""
        asset_key = activity.asset.upper()

        # Initialize history if needed
        if asset_key not in self._activity_history:
            max_entries = self.history_hours * 100
            self._activity_history[asset_key] = deque(maxlen=max_entries)

        self._activity_history[asset_key].append(activity)

        # Update wallet profile
        address = activity.wallet_address.lower()
        if address not in self._profiles:
            self._profiles[address] = WalletProfile(
                address=address,
                first_seen=activity.timestamp,
            )

        profile = self._profiles[address]
        profile.last_activity = activity.timestamp
        profile.total_volume_usd += activity.amount_usd
        profile.transaction_count += 1
        profile.avg_transaction_size = (
            profile.total_volume_usd / profile.transaction_count
        )

        # Update holdings
        if activity.action in ("buy", "transfer_in"):
            profile.holdings[activity.asset] = (
                profile.holdings.get(activity.asset, 0) + activity.amount
            )
        elif activity.action in ("sell", "transfer_out"):
            profile.holdings[activity.asset] = max(
                0, profile.holdings.get(activity.asset, 0) - activity.amount
            )

    def get_features(
        self,
        asset: str,
        window_hours: float = 1.0,
    ) -> Optional[WhaleFeatures]:
        """
        Get whale activity features.

        Args:
            asset: Asset symbol
            window_hours: Time window for aggregation

        Returns:
            Whale features or None if no data
        """
        asset_key = asset.upper()

        if asset_key not in self._activity_history:
            return self._empty_features(asset)

        history = list(self._activity_history[asset_key])
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [h for h in history if h.timestamp >= cutoff]

        if len(recent) == 0:
            return self._empty_features(asset)

        # Categorize activities
        buys = [h for h in recent if h.action in ("buy", "transfer_in")]
        sells = [h for h in recent if h.action in ("sell", "transfer_out")]

        # Volume metrics
        buy_volume = sum(h.amount_usd for h in buys)
        sell_volume = sum(h.amount_usd for h in sells)
        net_volume = buy_volume - sell_volume

        # Unique whale counts
        buy_wallets = set(h.wallet_address.lower() for h in buys)
        sell_wallets = set(h.wallet_address.lower() for h in sells)
        all_wallets = buy_wallets | sell_wallets

        # Whale sentiment
        total_volume = buy_volume + sell_volume
        sentiment = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

        # Top whale concentration
        wallet_volumes = {}
        for h in recent:
            addr = h.wallet_address.lower()
            wallet_volumes[addr] = wallet_volumes.get(addr, 0) + h.amount_usd

        sorted_volumes = sorted(wallet_volumes.values(), reverse=True)
        top_10_volume = sum(sorted_volumes[:10])
        top_whale_share = top_10_volume / total_volume if total_volume > 0 else 0

        # Accumulation/Distribution scores
        accumulation_score = self._calculate_accumulation_score(asset, recent)
        distribution_score = self._calculate_distribution_score(asset, recent)

        # Smart money signal
        smart_money_signal = self._calculate_smart_money_signal(asset, recent)

        return WhaleFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            active_whale_count=len(all_wallets),
            buy_whale_count=len(buy_wallets),
            sell_whale_count=len(sell_wallets),
            whale_buy_volume_usd=buy_volume,
            whale_sell_volume_usd=sell_volume,
            whale_net_volume_usd=net_volume,
            whale_sentiment=sentiment,
            top_whale_share=top_whale_share,
            accumulation_score=accumulation_score,
            distribution_score=distribution_score,
            smart_money_signal=smart_money_signal,
        )

    def _empty_features(self, asset: str) -> WhaleFeatures:
        """Return empty features when no data."""
        return WhaleFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            active_whale_count=0,
            buy_whale_count=0,
            sell_whale_count=0,
            whale_buy_volume_usd=0.0,
            whale_sell_volume_usd=0.0,
            whale_net_volume_usd=0.0,
            whale_sentiment=0.0,
            top_whale_share=0.0,
            accumulation_score=0.0,
            distribution_score=0.0,
            smart_money_signal=0.0,
        )

    def _calculate_accumulation_score(
        self,
        asset: str,
        recent_activity: List[WhaleActivity],
    ) -> float:
        """Calculate accumulation score based on whale behavior patterns."""
        if len(recent_activity) == 0:
            return 0.0

        # Look for consistent buying across multiple wallets
        buy_activities = [h for h in recent_activity if h.action in ("buy", "transfer_in")]

        if len(buy_activities) == 0:
            return 0.0

        # Count unique buyers
        unique_buyers = len(set(h.wallet_address.lower() for h in buy_activities))

        # Check for distribution of buy sizes (many small buys = accumulation)
        buy_sizes = [h.amount_usd for h in buy_activities]
        size_variance = np.std(buy_sizes) / np.mean(buy_sizes) if np.mean(buy_sizes) > 0 else 0

        # Lower variance + more unique buyers = higher accumulation score
        accumulation_score = min(1.0, unique_buyers / 10) * (1 / (1 + size_variance))

        return accumulation_score

    def _calculate_distribution_score(
        self,
        asset: str,
        recent_activity: List[WhaleActivity],
    ) -> float:
        """Calculate distribution score based on whale selling patterns."""
        if len(recent_activity) == 0:
            return 0.0

        sell_activities = [h for h in recent_activity if h.action in ("sell", "transfer_out")]

        if len(sell_activities) == 0:
            return 0.0

        # Large sells from few wallets = distribution
        unique_sellers = len(set(h.wallet_address.lower() for h in sell_activities))
        total_sell_volume = sum(h.amount_usd for h in sell_activities)

        # Check for concentrated selling
        wallet_sell_volumes = {}
        for h in sell_activities:
            addr = h.wallet_address.lower()
            wallet_sell_volumes[addr] = wallet_sell_volumes.get(addr, 0) + h.amount_usd

        # Top seller concentration
        max_seller_volume = max(wallet_sell_volumes.values())
        concentration = max_seller_volume / total_sell_volume if total_sell_volume > 0 else 0

        distribution_score = concentration * min(1.0, total_sell_volume / 10_000_000)

        return distribution_score

    def _calculate_smart_money_signal(
        self,
        asset: str,
        recent_activity: List[WhaleActivity],
    ) -> float:
        """Calculate signal from tracked smart money wallets."""
        if len(self._smart_money_wallets) == 0:
            return 0.0

        smart_money_activities = [
            h for h in recent_activity
            if h.wallet_address.lower() in self._smart_money_wallets
        ]

        if len(smart_money_activities) == 0:
            return 0.0

        buy_volume = sum(
            h.amount_usd for h in smart_money_activities
            if h.action in ("buy", "transfer_in")
        )
        sell_volume = sum(
            h.amount_usd for h in smart_money_activities
            if h.action in ("sell", "transfer_out")
        )

        total = buy_volume + sell_volume
        if total == 0:
            return 0.0

        return (buy_volume - sell_volume) / total

    def get_top_whales(
        self,
        asset: str,
        n: int = 10,
    ) -> List[WalletProfile]:
        """Get top whale profiles by holdings for an asset."""
        profiles_with_holdings = [
            p for p in self._profiles.values()
            if asset.upper() in p.holdings and p.holdings[asset.upper()] > 0
        ]

        sorted_profiles = sorted(
            profiles_with_holdings,
            key=lambda p: p.holdings.get(asset.upper(), 0),
            reverse=True,
        )

        return sorted_profiles[:n]
