"""Asset registry for managing asset-specific profiles and embeddings."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
import numpy as np
from loguru import logger


@dataclass
class AssetProfile:
    """Comprehensive profile for a tradeable asset."""

    # Identity
    symbol: str
    name: str
    asset_type: str  # token, coin, stablecoin, wrapped

    # Chain info
    native_chain: str
    contract_addresses: Dict[str, str] = field(default_factory=dict)  # chain -> address

    # Market characteristics (learned/computed)
    typical_volatility: float = 0.0
    typical_spread_bps: float = 0.0
    typical_daily_volume: float = 0.0
    liquidity_score: float = 0.5
    market_cap_tier: str = "unknown"  # mega, large, mid, small, micro

    # Correlations
    btc_correlation: float = 0.0
    eth_correlation: float = 0.0
    correlation_stability: float = 0.0

    # Behavioral patterns (learned)
    news_sensitivity: float = 0.5  # How much it reacts to news
    btc_sensitivity: float = 0.5  # How much it follows BTC
    funding_sensitivity: float = 0.5  # How much funding affects price
    whale_sensitivity: float = 0.5  # Impact of whale movements

    # Regime tendencies
    trending_tendency: float = 0.5  # Tendency to trend vs mean-revert
    volatility_clustering: float = 0.5  # Tendency for vol to cluster

    # Trading venue info
    available_exchanges: List[str] = field(default_factory=list)
    primary_exchange: str = ""
    has_perpetuals: bool = False
    has_options: bool = False

    # Metadata
    sector: str = "unknown"
    tags: Set[str] = field(default_factory=set)
    launch_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Learned embedding (updated by models)
    embedding: Optional[np.ndarray] = None

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert profile to feature dictionary."""
        return {
            "typical_volatility": self.typical_volatility,
            "typical_spread_bps": self.typical_spread_bps,
            "liquidity_score": self.liquidity_score,
            "btc_correlation": self.btc_correlation,
            "eth_correlation": self.eth_correlation,
            "correlation_stability": self.correlation_stability,
            "news_sensitivity": self.news_sensitivity,
            "btc_sensitivity": self.btc_sensitivity,
            "funding_sensitivity": self.funding_sensitivity,
            "whale_sensitivity": self.whale_sensitivity,
            "trending_tendency": self.trending_tendency,
            "volatility_clustering": self.volatility_clustering,
        }

    def to_array(self) -> np.ndarray:
        """Convert profile features to numpy array."""
        features = self.to_feature_dict()
        return np.array(list(features.values()), dtype=np.float32)


class AssetRegistry:
    """
    Registry for managing asset profiles and relationships.

    The system must build a DEDICATED INTERNAL MODEL for each traded coin.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
    ):
        self.embedding_dim = embedding_dim

        # Asset profiles
        self._profiles: Dict[str, AssetProfile] = {}

        # Cross-asset dependency graph (adjacency matrix)
        self._dependency_graph: Optional[np.ndarray] = None
        self._asset_indices: Dict[str, int] = {}

        # Dominance tracking
        self._btc_dominance: float = 0.0
        self._eth_dominance: float = 0.0
        self._altcoin_dominance: float = 0.0

        # Initialize default profiles
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize default asset profiles for major assets."""
        defaults = [
            AssetProfile(
                symbol="BTC",
                name="Bitcoin",
                asset_type="coin",
                native_chain="bitcoin",
                market_cap_tier="mega",
                sector="store_of_value",
                tags={"layer1", "pow", "store_of_value"},
                has_perpetuals=True,
                has_options=True,
                available_exchanges=["binance", "okx", "bybit"],
            ),
            AssetProfile(
                symbol="ETH",
                name="Ethereum",
                asset_type="coin",
                native_chain="ethereum",
                market_cap_tier="mega",
                sector="smart_contract_platform",
                tags={"layer1", "pos", "smart_contracts", "defi"},
                has_perpetuals=True,
                has_options=True,
                available_exchanges=["binance", "okx", "bybit"],
            ),
            AssetProfile(
                symbol="SOL",
                name="Solana",
                asset_type="coin",
                native_chain="solana",
                market_cap_tier="large",
                sector="smart_contract_platform",
                tags={"layer1", "pos", "smart_contracts", "high_throughput"},
                has_perpetuals=True,
                available_exchanges=["binance", "okx", "bybit"],
            ),
        ]

        for profile in defaults:
            self.register_asset(profile)

    def register_asset(self, profile: AssetProfile) -> None:
        """Register a new asset or update existing."""
        symbol = profile.symbol.upper()
        profile.last_updated = datetime.utcnow()

        # Initialize embedding if not present
        if profile.embedding is None:
            profile.embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            profile.embedding /= np.linalg.norm(profile.embedding)

        self._profiles[symbol] = profile

        # Update indices
        if symbol not in self._asset_indices:
            self._asset_indices[symbol] = len(self._asset_indices)

        logger.info(f"Registered asset: {symbol}")

    def get_profile(self, symbol: str) -> Optional[AssetProfile]:
        """Get asset profile by symbol."""
        return self._profiles.get(symbol.upper())

    def get_embedding(self, symbol: str) -> Optional[np.ndarray]:
        """Get learned embedding for an asset."""
        profile = self.get_profile(symbol)
        return profile.embedding if profile else None

    def update_embedding(self, symbol: str, embedding: np.ndarray) -> None:
        """Update the learned embedding for an asset."""
        profile = self.get_profile(symbol)
        if profile:
            profile.embedding = embedding
            profile.last_updated = datetime.utcnow()

    def update_characteristics(
        self,
        symbol: str,
        volatility: Optional[float] = None,
        spread_bps: Optional[float] = None,
        daily_volume: Optional[float] = None,
        btc_correlation: Optional[float] = None,
        eth_correlation: Optional[float] = None,
    ) -> None:
        """Update computed characteristics for an asset."""
        profile = self.get_profile(symbol)
        if not profile:
            logger.warning(f"Asset not found: {symbol}")
            return

        if volatility is not None:
            profile.typical_volatility = volatility
        if spread_bps is not None:
            profile.typical_spread_bps = spread_bps
        if daily_volume is not None:
            profile.typical_daily_volume = daily_volume
        if btc_correlation is not None:
            profile.btc_correlation = btc_correlation
        if eth_correlation is not None:
            profile.eth_correlation = eth_correlation

        profile.last_updated = datetime.utcnow()

    def update_sensitivities(
        self,
        symbol: str,
        news_sensitivity: Optional[float] = None,
        btc_sensitivity: Optional[float] = None,
        funding_sensitivity: Optional[float] = None,
        whale_sensitivity: Optional[float] = None,
    ) -> None:
        """Update behavioral sensitivities for an asset."""
        profile = self.get_profile(symbol)
        if not profile:
            return

        if news_sensitivity is not None:
            profile.news_sensitivity = news_sensitivity
        if btc_sensitivity is not None:
            profile.btc_sensitivity = btc_sensitivity
        if funding_sensitivity is not None:
            profile.funding_sensitivity = funding_sensitivity
        if whale_sensitivity is not None:
            profile.whale_sensitivity = whale_sensitivity

        profile.last_updated = datetime.utcnow()

    def update_dominance(
        self,
        btc_dominance: float,
        eth_dominance: float,
    ) -> None:
        """Update market dominance figures."""
        self._btc_dominance = btc_dominance
        self._eth_dominance = eth_dominance
        self._altcoin_dominance = 1.0 - btc_dominance - eth_dominance

    def get_dominance_context(self) -> Dict[str, float]:
        """Get current dominance context."""
        return {
            "btc_dominance": self._btc_dominance,
            "eth_dominance": self._eth_dominance,
            "altcoin_dominance": self._altcoin_dominance,
        }

    def build_dependency_graph(
        self,
        correlation_matrix: np.ndarray,
        threshold: float = 0.3,
    ) -> None:
        """
        Build cross-asset dependency graph from correlation matrix.

        Args:
            correlation_matrix: Asset correlation matrix
            threshold: Minimum correlation for edge
        """
        n_assets = len(self._profiles)
        self._dependency_graph = np.zeros((n_assets, n_assets), dtype=np.float32)

        # Build adjacency matrix
        for i, (sym1, profile1) in enumerate(self._profiles.items()):
            for j, (sym2, profile2) in enumerate(self._profiles.items()):
                if i != j and abs(correlation_matrix[i, j]) > threshold:
                    self._dependency_graph[i, j] = correlation_matrix[i, j]

    def get_related_assets(
        self,
        symbol: str,
        n: int = 5,
    ) -> List[tuple[str, float]]:
        """Get most related assets by embedding similarity."""
        base_profile = self.get_profile(symbol)
        if not base_profile or base_profile.embedding is None:
            return []

        similarities = []
        for other_symbol, profile in self._profiles.items():
            if other_symbol == symbol.upper() or profile.embedding is None:
                continue

            # Cosine similarity
            sim = np.dot(base_profile.embedding, profile.embedding)
            similarities.append((other_symbol, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def get_all_embeddings(self) -> np.ndarray:
        """Get all asset embeddings as a matrix."""
        symbols = sorted(self._profiles.keys())
        embeddings = []

        for symbol in symbols:
            profile = self._profiles[symbol]
            if profile.embedding is not None:
                embeddings.append(profile.embedding)
            else:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return np.stack(embeddings)

    def get_conditioning_vector(self, symbol: str) -> np.ndarray:
        """
        Get a conditioning vector for dominance-aware processing.

        Combines asset embedding with dominance context.
        """
        profile = self.get_profile(symbol)
        if not profile:
            return np.zeros(self.embedding_dim + 3, dtype=np.float32)

        embedding = profile.embedding if profile.embedding is not None else np.zeros(self.embedding_dim)

        dominance = np.array([
            self._btc_dominance,
            self._eth_dominance,
            self._altcoin_dominance,
        ], dtype=np.float32)

        return np.concatenate([embedding, dominance])

    def list_assets(self) -> List[str]:
        """List all registered assets."""
        return list(self._profiles.keys())

    def filter_by_tier(self, tier: str) -> List[str]:
        """Get assets by market cap tier."""
        return [
            symbol for symbol, profile in self._profiles.items()
            if profile.market_cap_tier == tier
        ]

    def filter_by_sector(self, sector: str) -> List[str]:
        """Get assets by sector."""
        return [
            symbol for symbol, profile in self._profiles.items()
            if profile.sector == sector
        ]

    def filter_by_tag(self, tag: str) -> List[str]:
        """Get assets by tag."""
        return [
            symbol for symbol, profile in self._profiles.items()
            if tag in profile.tags
        ]
