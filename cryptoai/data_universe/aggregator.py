"""Data aggregator combining all data sources."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from cryptoai.data_universe.market_microstructure.features import (
    MicrostructureFeatures,
    MicrostructureState,
)
from cryptoai.data_universe.derivatives.features import (
    DerivativesFeatures,
    DerivativesState,
)
from cryptoai.data_universe.onchain.features import OnChainFeatures, OnChainState
from cryptoai.data_universe.events.processor import EventProcessor, EventFeatures
from cryptoai.data_universe.asset_registry.registry import AssetRegistry
from cryptoai.data_universe.base import (
    TradeData,
    OrderBookSnapshot,
    OnChainData,
    EventData,
)


@dataclass
class UnifiedMarketState:
    """
    Unified market state combining all data sources.

    S_t = concat(
        order_flow_embedding,
        liquidity_embedding,
        derivatives_embedding,
        onchain_embedding,
        event_embedding,
        asset_embedding,
        regime_embedding
    )
    """

    timestamp: datetime
    asset: str
    exchange: str

    # Component states
    microstructure: Optional[MicrostructureState]
    derivatives: Optional[DerivativesState]
    onchain: Optional[OnChainState]
    events: Optional[EventFeatures]

    # Asset context
    asset_embedding: np.ndarray
    dominance_context: Dict[str, float]

    # Derived features
    overall_sentiment: float  # -1 to 1
    overall_risk: float  # 0 to 1
    regime_estimate: str  # trending_up, trending_down, ranging, volatile, crisis

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        arrays = []

        # Microstructure features
        if self.microstructure:
            arrays.append(self.microstructure.to_array())
        else:
            arrays.append(np.zeros(40, dtype=np.float32))

        # Derivatives features
        if self.derivatives:
            arrays.append(self.derivatives.to_array())
        else:
            arrays.append(np.zeros(42, dtype=np.float32))

        # On-chain features
        if self.onchain:
            arrays.append(self.onchain.to_array())
        else:
            arrays.append(np.zeros(33, dtype=np.float32))

        # Event features
        if self.events:
            arrays.append(self.events.to_array())
        else:
            arrays.append(np.zeros(15, dtype=np.float32))

        # Asset embedding + dominance
        arrays.append(self.asset_embedding)
        arrays.append(np.array(list(self.dominance_context.values()), dtype=np.float32))

        # Derived features
        arrays.append(np.array([
            self.overall_sentiment,
            self.overall_risk,
            self._encode_regime(self.regime_estimate),
        ], dtype=np.float32))

        return np.concatenate(arrays)

    def _encode_regime(self, regime: str) -> float:
        """Encode regime as a float."""
        regime_map = {
            "trending_up": 1.0,
            "trending_down": -1.0,
            "ranging": 0.0,
            "volatile": 0.5,
            "crisis": -0.8,
        }
        return regime_map.get(regime, 0.0)


class DataAggregator:
    """
    Central data aggregator combining all data sources.

    This is the main interface for the trading system to access
    unified market state.
    """

    def __init__(
        self,
        exchanges: List[str] = None,
        chains: List[str] = None,
        embedding_dim: int = 64,
    ):
        self.exchanges = exchanges or ["binance", "okx", "bybit"]
        self.chains = chains or ["ethereum", "bitcoin", "arbitrum"]

        # Initialize processors
        self.microstructure = MicrostructureFeatures(
            orderbook_depth=20,
            trade_buffer_size=100000,
            aggregation_window_ms=1000,
        )
        self.derivatives = DerivativesFeatures(
            exchanges=self.exchanges,
        )
        self.onchain = OnChainFeatures(
            chains=self.chains,
        )
        self.events = EventProcessor()
        self.asset_registry = AssetRegistry(embedding_dim=embedding_dim)

        # State history for each asset
        self._state_history: Dict[str, List[UnifiedMarketState]] = {}

    # Data ingestion methods

    def process_trade(self, trade: TradeData) -> None:
        """Process incoming trade data."""
        self.microstructure.process_trade(trade)

    def process_orderbook(self, snapshot: OrderBookSnapshot) -> None:
        """Process order book snapshot."""
        self.microstructure.process_orderbook(snapshot)

    def process_derivatives_data(
        self,
        asset: str,
        exchange: str,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        liquidations: Optional[List] = None,
    ) -> None:
        """Process derivatives data."""
        from cryptoai.data_universe.derivatives.funding import FundingRateData
        from cryptoai.data_universe.derivatives.open_interest import OpenInterestData
        from cryptoai.data_universe.derivatives.liquidations import LiquidationData

        timestamp = datetime.utcnow()

        if funding_rate is not None:
            self.derivatives.process_funding(FundingRateData(
                timestamp=timestamp,
                asset=asset,
                exchange=exchange,
                funding_rate=funding_rate,
            ))

        if open_interest is not None:
            self.derivatives.process_open_interest(OpenInterestData(
                timestamp=timestamp,
                asset=asset,
                exchange=exchange,
                open_interest=open_interest,
                open_interest_usd=open_interest,  # Simplified
            ))

    def process_onchain(self, data: OnChainData) -> None:
        """Process on-chain data."""
        self.onchain.process_flow(data)

    def process_event(self, event: EventData) -> None:
        """Process event data."""
        self.events.process_event(event)

    # State retrieval

    def get_unified_state(
        self,
        asset: str,
        exchange: str = "binance",
        chain: str = "ethereum",
    ) -> UnifiedMarketState:
        """
        Get unified market state for an asset.

        This is the main method for retrieving the complete market state
        that will be fed into the neural encoders.
        """
        # Get component states
        micro_state = self.microstructure.get_state(asset, exchange)
        deriv_state = self.derivatives.get_state(asset)
        onchain_state = self.onchain.get_state(asset, chain)
        event_features = self.events.get_features(asset)

        # Get asset context
        asset_embedding = self.asset_registry.get_conditioning_vector(asset)
        dominance = self.asset_registry.get_dominance_context()

        # Calculate derived metrics
        sentiment = self._calculate_overall_sentiment(
            micro_state, deriv_state, onchain_state, event_features
        )
        risk = self._calculate_overall_risk(
            micro_state, deriv_state, onchain_state, event_features
        )
        regime = self._estimate_regime(
            micro_state, deriv_state, sentiment, risk
        )

        state = UnifiedMarketState(
            timestamp=datetime.utcnow(),
            asset=asset,
            exchange=exchange,
            microstructure=micro_state,
            derivatives=deriv_state,
            onchain=onchain_state,
            events=event_features,
            asset_embedding=asset_embedding,
            dominance_context=dominance,
            overall_sentiment=sentiment,
            overall_risk=risk,
            regime_estimate=regime,
        )

        # Store in history
        asset_key = f"{asset}:{exchange}"
        if asset_key not in self._state_history:
            self._state_history[asset_key] = []
        self._state_history[asset_key].append(state)

        # Keep history bounded
        if len(self._state_history[asset_key]) > 10000:
            self._state_history[asset_key] = self._state_history[asset_key][-5000:]

        return state

    def get_state_tensor(
        self,
        asset: str,
        exchange: str = "binance",
        sequence_length: int = 100,
    ) -> np.ndarray:
        """
        Get unified state as a time series tensor.

        Args:
            asset: Asset symbol
            exchange: Exchange name
            sequence_length: Number of historical states

        Returns:
            Tensor of shape (sequence_length, feature_dim)
        """
        asset_key = f"{asset}:{exchange}"

        if asset_key not in self._state_history:
            # Get feature dimension from a sample state
            sample_state = self.get_unified_state(asset, exchange)
            feature_dim = len(sample_state.to_array())
            return np.zeros((sequence_length, feature_dim), dtype=np.float32)

        states = self._state_history[asset_key][-sequence_length:]

        if len(states) == 0:
            return np.zeros((sequence_length, 200), dtype=np.float32)

        feature_dim = len(states[0].to_array())
        tensor = np.zeros((sequence_length, feature_dim), dtype=np.float32)

        for i, state in enumerate(states):
            idx = sequence_length - len(states) + i
            tensor[idx] = state.to_array()

        return tensor

    def _calculate_overall_sentiment(
        self,
        micro: Optional[MicrostructureState],
        deriv: Optional[DerivativesState],
        onchain: Optional[OnChainState],
        events: Optional[EventFeatures],
    ) -> float:
        """Calculate overall market sentiment from all sources."""
        signals = []
        weights = []

        if micro:
            # Order flow sentiment
            if micro.orderbook_features:
                signals.append(micro.orderbook_features.imbalance_10)
                weights.append(0.3)

        if deriv:
            # Positioning sentiment
            signals.append(deriv.positioning_score)
            weights.append(0.25)

        if onchain:
            # On-chain sentiment
            signals.append(onchain.onchain_sentiment)
            weights.append(0.25)

        if events:
            # Event sentiment
            signals.append(events.net_directional_bias)
            weights.append(0.2)

        if not signals:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        sentiment = sum(s * w for s, w in zip(signals, weights)) / total_weight

        return float(np.clip(sentiment, -1, 1))

    def _calculate_overall_risk(
        self,
        micro: Optional[MicrostructureState],
        deriv: Optional[DerivativesState],
        onchain: Optional[OnChainState],
        events: Optional[EventFeatures],
    ) -> float:
        """Calculate overall risk level from all sources."""
        risks = []

        if micro:
            # Liquidity risk (inverse of liquidity score)
            risks.append(1 - micro.liquidity_score)
            # Toxicity risk
            risks.append(micro.toxicity_score)

        if deriv:
            # Derivatives stress
            risks.append(deriv.derivatives_stress_score)
            # Leverage risk
            risks.append(deriv.leverage_score)

        if events:
            # Event risk
            risks.append(events.event_risk_score)

        if not risks:
            return 0.5

        # Use max risk as overall (conservative approach)
        return float(max(risks))

    def _estimate_regime(
        self,
        micro: Optional[MicrostructureState],
        deriv: Optional[DerivativesState],
        sentiment: float,
        risk: float,
    ) -> str:
        """Estimate current market regime."""
        # Crisis regime
        if risk > 0.8:
            return "crisis"

        # Volatile regime
        if risk > 0.6:
            return "volatile"

        # Trending regimes
        if sentiment > 0.3:
            return "trending_up"
        if sentiment < -0.3:
            return "trending_down"

        return "ranging"

    def get_multi_asset_state(
        self,
        assets: List[str],
        exchange: str = "binance",
    ) -> Dict[str, UnifiedMarketState]:
        """Get unified states for multiple assets."""
        return {
            asset: self.get_unified_state(asset, exchange)
            for asset in assets
        }

    def get_cross_asset_features(
        self,
        assets: List[str],
        exchange: str = "binance",
    ) -> np.ndarray:
        """
        Get cross-asset feature matrix.

        Returns features showing relationships between assets.
        """
        states = self.get_multi_asset_state(assets, exchange)

        n_assets = len(assets)
        features = np.zeros((n_assets, n_assets), dtype=np.float32)

        # Build correlation-like matrix from sentiment and risk
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    features[i, j] = 1.0
                else:
                    # Simple similarity based on sentiment and risk alignment
                    s1, s2 = states[asset1], states[asset2]
                    sentiment_sim = 1 - abs(s1.overall_sentiment - s2.overall_sentiment)
                    risk_sim = 1 - abs(s1.overall_risk - s2.overall_risk)
                    features[i, j] = (sentiment_sim + risk_sim) / 2

        return features
