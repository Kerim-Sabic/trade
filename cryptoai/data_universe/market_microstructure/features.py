"""Combined microstructure feature extraction."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from cryptoai.data_universe.market_microstructure.orderbook import (
    OrderBookProcessor,
    OrderBookFeatures,
)
from cryptoai.data_universe.market_microstructure.trades import (
    TradeProcessor,
    TradeFeatures,
)
from cryptoai.data_universe.base import OrderBookSnapshot, TradeData


@dataclass
class MicrostructureState:
    """Combined microstructure state."""

    timestamp: datetime
    asset: str
    exchange: str

    orderbook_features: Optional[OrderBookFeatures]
    trade_features: Optional[TradeFeatures]

    # Combined features
    liquidity_score: float  # 0-1, higher = more liquid
    urgency_score: float  # 0-1, higher = more urgent activity
    toxicity_score: float  # 0-1, higher = more toxic flow

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        ob_array = (
            self.orderbook_features.to_array()
            if self.orderbook_features
            else np.zeros(20, dtype=np.float32)
        )
        trade_array = (
            self.trade_features.to_array()
            if self.trade_features
            else np.zeros(17, dtype=np.float32)
        )
        combined = np.array(
            [self.liquidity_score, self.urgency_score, self.toxicity_score],
            dtype=np.float32,
        )
        return np.concatenate([ob_array, trade_array, combined])


class MicrostructureFeatures:
    """
    Unified microstructure feature extraction.

    Combines order book and trade data into a comprehensive market state.
    """

    def __init__(
        self,
        orderbook_depth: int = 20,
        trade_buffer_size: int = 100000,
        aggregation_window_ms: int = 1000,
    ):
        self.orderbook_processor = OrderBookProcessor(
            depth=orderbook_depth,
            buffer_size=1000,
            track_cancellations=True,
        )
        self.trade_processor = TradeProcessor(
            buffer_size=trade_buffer_size,
            aggregation_window_ms=aggregation_window_ms,
        )

        # Historical states for time series
        self._state_history: Dict[str, List[MicrostructureState]] = {}

    def _get_key(self, asset: str, exchange: str) -> str:
        """Get unique key for asset-exchange pair."""
        return f"{asset}:{exchange}"

    def process_orderbook(self, snapshot: OrderBookSnapshot) -> OrderBookFeatures:
        """Process an order book snapshot."""
        return self.orderbook_processor.process(snapshot)

    def process_trade(self, trade: TradeData) -> None:
        """Process a trade."""
        self.trade_processor.add_trade(trade)

    def get_state(
        self,
        asset: str,
        exchange: str,
        orderbook_snapshot: Optional[OrderBookSnapshot] = None,
    ) -> MicrostructureState:
        """
        Get current microstructure state.

        Args:
            asset: Asset symbol
            exchange: Exchange name
            orderbook_snapshot: Optional current orderbook snapshot

        Returns:
            Current microstructure state
        """
        key = self._get_key(asset, exchange)

        # Get features
        ob_features = None
        if orderbook_snapshot:
            ob_features = self.process_orderbook(orderbook_snapshot)

        trade_features = self.trade_processor.get_features(asset, exchange)

        # Calculate combined scores
        liquidity_score = self._calculate_liquidity_score(ob_features, trade_features)
        urgency_score = self._calculate_urgency_score(ob_features, trade_features)
        toxicity_score = self._calculate_toxicity_score(ob_features, trade_features)

        state = MicrostructureState(
            timestamp=datetime.utcnow(),
            asset=asset,
            exchange=exchange,
            orderbook_features=ob_features,
            trade_features=trade_features,
            liquidity_score=liquidity_score,
            urgency_score=urgency_score,
            toxicity_score=toxicity_score,
        )

        # Store in history
        if key not in self._state_history:
            self._state_history[key] = []
        self._state_history[key].append(state)

        # Keep history bounded
        if len(self._state_history[key]) > 10000:
            self._state_history[key] = self._state_history[key][-5000:]

        return state

    def _calculate_liquidity_score(
        self,
        ob_features: Optional[OrderBookFeatures],
        trade_features: Optional[TradeFeatures],
    ) -> float:
        """
        Calculate overall liquidity score (0-1).

        Higher = more liquid market.
        """
        scores = []

        if ob_features:
            # Spread component (lower spread = higher score)
            spread_score = 1.0 / (1.0 + ob_features.spread_bps / 10.0)
            scores.append(spread_score)

            # Depth component
            total_depth = ob_features.bid_depth_10 + ob_features.ask_depth_10
            depth_score = min(1.0, total_depth / 1000.0)  # Normalize
            scores.append(depth_score)

            # Elasticity component (lower = better)
            avg_elasticity = (ob_features.bid_elasticity + ob_features.ask_elasticity) / 2
            elasticity_score = 1.0 / (1.0 + avg_elasticity * 10000)
            scores.append(elasticity_score)

        if trade_features:
            # Volume intensity component
            volume_score = min(1.0, trade_features.volume_intensity / 100.0)
            scores.append(volume_score)

        return np.mean(scores) if scores else 0.5

    def _calculate_urgency_score(
        self,
        ob_features: Optional[OrderBookFeatures],
        trade_features: Optional[TradeFeatures],
    ) -> float:
        """
        Calculate urgency score (0-1).

        Higher = more urgent/aggressive trading activity.
        """
        scores = []

        if ob_features:
            # Imbalance magnitude (absolute)
            imbalance_score = abs(ob_features.imbalance_10)
            scores.append(imbalance_score)

            # Cancellation rate (higher = more urgency)
            if ob_features.cancellation_rate is not None:
                scores.append(min(1.0, ob_features.cancellation_rate * 5))

        if trade_features:
            # Trade intensity relative to baseline
            intensity_score = min(1.0, trade_features.trade_intensity / 50.0)
            scores.append(intensity_score)

            # Aggressor ratio (higher taker activity = more urgency)
            scores.append(trade_features.aggressor_ratio)

            # Large trade activity
            scores.append(trade_features.large_trade_ratio)

        return np.mean(scores) if scores else 0.5

    def _calculate_toxicity_score(
        self,
        ob_features: Optional[OrderBookFeatures],
        trade_features: Optional[TradeFeatures],
    ) -> float:
        """
        Calculate flow toxicity score (0-1).

        Higher = more toxic (informed) flow.
        Based on VPIN-like concepts.
        """
        scores = []

        if trade_features:
            # Volume imbalance magnitude
            imbalance_score = abs(trade_features.volume_imbalance)
            scores.append(imbalance_score)

            # Large trade concentration
            scores.append(trade_features.large_trade_ratio)

            # Aggressive vs passive ratio
            aggressor_toxicity = trade_features.aggressor_ratio
            scores.append(aggressor_toxicity)

        if ob_features:
            # Order book imbalance at deep levels
            deep_imbalance = abs(ob_features.imbalance_20)
            scores.append(deep_imbalance)

        return np.mean(scores) if scores else 0.5

    def get_state_tensor(
        self,
        asset: str,
        exchange: str,
        sequence_length: int = 100,
    ) -> np.ndarray:
        """
        Get state as a tensor for neural network input.

        Args:
            asset: Asset symbol
            exchange: Exchange name
            sequence_length: Number of historical states

        Returns:
            Tensor of shape (sequence_length, feature_dim)
        """
        key = self._get_key(asset, exchange)

        if key not in self._state_history:
            feature_dim = 20 + 17 + 3  # ob + trade + combined
            return np.zeros((sequence_length, feature_dim), dtype=np.float32)

        states = self._state_history[key][-sequence_length:]

        # Pad if needed
        feature_dim = 20 + 17 + 3
        tensor = np.zeros((sequence_length, feature_dim), dtype=np.float32)

        for i, state in enumerate(states):
            idx = sequence_length - len(states) + i
            tensor[idx] = state.to_array()

        return tensor
