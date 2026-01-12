"""Combined on-chain feature extraction."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from cryptoai.data_universe.onchain.flows import OnChainFlowProcessor, FlowFeatures
from cryptoai.data_universe.onchain.whale_tracker import WhaleTracker, WhaleFeatures, WhaleActivity
from cryptoai.data_universe.base import OnChainData


@dataclass
class StablecoinFeatures:
    """Stablecoin flow features."""

    timestamp: datetime

    # Supply changes
    total_supply_change_24h: float  # Across major stables
    usdt_supply_change: float
    usdc_supply_change: float

    # Exchange flows
    stablecoin_exchange_inflow: float
    stablecoin_exchange_outflow: float

    # Buying power indicator
    dry_powder_score: float  # 0-1, higher = more stables ready to buy

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.total_supply_change_24h / 1e9,  # In billions
            self.usdt_supply_change / 1e9,
            self.usdc_supply_change / 1e9,
            np.log1p(self.stablecoin_exchange_inflow / 1e6),
            np.log1p(self.stablecoin_exchange_outflow / 1e6),
            self.dry_powder_score,
        ], dtype=np.float32)


@dataclass
class OnChainState:
    """Combined on-chain state."""

    timestamp: datetime
    asset: str

    flow_features: Optional[FlowFeatures]
    whale_features: Optional[WhaleFeatures]
    stablecoin_features: Optional[StablecoinFeatures]

    # Combined metrics
    onchain_sentiment: float  # -1 to 1
    accumulation_phase: bool
    distribution_phase: bool
    network_activity_score: float  # 0-1

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        flow_arr = self.flow_features.to_array() if self.flow_features else np.zeros(12, dtype=np.float32)
        whale_arr = self.whale_features.to_array() if self.whale_features else np.zeros(11, dtype=np.float32)
        stable_arr = self.stablecoin_features.to_array() if self.stablecoin_features else np.zeros(6, dtype=np.float32)
        combined = np.array([
            self.onchain_sentiment,
            float(self.accumulation_phase),
            float(self.distribution_phase),
            self.network_activity_score,
        ], dtype=np.float32)

        return np.concatenate([flow_arr, whale_arr, stable_arr, combined])


class OnChainFeatures:
    """
    Unified on-chain feature extraction.

    Combines exchange flows, whale activity, and stablecoin metrics.
    """

    def __init__(
        self,
        chains: List[str] = None,
        whale_threshold_usd: float = 1_000_000.0,
        history_hours: int = 168,
    ):
        self.flow_processor = OnChainFlowProcessor(
            chains=chains,
            history_hours=history_hours,
            whale_threshold_usd=whale_threshold_usd,
        )
        self.whale_tracker = WhaleTracker(
            whale_threshold_usd=whale_threshold_usd,
            history_hours=history_hours,
        )

        # Stablecoin supply tracking
        self._stablecoin_supply: Dict[str, float] = {
            "USDT": 0.0,
            "USDC": 0.0,
            "DAI": 0.0,
            "BUSD": 0.0,
        }
        self._stablecoin_supply_history: List[Dict] = []

        # State history
        self._state_history: Dict[str, List[OnChainState]] = {}

    def process_flow(self, data: OnChainData) -> None:
        """Process on-chain flow data."""
        self.flow_processor.add_flow(data)

    def process_whale_activity(self, activity: WhaleActivity) -> None:
        """Process whale activity."""
        self.whale_tracker.add_activity(activity)

    def update_stablecoin_supply(
        self,
        stablecoin: str,
        supply: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update stablecoin supply data."""
        stable_key = stablecoin.upper()
        if stable_key in self._stablecoin_supply:
            self._stablecoin_supply[stable_key] = supply
            self._stablecoin_supply_history.append({
                "timestamp": timestamp or datetime.utcnow(),
                "stablecoin": stable_key,
                "supply": supply,
            })

            # Keep history bounded
            if len(self._stablecoin_supply_history) > 10000:
                self._stablecoin_supply_history = self._stablecoin_supply_history[-5000:]

    def get_state(
        self,
        asset: str,
        chain: str = "ethereum",
    ) -> OnChainState:
        """
        Get current on-chain state.

        Args:
            asset: Asset symbol
            chain: Blockchain network

        Returns:
            Current on-chain state
        """
        # Get individual features
        flow_features = self.flow_processor.get_features(asset, chain)
        whale_features = self.whale_tracker.get_features(asset)
        stablecoin_features = self._get_stablecoin_features()

        # Calculate combined metrics
        onchain_sentiment = self._calculate_onchain_sentiment(
            flow_features, whale_features, stablecoin_features
        )
        accumulation_phase = self._detect_accumulation(flow_features, whale_features)
        distribution_phase = self._detect_distribution(flow_features, whale_features)
        network_activity = self._calculate_network_activity(flow_features, whale_features)

        state = OnChainState(
            timestamp=datetime.utcnow(),
            asset=asset,
            flow_features=flow_features,
            whale_features=whale_features,
            stablecoin_features=stablecoin_features,
            onchain_sentiment=onchain_sentiment,
            accumulation_phase=accumulation_phase,
            distribution_phase=distribution_phase,
            network_activity_score=network_activity,
        )

        # Store in history
        asset_key = asset.upper()
        if asset_key not in self._state_history:
            self._state_history[asset_key] = []
        self._state_history[asset_key].append(state)

        # Keep history bounded
        if len(self._state_history[asset_key]) > 10000:
            self._state_history[asset_key] = self._state_history[asset_key][-5000:]

        return state

    def _get_stablecoin_features(self) -> StablecoinFeatures:
        """Get stablecoin-related features."""
        # Calculate supply changes (simplified - in production, use actual historical data)
        total_supply = sum(self._stablecoin_supply.values())

        # Calculate 24h change from history
        total_change = 0.0
        usdt_change = 0.0
        usdc_change = 0.0

        if len(self._stablecoin_supply_history) > 0:
            # Get oldest entry within 24h
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(hours=24)
            old_supplies = {}

            for entry in self._stablecoin_supply_history:
                if entry["timestamp"] >= cutoff:
                    if entry["stablecoin"] not in old_supplies:
                        old_supplies[entry["stablecoin"]] = entry["supply"]

            for stable, current in self._stablecoin_supply.items():
                if stable in old_supplies:
                    change = current - old_supplies[stable]
                    total_change += change
                    if stable == "USDT":
                        usdt_change = change
                    elif stable == "USDC":
                        usdc_change = change

        # Dry powder score based on stablecoin inflows to exchanges
        # Higher stablecoin on exchanges = more potential buying power
        dry_powder = min(1.0, total_supply / 150e9)  # Normalize to ~150B total

        return StablecoinFeatures(
            timestamp=datetime.utcnow(),
            total_supply_change_24h=total_change,
            usdt_supply_change=usdt_change,
            usdc_supply_change=usdc_change,
            stablecoin_exchange_inflow=0.0,  # Would need dedicated tracking
            stablecoin_exchange_outflow=0.0,
            dry_powder_score=dry_powder,
        )

    def _calculate_onchain_sentiment(
        self,
        flow: Optional[FlowFeatures],
        whale: Optional[WhaleFeatures],
        stable: Optional[StablecoinFeatures],
    ) -> float:
        """Calculate overall on-chain sentiment (-1 to 1)."""
        signals = []

        if flow:
            # Exchange outflow = bullish (hodling)
            if flow.exchange_outflow_usd > flow.exchange_inflow_usd:
                flow_signal = 0.3
            else:
                flow_signal = -0.3
            signals.append(flow_signal)

            # Net flow direction
            if flow.net_flow_usd != 0:
                net_signal = -np.sign(flow.net_flow_usd) * 0.2  # Negative because inflow = bearish
                signals.append(net_signal)

        if whale:
            # Whale sentiment
            signals.append(whale.whale_sentiment * 0.4)

            # Smart money signal
            signals.append(whale.smart_money_signal * 0.3)

        if stable:
            # Rising stablecoin supply = potential buying power
            if stable.total_supply_change_24h > 0:
                signals.append(0.1)
            elif stable.total_supply_change_24h < 0:
                signals.append(-0.1)

        return np.clip(np.sum(signals), -1, 1) if signals else 0.0

    def _detect_accumulation(
        self,
        flow: Optional[FlowFeatures],
        whale: Optional[WhaleFeatures],
    ) -> bool:
        """Detect if in accumulation phase."""
        signals = 0

        if flow:
            # Exchange outflows dominant
            if flow.exchange_outflow_usd > flow.exchange_inflow_usd * 1.2:
                signals += 1

        if whale:
            # High accumulation score
            if whale.accumulation_score > 0.6:
                signals += 1

            # More buyers than sellers
            if whale.buy_whale_count > whale.sell_whale_count:
                signals += 1

        return signals >= 2

    def _detect_distribution(
        self,
        flow: Optional[FlowFeatures],
        whale: Optional[WhaleFeatures],
    ) -> bool:
        """Detect if in distribution phase."""
        signals = 0

        if flow:
            # Exchange inflows dominant
            if flow.exchange_inflow_usd > flow.exchange_outflow_usd * 1.2:
                signals += 1

        if whale:
            # High distribution score
            if whale.distribution_score > 0.6:
                signals += 1

            # More sellers than buyers
            if whale.sell_whale_count > whale.buy_whale_count:
                signals += 1

        return signals >= 2

    def _calculate_network_activity(
        self,
        flow: Optional[FlowFeatures],
        whale: Optional[WhaleFeatures],
    ) -> float:
        """Calculate network activity score (0-1)."""
        scores = []

        if flow:
            # Transaction count
            total_txns = flow.inflow_count + flow.outflow_count
            txn_score = min(1.0, total_txns / 100)
            scores.append(txn_score)

            # Volume
            volume_score = min(1.0, np.log1p(flow.exchange_inflow_usd + flow.exchange_outflow_usd) / 20)
            scores.append(volume_score)

        if whale:
            # Active whales
            whale_activity_score = min(1.0, whale.active_whale_count / 20)
            scores.append(whale_activity_score)

        return np.mean(scores) if scores else 0.5

    def get_state_tensor(
        self,
        asset: str,
        sequence_length: int = 100,
    ) -> np.ndarray:
        """
        Get state as a tensor for neural network input.

        Args:
            asset: Asset symbol
            sequence_length: Number of historical states

        Returns:
            Tensor of shape (sequence_length, feature_dim)
        """
        asset_key = asset.upper()

        if asset_key not in self._state_history:
            feature_dim = 12 + 11 + 6 + 4  # flow + whale + stable + combined
            return np.zeros((sequence_length, feature_dim), dtype=np.float32)

        states = self._state_history[asset_key][-sequence_length:]

        feature_dim = 12 + 11 + 6 + 4
        tensor = np.zeros((sequence_length, feature_dim), dtype=np.float32)

        for i, state in enumerate(states):
            idx = sequence_length - len(states) + i
            tensor[idx] = state.to_array()

        return tensor
