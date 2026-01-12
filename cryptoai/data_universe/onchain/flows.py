"""On-chain flow processing and analysis."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import numpy as np
from loguru import logger

from cryptoai.data_universe.base import OnChainData


class FlowType(str, Enum):
    """Types of on-chain flows."""

    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    WHALE_TRANSFER = "whale_transfer"
    STABLECOIN_MINT = "stablecoin_mint"
    STABLECOIN_BURN = "stablecoin_burn"
    BRIDGE_IN = "bridge_in"
    BRIDGE_OUT = "bridge_out"
    CONTRACT_INTERACTION = "contract_interaction"


@dataclass
class FlowFeatures:
    """Extracted on-chain flow features."""

    timestamp: datetime
    asset: str
    chain: str

    # Exchange flows
    exchange_inflow_usd: float
    exchange_outflow_usd: float
    net_flow_usd: float  # Positive = inflow
    flow_ratio: float  # inflow / outflow

    # Flow counts
    inflow_count: int
    outflow_count: int

    # Whale activity
    whale_inflow_usd: float
    whale_outflow_usd: float
    whale_count: int

    # Change metrics
    net_flow_change_1h: float
    net_flow_change_24h: float

    # Pressure metrics
    exchange_supply_pressure: float  # 0-1, higher = more selling pressure

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            np.log1p(self.exchange_inflow_usd / 1e6),
            np.log1p(self.exchange_outflow_usd / 1e6),
            np.sign(self.net_flow_usd) * np.log1p(abs(self.net_flow_usd) / 1e6),
            self.flow_ratio,
            self.inflow_count,
            self.outflow_count,
            np.log1p(self.whale_inflow_usd / 1e6),
            np.log1p(self.whale_outflow_usd / 1e6),
            self.whale_count,
            self.net_flow_change_1h,
            self.net_flow_change_24h,
            self.exchange_supply_pressure,
        ], dtype=np.float32)


class OnChainFlowProcessor:
    """Process on-chain flow data and extract features."""

    def __init__(
        self,
        chains: List[str] = None,
        history_hours: int = 168,  # 7 days
        whale_threshold_usd: float = 1_000_000.0,
    ):
        self.chains = chains or ["ethereum", "bitcoin", "arbitrum", "optimism"]
        self.history_hours = history_hours
        self.whale_threshold_usd = whale_threshold_usd

        # Known exchange addresses (in production, load from database)
        self._exchange_addresses: Set[str] = set()

        # Historical flows per asset-chain
        self._history: Dict[str, deque] = {}

    def _get_key(self, asset: str, chain: str) -> str:
        """Get unique key for asset-chain pair."""
        return f"{asset.upper()}:{chain.lower()}"

    def add_exchange_address(self, address: str, exchange_name: str) -> None:
        """Register a known exchange address."""
        self._exchange_addresses.add(address.lower())

    def classify_flow(self, data: OnChainData) -> FlowType:
        """Classify the type of on-chain flow."""
        from_is_exchange = (
            data.from_address
            and data.from_address.lower() in self._exchange_addresses
        )
        to_is_exchange = (
            data.to_address
            and data.to_address.lower() in self._exchange_addresses
        )

        # Exchange flow classification
        if to_is_exchange and not from_is_exchange:
            return FlowType.EXCHANGE_INFLOW
        elif from_is_exchange and not to_is_exchange:
            return FlowType.EXCHANGE_OUTFLOW

        # Label-based classification
        if data.to_label == "exchange":
            return FlowType.EXCHANGE_INFLOW
        elif data.from_label == "exchange":
            return FlowType.EXCHANGE_OUTFLOW

        # Whale classification
        if data.amount_usd >= self.whale_threshold_usd:
            return FlowType.WHALE_TRANSFER

        return FlowType.CONTRACT_INTERACTION

    def add_flow(self, data: OnChainData) -> None:
        """Add an on-chain flow observation."""
        key = self._get_key(data.asset, data.chain)

        if key not in self._history:
            max_entries = self.history_hours * 100  # Assume max 100 flows/hour
            self._history[key] = deque(maxlen=max_entries)

        # Classify and add flow type
        data.flow_type = self.classify_flow(data).value
        self._history[key].append(data)

    def get_features(
        self,
        asset: str,
        chain: str = "ethereum",
        window_hours: float = 1.0,
    ) -> Optional[FlowFeatures]:
        """
        Get on-chain flow features.

        Args:
            asset: Asset symbol
            chain: Blockchain network
            window_hours: Time window for aggregation

        Returns:
            Flow features or None if no data
        """
        key = self._get_key(asset, chain)

        if key not in self._history or len(self._history[key]) == 0:
            return None

        history = list(self._history[key])
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [h for h in history if h.timestamp >= cutoff]

        if len(recent) == 0:
            return self._empty_features(asset, chain)

        # Classify flows
        inflows = [
            h for h in recent
            if h.flow_type == FlowType.EXCHANGE_INFLOW.value
        ]
        outflows = [
            h for h in recent
            if h.flow_type == FlowType.EXCHANGE_OUTFLOW.value
        ]
        whale_transfers = [
            h for h in recent
            if h.amount_usd >= self.whale_threshold_usd
        ]

        # Exchange flow metrics
        exchange_inflow_usd = sum(h.amount_usd for h in inflows)
        exchange_outflow_usd = sum(h.amount_usd for h in outflows)
        net_flow = exchange_inflow_usd - exchange_outflow_usd
        flow_ratio = (
            exchange_inflow_usd / exchange_outflow_usd
            if exchange_outflow_usd > 0 else float("inf")
        )

        # Whale metrics
        whale_inflows = [
            h for h in whale_transfers
            if h.flow_type == FlowType.EXCHANGE_INFLOW.value
        ]
        whale_outflows = [
            h for h in whale_transfers
            if h.flow_type == FlowType.EXCHANGE_OUTFLOW.value
        ]
        whale_inflow_usd = sum(h.amount_usd for h in whale_inflows)
        whale_outflow_usd = sum(h.amount_usd for h in whale_outflows)

        # Change metrics
        net_flow_change_1h = self._calculate_flow_change(history, hours=1)
        net_flow_change_24h = self._calculate_flow_change(history, hours=24)

        # Exchange supply pressure
        total_flow = exchange_inflow_usd + exchange_outflow_usd
        supply_pressure = exchange_inflow_usd / total_flow if total_flow > 0 else 0.5

        return FlowFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            chain=chain,
            exchange_inflow_usd=exchange_inflow_usd,
            exchange_outflow_usd=exchange_outflow_usd,
            net_flow_usd=net_flow,
            flow_ratio=min(flow_ratio, 10.0),  # Cap at 10
            inflow_count=len(inflows),
            outflow_count=len(outflows),
            whale_inflow_usd=whale_inflow_usd,
            whale_outflow_usd=whale_outflow_usd,
            whale_count=len(whale_transfers),
            net_flow_change_1h=net_flow_change_1h,
            net_flow_change_24h=net_flow_change_24h,
            exchange_supply_pressure=supply_pressure,
        )

    def _empty_features(self, asset: str, chain: str) -> FlowFeatures:
        """Return empty features when no data."""
        return FlowFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            chain=chain,
            exchange_inflow_usd=0.0,
            exchange_outflow_usd=0.0,
            net_flow_usd=0.0,
            flow_ratio=1.0,
            inflow_count=0,
            outflow_count=0,
            whale_inflow_usd=0.0,
            whale_outflow_usd=0.0,
            whale_count=0,
            net_flow_change_1h=0.0,
            net_flow_change_24h=0.0,
            exchange_supply_pressure=0.5,
        )

    def _calculate_flow_change(
        self,
        history: List[OnChainData],
        hours: int,
    ) -> float:
        """Calculate change in net flow over time."""
        if len(history) < 2:
            return 0.0

        now = datetime.utcnow()

        # Current net flow
        current_cutoff = now - timedelta(hours=1)
        current = [h for h in history if h.timestamp >= current_cutoff]
        current_net = sum(
            h.amount_usd if h.flow_type == FlowType.EXCHANGE_INFLOW.value
            else -h.amount_usd if h.flow_type == FlowType.EXCHANGE_OUTFLOW.value
            else 0
            for h in current
        )

        # Previous net flow
        prev_start = current_cutoff - timedelta(hours=hours)
        prev_end = current_cutoff
        previous = [h for h in history if prev_start <= h.timestamp < prev_end]
        prev_net = sum(
            h.amount_usd if h.flow_type == FlowType.EXCHANGE_INFLOW.value
            else -h.amount_usd if h.flow_type == FlowType.EXCHANGE_OUTFLOW.value
            else 0
            for h in previous
        )

        # Normalize change
        baseline = max(abs(prev_net), 1e6)  # Minimum $1M baseline
        return (current_net - prev_net) / baseline

    def get_aggregated_flows(
        self,
        asset: str,
        hours: int = 24,
    ) -> Dict[str, float]:
        """Get aggregated flows across all chains."""
        total_inflow = 0.0
        total_outflow = 0.0

        for chain in self.chains:
            features = self.get_features(asset, chain, window_hours=hours)
            if features:
                total_inflow += features.exchange_inflow_usd
                total_outflow += features.exchange_outflow_usd

        return {
            "total_inflow_usd": total_inflow,
            "total_outflow_usd": total_outflow,
            "net_flow_usd": total_inflow - total_outflow,
        }
