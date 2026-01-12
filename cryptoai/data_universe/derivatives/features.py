"""Combined derivatives feature extraction."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from cryptoai.data_universe.derivatives.funding import (
    FundingProcessor,
    FundingFeatures,
    FundingRateData,
)
from cryptoai.data_universe.derivatives.open_interest import (
    OpenInterestProcessor,
    OpenInterestFeatures,
    OpenInterestData,
)
from cryptoai.data_universe.derivatives.liquidations import (
    LiquidationProcessor,
    LiquidationFeatures,
    LiquidationData,
)


@dataclass
class DerivativesState:
    """Combined derivatives market state."""

    timestamp: datetime
    asset: str

    funding: Optional[FundingFeatures]
    open_interest: Optional[OpenInterestFeatures]
    liquidations: Optional[LiquidationFeatures]

    # Combined metrics
    derivatives_stress_score: float  # 0-1, higher = more stress
    positioning_score: float  # -1 to 1, positive = net long sentiment
    leverage_score: float  # 0-1, higher = more leveraged market

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        funding_arr = self.funding.to_array() if self.funding else np.zeros(14, dtype=np.float32)
        oi_arr = self.open_interest.to_array() if self.open_interest else np.zeros(11, dtype=np.float32)
        liq_arr = self.liquidations.to_array() if self.liquidations else np.zeros(14, dtype=np.float32)
        combined = np.array([
            self.derivatives_stress_score,
            self.positioning_score,
            self.leverage_score,
        ], dtype=np.float32)

        return np.concatenate([funding_arr, oi_arr, liq_arr, combined])


class DerivativesFeatures:
    """
    Unified derivatives feature extraction.

    Combines funding rates, open interest, and liquidation data.
    """

    def __init__(
        self,
        exchanges: List[str] = None,
        funding_history_hours: int = 168,
        oi_history_hours: int = 168,
        liquidation_history_hours: int = 72,
    ):
        self.funding_processor = FundingProcessor(
            exchanges=exchanges,
            history_hours=funding_history_hours,
        )
        self.oi_processor = OpenInterestProcessor(
            exchanges=exchanges,
            history_hours=oi_history_hours,
        )
        self.liquidation_processor = LiquidationProcessor(
            history_hours=liquidation_history_hours,
        )

        # State history
        self._state_history: Dict[str, List[DerivativesState]] = {}

    def process_funding(self, data: FundingRateData) -> None:
        """Process funding rate data."""
        self.funding_processor.add_funding_rate(data)

    def process_open_interest(self, data: OpenInterestData) -> None:
        """Process open interest data."""
        self.oi_processor.add_open_interest(data)

    def process_liquidation(self, data: LiquidationData) -> None:
        """Process liquidation data."""
        self.liquidation_processor.add_liquidation(data)

    def get_state(self, asset: str) -> DerivativesState:
        """
        Get current derivatives state.

        Args:
            asset: Asset symbol

        Returns:
            Current derivatives state
        """
        # Get individual features
        funding = self.funding_processor.get_features(asset)
        oi = self.oi_processor.get_features(asset)
        liquidations = self.liquidation_processor.get_features(asset)

        # Calculate combined scores
        stress_score = self._calculate_stress_score(funding, oi, liquidations)
        positioning_score = self._calculate_positioning_score(funding, oi)
        leverage_score = self._calculate_leverage_score(oi, liquidations)

        state = DerivativesState(
            timestamp=datetime.utcnow(),
            asset=asset,
            funding=funding,
            open_interest=oi,
            liquidations=liquidations,
            derivatives_stress_score=stress_score,
            positioning_score=positioning_score,
            leverage_score=leverage_score,
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

    def _calculate_stress_score(
        self,
        funding: Optional[FundingFeatures],
        oi: Optional[OpenInterestFeatures],
        liquidations: Optional[LiquidationFeatures],
    ) -> float:
        """
        Calculate derivatives stress score (0-1).

        Higher = more market stress.
        """
        scores = []

        if funding:
            # Extreme funding rates indicate stress
            extreme_score = float(funding.is_extreme)
            scores.append(extreme_score)

            # High rate dispersion indicates fragmented market
            dispersion_score = min(1.0, funding.rate_dispersion * 10000 / 50)  # 50 bps max
            scores.append(dispersion_score)

        if oi:
            # Rapid OI changes indicate stress
            oi_change_score = min(1.0, abs(oi.oi_change_4h_pct) / 0.2)  # 20% change max
            scores.append(oi_change_score)

            # High positioning imbalance
            imbalance_score = abs(oi.positioning_imbalance)
            scores.append(imbalance_score)

        if liquidations:
            # High liquidation rate
            liq_rate_score = min(1.0, liquidations.liquidations_per_hour / 100)  # 100/hr max
            scores.append(liq_rate_score)

            # Cascade detection
            scores.append(liquidations.cascade_score)

            # Large liquidations
            large_score = min(1.0, liquidations.large_liquidation_count / 10)
            scores.append(large_score)

        return np.mean(scores) if scores else 0.5

    def _calculate_positioning_score(
        self,
        funding: Optional[FundingFeatures],
        oi: Optional[OpenInterestFeatures],
    ) -> float:
        """
        Calculate net positioning score (-1 to 1).

        Positive = net long sentiment, Negative = net short.
        """
        scores = []

        if funding:
            # Positive funding = longs pay shorts = net long
            funding_signal = np.clip(funding.current_rate * 1000, -1, 1)
            scores.append(funding_signal)

        if oi:
            # L/S ratio > 1 = more longs
            scores.append(oi.positioning_imbalance)

            # Top trader positioning
            if oi.top_trader_long_ratio and oi.top_trader_short_ratio:
                top_signal = oi.top_trader_long_ratio - oi.top_trader_short_ratio
                scores.append(top_signal)

        return np.mean(scores) if scores else 0.0

    def _calculate_leverage_score(
        self,
        oi: Optional[OpenInterestFeatures],
        liquidations: Optional[LiquidationFeatures],
    ) -> float:
        """
        Calculate market leverage score (0-1).

        Higher = more leveraged market (higher risk).
        """
        scores = []

        if oi:
            # OI building indicates leverage buildup
            if oi.is_oi_building:
                scores.append(0.7)
            elif oi.is_oi_unwinding:
                scores.append(0.3)
            else:
                scores.append(0.5)

        if liquidations:
            # High liquidations = high leverage being unwound
            liq_vol_score = min(1.0, np.log1p(liquidations.total_liquidations_usd / 1e6) / 5)
            scores.append(liq_vol_score)

            # Liquidation trend
            if liquidations.liquidation_trend_1h > 0.5:
                scores.append(0.8)
            elif liquidations.liquidation_trend_1h < -0.3:
                scores.append(0.3)
            else:
                scores.append(0.5)

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
            feature_dim = 14 + 11 + 14 + 3  # funding + oi + liq + combined
            return np.zeros((sequence_length, feature_dim), dtype=np.float32)

        states = self._state_history[asset_key][-sequence_length:]

        # Get feature dimension from first state
        feature_dim = 14 + 11 + 14 + 3
        tensor = np.zeros((sequence_length, feature_dim), dtype=np.float32)

        for i, state in enumerate(states):
            idx = sequence_length - len(states) + i
            tensor[idx] = state.to_array()

        return tensor

    def get_basis_contango(
        self,
        asset: str,
        spot_price: float,
        futures_prices: Dict[str, float],  # expiry -> price
    ) -> Dict[str, float]:
        """
        Calculate basis and contango/backwardation for futures.

        Args:
            asset: Asset symbol
            spot_price: Current spot price
            futures_prices: Dict of expiry dates to futures prices

        Returns:
            Dict with basis calculations for each expiry
        """
        if spot_price <= 0:
            return {}

        results = {}
        for expiry, fut_price in futures_prices.items():
            basis = (fut_price - spot_price) / spot_price
            results[f"{expiry}_basis"] = basis

            # Annualized basis
            # Assume expiry format YYYYMMDD for simple calculation
            # In production, parse actual dates
            results[f"{expiry}_basis_annual"] = basis * 4  # Rough quarterly approximation

        return results
