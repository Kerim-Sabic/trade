"""Market simulator with realistic crypto market frictions."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class SimulationConfig:
    """Market simulation configuration."""

    # Fees
    maker_fee: float = 0.0002  # 2 bps
    taker_fee: float = 0.0005  # 5 bps

    # Slippage
    slippage_model: str = "orderbook_depth"  # fixed, linear, orderbook_depth
    fixed_slippage: float = 0.0001  # 1 bp
    impact_coefficient: float = 0.0001

    # Latency
    order_submission_ms: int = 50
    market_data_ms: int = 20

    # Partial fills
    partial_fills_enabled: bool = True
    fill_probability: float = 0.85

    # Funding
    funding_enabled: bool = True
    funding_interval_hours: int = 8

    # Liquidation
    liquidation_enabled: bool = True
    maintenance_margin: float = 0.05  # 5%


class MarketSimulator:
    """
    Simulates realistic crypto market conditions.

    Handles:
    - Order execution with slippage
    - Fee calculation
    - Latency simulation
    - Partial fills
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

        # Order book state (for slippage calculation)
        self._orderbook_depth: Dict[str, float] = {}

    def set_orderbook_depth(self, asset: str, depth: float):
        """Set order book depth for an asset (USD available at best levels)."""
        self._orderbook_depth[asset] = depth

    def calculate_fees(
        self,
        notional_value: float,
        order_type: str,
    ) -> float:
        """
        Calculate trading fees.

        Args:
            notional_value: Order value in USD
            order_type: market or limit

        Returns:
            Fee amount in USD
        """
        if order_type == "market":
            return notional_value * self.config.taker_fee
        else:
            return notional_value * self.config.maker_fee

    def calculate_slippage(
        self,
        quantity: float,
        side: str,
        price: float,
        asset: str = None,
    ) -> float:
        """
        Calculate price slippage.

        Args:
            quantity: Order quantity
            side: buy or sell
            price: Base price
            asset: Asset being traded

        Returns:
            Slippage as a fraction (e.g., 0.001 = 0.1%)
        """
        order_value = quantity * price

        if self.config.slippage_model == "fixed":
            return self.config.fixed_slippage

        elif self.config.slippage_model == "linear":
            return self.config.impact_coefficient * order_value

        elif self.config.slippage_model == "orderbook_depth":
            # Slippage increases with order size relative to available depth
            depth = self._orderbook_depth.get(asset, 1_000_000)  # Default 1M USD
            if depth <= 0:
                return self.config.fixed_slippage

            # Square root impact model
            impact = self.config.impact_coefficient * np.sqrt(order_value / depth)
            return min(impact, 0.01)  # Cap at 1%

        return self.config.fixed_slippage

    def simulate_fill_time(self, order_type: str) -> float:
        """
        Simulate order fill time in milliseconds.

        Returns time until fill for market orders.
        """
        base_time = self.config.order_submission_ms

        if order_type == "market":
            # Add random network jitter
            jitter = np.random.exponential(10)
            return base_time + jitter
        else:
            # Limit orders have variable fill time
            return base_time + np.random.exponential(100)

    def simulate_partial_fill(
        self,
        quantity: float,
        available_liquidity: float,
    ) -> tuple[float, bool]:
        """
        Simulate partial fill based on available liquidity.

        Returns:
            Tuple of (filled_quantity, is_complete)
        """
        if not self.config.partial_fills_enabled:
            return quantity, True

        # Check if order exceeds available liquidity
        if quantity > available_liquidity:
            # Partial fill
            fill_ratio = available_liquidity / quantity
            fill_ratio *= np.random.uniform(0.8, 1.0)  # Some randomness
            return quantity * fill_ratio, False

        # Full fill with some probability
        if np.random.random() < self.config.fill_probability:
            return quantity, True
        else:
            # Partial fill
            fill_ratio = np.random.uniform(0.7, 0.99)
            return quantity * fill_ratio, False

    def calculate_liquidation_price(
        self,
        entry_price: float,
        side: str,
        leverage: float,
    ) -> float:
        """
        Calculate liquidation price.

        Args:
            entry_price: Position entry price
            side: long or short
            leverage: Position leverage

        Returns:
            Liquidation price
        """
        mm = self.config.maintenance_margin

        if side == "long":
            # Liquidated when loss exceeds (1/leverage - mm) of position
            return entry_price * (1 - (1 / leverage) + mm)
        else:
            return entry_price * (1 + (1 / leverage) - mm)

    def calculate_funding_payment(
        self,
        position_value: float,
        funding_rate: float,
        side: str,
    ) -> float:
        """
        Calculate funding payment.

        Args:
            position_value: Position notional value
            funding_rate: Current funding rate
            side: long or short

        Returns:
            Funding payment (positive = pay, negative = receive)
        """
        if side == "long":
            return position_value * funding_rate
        else:
            return -position_value * funding_rate


class OrderBookSimulator:
    """
    Simulates order book dynamics for more realistic execution.
    """

    def __init__(
        self,
        num_levels: int = 20,
        tick_size: float = 0.01,
        base_depth_usd: float = 100000,
    ):
        self.num_levels = num_levels
        self.tick_size = tick_size
        self.base_depth_usd = base_depth_usd

    def generate_orderbook(
        self,
        mid_price: float,
        spread_bps: float = 5,
        imbalance: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a simulated order book.

        Args:
            mid_price: Mid price
            spread_bps: Spread in basis points
            imbalance: Order book imbalance (-1 to 1)

        Returns:
            Dict with bids and asks arrays (price, quantity)
        """
        spread = mid_price * spread_bps / 10000
        half_spread = spread / 2

        # Generate price levels
        bid_prices = np.array([
            mid_price - half_spread - i * self.tick_size
            for i in range(self.num_levels)
        ])
        ask_prices = np.array([
            mid_price + half_spread + i * self.tick_size
            for i in range(self.num_levels)
        ])

        # Generate quantities with exponential decay from best
        base_qty = self.base_depth_usd / mid_price
        decay = 0.9

        bid_base = base_qty * (1 + imbalance * 0.5)
        ask_base = base_qty * (1 - imbalance * 0.5)

        bid_qtys = np.array([
            bid_base * (decay ** i) * np.random.uniform(0.8, 1.2)
            for i in range(self.num_levels)
        ])
        ask_qtys = np.array([
            ask_base * (decay ** i) * np.random.uniform(0.8, 1.2)
            for i in range(self.num_levels)
        ])

        return {
            "bids": np.column_stack([bid_prices, bid_qtys]),
            "asks": np.column_stack([ask_prices, ask_qtys]),
        }

    def calculate_execution_price(
        self,
        orderbook: Dict[str, np.ndarray],
        side: str,
        quantity: float,
    ) -> tuple[float, float]:
        """
        Calculate execution price walking through order book.

        Returns:
            Tuple of (average_price, total_slippage)
        """
        if side == "buy":
            levels = orderbook["asks"]
        else:
            levels = orderbook["bids"]

        remaining = quantity
        total_value = 0.0

        for price, qty in levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, qty)
            total_value += fill_qty * price
            remaining -= fill_qty

        if quantity > 0:
            avg_price = total_value / (quantity - remaining)
        else:
            avg_price = levels[0, 0]

        # Calculate slippage from best price
        best_price = levels[0, 0]
        slippage = abs(avg_price - best_price) / best_price

        return avg_price, slippage
