"""Main backtesting engine with event-driven simulation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from loguru import logger

from cryptoai.backtesting.simulator import MarketSimulator, SimulationConfig
from cryptoai.backtesting.metrics import BacktestMetrics, PerformanceReport


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    # Time range
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: float = 100000.0

    # Market simulation
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Walk-forward settings
    train_window_days: int = 90
    test_window_days: int = 30
    step_days: int = 30

    # Validation
    no_lookahead: bool = True
    strict_walk_forward: bool = True

    # Data frequency
    data_frequency: str = "1min"  # 1min, 5min, 15min, 1h

    # Reproducibility
    random_seed: int = 42  # Seed for deterministic simulation


@dataclass
class Order:
    """Order representation."""

    id: str
    timestamp: datetime
    asset: str
    side: str  # buy, sell
    quantity: float
    order_type: str  # market, limit
    limit_price: Optional[float] = None
    leverage: float = 1.0
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status: str = "pending"  # pending, partial, filled, cancelled
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Position tracking."""

    asset: str
    side: str  # long, short
    quantity: float
    entry_price: float
    entry_time: datetime
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    funding_paid: float = 0.0
    liquidation_price: Optional[float] = None


@dataclass
class BacktestState:
    """Current backtest state."""

    timestamp: datetime
    capital: float
    equity: float
    positions: Dict[str, Position]
    pending_orders: List[Order]
    filled_orders: List[Order]
    daily_pnl: float
    total_fees: float
    total_slippage: float
    total_funding: float
    drawdown: float
    max_equity: float


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Crypto-realistic simulation with all market frictions.
    """

    def __init__(
        self,
        config: BacktestConfig,
        market_data: Dict[str, np.ndarray],  # asset -> OHLCV data
        orderbook_data: Optional[Dict[str, np.ndarray]] = None,
        funding_data: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.config = config
        self.market_data = market_data
        self.orderbook_data = orderbook_data
        self.funding_data = funding_data

        # Initialize seeded RNG for reproducibility
        self._rng = np.random.default_rng(config.random_seed)

        # Initialize simulator
        self.simulator = MarketSimulator(config.simulation)

        # Initialize state
        self.state = BacktestState(
            timestamp=config.start_date,
            capital=config.initial_capital,
            equity=config.initial_capital,
            positions={},
            pending_orders=[],
            filled_orders=[],
            daily_pnl=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            total_funding=0.0,
            drawdown=0.0,
            max_equity=config.initial_capital,
        )

        # Track history for metrics
        self._equity_history: List[Tuple[datetime, float]] = []
        self._trade_history: List[Dict] = []
        self._daily_returns: List[float] = []

        # Order ID counter
        self._order_id = 0

        # Callbacks
        self._on_fill: Optional[Callable] = None
        self._on_liquidation: Optional[Callable] = None

    def register_callbacks(
        self,
        on_fill: Optional[Callable] = None,
        on_liquidation: Optional[Callable] = None,
    ):
        """Register event callbacks."""
        self._on_fill = on_fill
        self._on_liquidation = on_liquidation

    def submit_order(
        self,
        asset: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        leverage: float = 1.0,
    ) -> Order:
        """
        Submit an order.

        Args:
            asset: Asset to trade
            side: buy or sell
            quantity: Order quantity
            order_type: market or limit
            limit_price: Limit price for limit orders
            leverage: Leverage to use

        Returns:
            Order object
        """
        self._order_id += 1
        order = Order(
            id=f"order_{self._order_id}",
            timestamp=self.state.timestamp,
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            leverage=leverage,
        )

        self.state.pending_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for i, order in enumerate(self.state.pending_orders):
            if order.id == order_id:
                order.status = "cancelled"
                self.state.pending_orders.pop(i)
                return True
        return False

    def close_position(self, asset: str) -> Optional[Order]:
        """Close an existing position."""
        if asset not in self.state.positions:
            return None

        position = self.state.positions[asset]
        side = "sell" if position.side == "long" else "buy"

        return self.submit_order(
            asset=asset,
            side=side,
            quantity=position.quantity,
            order_type="market",
        )

    def step(self, current_price: Dict[str, float]) -> BacktestState:
        """
        Step simulation forward.

        Processes pending orders and updates state.

        Args:
            current_price: Dict of asset -> current price

        Returns:
            Updated state
        """
        # Process pending orders
        self._process_orders(current_price)

        # Update position P&L
        self._update_positions(current_price)

        # Check liquidations
        self._check_liquidations(current_price)

        # Apply funding (if 8h mark)
        self._apply_funding()

        # Update equity and drawdown
        self._update_equity()

        # Record history
        self._equity_history.append((self.state.timestamp, self.state.equity))

        return self.state

    def _process_orders(self, prices: Dict[str, float]):
        """Process pending orders."""
        still_pending = []

        for order in self.state.pending_orders:
            if order.asset not in prices:
                still_pending.append(order)
                continue

            current_price = prices[order.asset]
            can_fill = False
            fill_price = current_price

            if order.order_type == "market":
                can_fill = True
            elif order.order_type == "limit":
                if order.side == "buy" and current_price <= order.limit_price:
                    can_fill = True
                    fill_price = order.limit_price
                elif order.side == "sell" and current_price >= order.limit_price:
                    can_fill = True
                    fill_price = order.limit_price

            if can_fill:
                self._execute_fill(order, fill_price)
            else:
                still_pending.append(order)

        self.state.pending_orders = still_pending

    def _execute_fill(self, order: Order, base_price: float):
        """Execute order fill with realistic simulation."""
        # Calculate slippage
        slippage = self.simulator.calculate_slippage(
            order.quantity, order.side, base_price
        )
        fill_price = base_price * (1 + slippage) if order.side == "buy" else base_price * (1 - slippage)

        # Calculate fees
        fees = self.simulator.calculate_fees(
            order.quantity * fill_price,
            order.order_type,
        )

        # Simulate partial fills (using seeded RNG for reproducibility)
        fill_quantity = order.quantity
        if self.simulator.config.partial_fills_enabled:
            fill_ratio = self._rng.uniform(
                self.simulator.config.fill_probability, 1.0
            )
            fill_quantity = order.quantity * fill_ratio

        # Update order
        order.filled_quantity = fill_quantity
        order.filled_price = fill_price
        order.fees = fees
        order.slippage = abs(fill_price - base_price) * fill_quantity
        order.status = "filled" if fill_quantity == order.quantity else "partial"

        # Update capital
        trade_value = fill_quantity * fill_price
        self.state.capital -= fees
        self.state.total_fees += fees
        self.state.total_slippage += order.slippage

        # Update position
        self._update_position_from_fill(order)

        # Record fill
        self.state.filled_orders.append(order)
        self._record_trade(order)

        # Callback
        if self._on_fill:
            self._on_fill(order)

    def _update_position_from_fill(self, order: Order):
        """Update positions from a filled order."""
        asset = order.asset

        if asset in self.state.positions:
            position = self.state.positions[asset]

            # Same side - add to position
            if (position.side == "long" and order.side == "buy") or \
               (position.side == "short" and order.side == "sell"):
                # Average entry price
                total_qty = position.quantity + order.filled_quantity
                avg_price = (
                    position.entry_price * position.quantity +
                    order.filled_price * order.filled_quantity
                ) / total_qty
                position.quantity = total_qty
                position.entry_price = avg_price

            else:
                # Opposite side - reduce/close position
                if order.filled_quantity >= position.quantity:
                    # Close position and possibly open opposite
                    pnl = self._calculate_close_pnl(position, order.filled_price)
                    position.realized_pnl += pnl
                    self.state.capital += pnl

                    remaining = order.filled_quantity - position.quantity
                    if remaining > 0:
                        # Open opposite position
                        new_side = "long" if order.side == "buy" else "short"
                        self.state.positions[asset] = Position(
                            asset=asset,
                            side=new_side,
                            quantity=remaining,
                            entry_price=order.filled_price,
                            entry_time=self.state.timestamp,
                            leverage=order.leverage,
                        )
                    else:
                        del self.state.positions[asset]
                else:
                    # Partial close
                    close_qty = order.filled_quantity
                    pnl = self._calculate_partial_close_pnl(
                        position, close_qty, order.filled_price
                    )
                    position.realized_pnl += pnl
                    position.quantity -= close_qty
                    self.state.capital += pnl

        else:
            # New position
            side = "long" if order.side == "buy" else "short"
            self.state.positions[asset] = Position(
                asset=asset,
                side=side,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                entry_time=self.state.timestamp,
                leverage=order.leverage,
                liquidation_price=self._calculate_liquidation_price(
                    order.filled_price, side, order.leverage
                ),
            )

    def _calculate_close_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate P&L from closing a position."""
        if position.side == "long":
            return (exit_price - position.entry_price) * position.quantity * position.leverage
        else:
            return (position.entry_price - exit_price) * position.quantity * position.leverage

    def _calculate_partial_close_pnl(
        self, position: Position, close_qty: float, exit_price: float
    ) -> float:
        """Calculate P&L from partial close."""
        if position.side == "long":
            return (exit_price - position.entry_price) * close_qty * position.leverage
        else:
            return (position.entry_price - exit_price) * close_qty * position.leverage

    def _calculate_liquidation_price(
        self, entry_price: float, side: str, leverage: float
    ) -> float:
        """Calculate liquidation price."""
        maint_margin = self.config.simulation.maintenance_margin
        if side == "long":
            return entry_price * (1 - 1 / leverage + maint_margin)
        else:
            return entry_price * (1 + 1 / leverage - maint_margin)

    def _update_positions(self, prices: Dict[str, float]):
        """Update position unrealized P&L."""
        for asset, position in self.state.positions.items():
            if asset in prices:
                current_price = prices[asset]
                if position.side == "long":
                    position.unrealized_pnl = (
                        (current_price - position.entry_price) *
                        position.quantity * position.leverage
                    )
                else:
                    position.unrealized_pnl = (
                        (position.entry_price - current_price) *
                        position.quantity * position.leverage
                    )

    def _check_liquidations(self, prices: Dict[str, float]):
        """Check for liquidations."""
        to_liquidate = []

        for asset, position in self.state.positions.items():
            if asset in prices and position.liquidation_price:
                current_price = prices[asset]
                is_liquidated = False

                if position.side == "long" and current_price <= position.liquidation_price:
                    is_liquidated = True
                elif position.side == "short" and current_price >= position.liquidation_price:
                    is_liquidated = True

                if is_liquidated:
                    to_liquidate.append(asset)

        for asset in to_liquidate:
            position = self.state.positions[asset]
            loss = -abs(position.unrealized_pnl) * 1.05  # Extra 5% liquidation fee
            self.state.capital += loss
            logger.warning(f"Liquidation: {asset} position, loss: {loss:.2f}")

            if self._on_liquidation:
                self._on_liquidation(position)

            del self.state.positions[asset]

    def _apply_funding(self):
        """Apply funding rates (every 8 hours)."""
        if self.funding_data is None:
            return

        # Check if at 8h boundary (0, 8, 16 hours)
        hour = self.state.timestamp.hour
        if hour in [0, 8, 16] and self.state.timestamp.minute == 0:
            for asset, position in self.state.positions.items():
                if asset in self.funding_data:
                    # Get funding rate from historical data
                    # funding_data is expected to be array with columns: [timestamp, rate]
                    # or a dict mapping timestamps to rates
                    asset_funding = self.funding_data[asset]

                    if isinstance(asset_funding, dict):
                        # Dict mapping timestamps to rates
                        funding_rate = asset_funding.get(self.state.timestamp, 0.0001)
                    elif isinstance(asset_funding, np.ndarray):
                        # Find closest timestamp in funding data
                        # Assumes funding_data[asset] has shape (N, 2) with [timestamp_idx, rate]
                        # or just (N,) with rates at regular 8h intervals
                        if asset_funding.ndim == 1:
                            # Calculate which funding period we're in
                            start_ts = self.config.start_date.timestamp()
                            current_ts = self.state.timestamp.timestamp()
                            hours_elapsed = (current_ts - start_ts) / 3600
                            funding_idx = int(hours_elapsed // 8)
                            if 0 <= funding_idx < len(asset_funding):
                                funding_rate = float(asset_funding[funding_idx])
                            else:
                                funding_rate = 0.0001  # Default if out of range
                        else:
                            # 2D array: find matching timestamp
                            funding_rate = 0.0001  # Default
                            for row in asset_funding:
                                if len(row) >= 2 and abs(row[0] - self.state.timestamp.timestamp()) < 3600:
                                    funding_rate = float(row[1])
                                    break
                    else:
                        # Fallback default
                        funding_rate = 0.0001
                        logger.warning(f"Unknown funding_data format for {asset}, using default rate")

                    # Long pays positive funding, short receives
                    position_value = position.quantity * position.entry_price
                    if position.side == "long":
                        funding_cost = position_value * funding_rate
                    else:
                        funding_cost = -position_value * funding_rate

                    position.funding_paid += funding_cost
                    self.state.total_funding += funding_cost
                    self.state.capital -= funding_cost

    def _update_equity(self):
        """Update equity and drawdown."""
        unrealized = sum(p.unrealized_pnl for p in self.state.positions.values())
        self.state.equity = self.state.capital + unrealized

        self.state.max_equity = max(self.state.max_equity, self.state.equity)
        self.state.drawdown = (
            (self.state.max_equity - self.state.equity) / self.state.max_equity
            if self.state.max_equity > 0 else 0
        )

    def _record_trade(self, order: Order):
        """Record trade for analysis."""
        self._trade_history.append({
            "timestamp": self.state.timestamp,
            "asset": order.asset,
            "side": order.side,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "fees": order.fees,
            "slippage": order.slippage,
        })

    def get_metrics(self) -> BacktestMetrics:
        """Get backtest performance metrics."""
        return BacktestMetrics.from_equity_history(
            self._equity_history,
            self._trade_history,
            self.config.initial_capital,
        )

    def reset(self):
        """Reset backtest state."""
        self.state = BacktestState(
            timestamp=self.config.start_date,
            capital=self.config.initial_capital,
            equity=self.config.initial_capital,
            positions={},
            pending_orders=[],
            filled_orders=[],
            daily_pnl=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            total_funding=0.0,
            drawdown=0.0,
            max_equity=self.config.initial_capital,
        )
        self._equity_history = []
        self._trade_history = []
        self._daily_returns = []
        self._order_id = 0
        # Reset RNG to ensure reproducibility on re-runs
        self._rng = np.random.default_rng(self.config.random_seed)
