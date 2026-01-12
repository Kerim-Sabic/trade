"""Order management and lifecycle tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import asyncio
from loguru import logger

from cryptoai.execution.exchange_client import (
    ExchangeClient,
    OrderResult,
    OrderStatus,
)


class OrderState(Enum):
    """Order lifecycle state."""
    CREATED = "created"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class ManagedOrder:
    """Order with lifecycle management."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    order_type: str
    state: OrderState = OrderState.CREATED
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """
    Manages order lifecycle and tracking.

    Handles:
    - Order state tracking
    - Order timeout management
    - Fill tracking
    - Order history
    """

    def __init__(
        self,
        exchange_client: ExchangeClient,
        order_timeout_seconds: float = 60.0,
    ):
        self.exchange = exchange_client
        self.order_timeout = order_timeout_seconds
        self._lock = threading.Lock()

        # Order storage
        self._orders: Dict[str, ManagedOrder] = {}
        self._order_history: List[ManagedOrder] = []

        # Callbacks
        self._on_fill: Optional[Callable[[ManagedOrder], None]] = None
        self._on_cancel: Optional[Callable[[ManagedOrder], None]] = None

        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None

    def track_order(self, order_result: OrderResult, metadata: Optional[Dict] = None) -> ManagedOrder:
        """Start tracking an order."""
        managed = ManagedOrder(
            order_id=order_result.order_id,
            client_order_id=order_result.client_order_id or "",
            symbol=order_result.symbol,
            side=order_result.side.value,
            quantity=order_result.quantity,
            price=order_result.price,
            order_type=order_result.order_type.value,
            state=self._map_status(order_result.status),
            filled_quantity=order_result.filled_quantity,
            avg_fill_price=order_result.avg_fill_price or 0.0,
            fees=order_result.fee,
            metadata=metadata or {},
        )

        with self._lock:
            self._orders[order_result.order_id] = managed

        logger.info(f"Tracking order: {order_result.order_id}")
        return managed

    def _map_status(self, status: OrderStatus) -> OrderState:
        """Map exchange status to order state."""
        mapping = {
            OrderStatus.PENDING: OrderState.SUBMITTED,
            OrderStatus.OPEN: OrderState.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED: OrderState.PARTIALLY_FILLED,
            OrderStatus.FILLED: OrderState.FILLED,
            OrderStatus.CANCELLED: OrderState.CANCELLED,
            OrderStatus.REJECTED: OrderState.REJECTED,
            OrderStatus.EXPIRED: OrderState.EXPIRED,
        }
        return mapping.get(status, OrderState.ERROR)

    async def update_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Update order status from exchange."""
        with self._lock:
            managed = self._orders.get(order_id)
            if not managed:
                return None

        # Fetch current status
        order_result = await self.exchange.get_order(order_id, managed.symbol)

        if not order_result:
            return managed

        # Update managed order
        with self._lock:
            managed.state = self._map_status(order_result.status)
            managed.filled_quantity = order_result.filled_quantity
            managed.avg_fill_price = order_result.avg_fill_price or 0.0
            managed.fees = order_result.fee
            managed.updated_at = datetime.utcnow()

        # Trigger callbacks
        if managed.state == OrderState.FILLED and self._on_fill:
            self._on_fill(managed)
        elif managed.state == OrderState.CANCELLED and self._on_cancel:
            self._on_cancel(managed)

        return managed

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        with self._lock:
            managed = self._orders.get(order_id)
            if not managed:
                return False

        success = await self.exchange.cancel_order(order_id, managed.symbol)

        if success:
            await self.update_order(order_id)

        return success

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        cancelled = 0

        with self._lock:
            orders = list(self._orders.values())

        for order in orders:
            if order.state not in [OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED]:
                if symbol is None or order.symbol == symbol:
                    if await self.cancel_order(order.order_id):
                        cancelled += 1

        return cancelled

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by ID."""
        with self._lock:
            return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[ManagedOrder]:
        """Get all open orders."""
        open_states = [
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
        ]

        with self._lock:
            orders = [
                o for o in self._orders.values()
                if o.state in open_states
            ]

            if symbol:
                orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_filled_orders(self, hours: int = 24) -> List[ManagedOrder]:
        """Get filled orders within timeframe."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            return [
                o for o in self._orders.values()
                if o.state == OrderState.FILLED and o.updated_at > cutoff
            ]

    async def start_monitoring(self) -> None:
        """Start order monitoring loop."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop order monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self._check_orders()
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order monitor error: {e}")
                await asyncio.sleep(5.0)

    async def _check_orders(self) -> None:
        """Check all open orders."""
        open_orders = self.get_open_orders()

        for order in open_orders:
            # Update status
            await self.update_order(order.order_id)

            # Check timeout
            age = (datetime.utcnow() - order.created_at).total_seconds()
            if age > self.order_timeout:
                logger.warning(f"Order timeout: {order.order_id}")
                await self.cancel_order(order.order_id)

    def archive_completed_orders(self) -> int:
        """Move completed orders to history."""
        completed_states = [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        ]

        archived = 0

        with self._lock:
            to_archive = [
                order_id for order_id, order in self._orders.items()
                if order.state in completed_states
            ]

            for order_id in to_archive:
                order = self._orders.pop(order_id)
                self._order_history.append(order)
                archived += 1

        return archived

    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        with self._lock:
            all_orders = list(self._orders.values()) + self._order_history

        if not all_orders:
            return {}

        filled = [o for o in all_orders if o.state == OrderState.FILLED]
        cancelled = [o for o in all_orders if o.state == OrderState.CANCELLED]
        rejected = [o for o in all_orders if o.state == OrderState.REJECTED]

        total_volume = sum(o.filled_quantity * o.avg_fill_price for o in filled)
        total_fees = sum(o.fees for o in filled)

        return {
            "total_orders": len(all_orders),
            "filled_orders": len(filled),
            "cancelled_orders": len(cancelled),
            "rejected_orders": len(rejected),
            "fill_rate": len(filled) / len(all_orders) if all_orders else 0,
            "total_volume": total_volume,
            "total_fees": total_fees,
            "avg_fill_rate": sum(o.filled_quantity / o.quantity for o in filled) / len(filled) if filled else 0,
        }

    def set_callbacks(
        self,
        on_fill: Optional[Callable[[ManagedOrder], None]] = None,
        on_cancel: Optional[Callable[[ManagedOrder], None]] = None,
    ) -> None:
        """Set order callbacks."""
        self._on_fill = on_fill
        self._on_cancel = on_cancel
