"""Trading executor for live execution."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio
import numpy as np
from loguru import logger

from cryptoai.execution.exchange_client import (
    ExchangeClient,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
)
from cryptoai.risk_engine import RiskController


@dataclass
class ExecutionConfig:
    """Configuration for trading executor."""

    # Execution mode
    mode: str = "shadow"  # shadow, paper, live

    # Order settings
    default_order_type: str = "limit"  # market, limit
    limit_offset_bps: float = 5.0  # Offset for limit orders
    max_slippage_bps: float = 50.0  # Maximum acceptable slippage

    # Position settings
    max_position_size_usd: float = 10000.0
    max_leverage: int = 5
    default_leverage: int = 3

    # Risk settings
    max_loss_per_trade_pct: float = 1.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0

    # Execution timing
    order_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class TradeDecision:
    """Trading decision from the AI model."""
    asset: str
    direction: float  # -1 to 1 (sell to buy)
    size_pct: float  # 0 to 1 (fraction of max position)
    confidence: float  # 0 to 1
    urgency: float  # 0 to 1 (higher = prefer market order)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    success: bool
    decision: TradeDecision
    orders: List[OrderResult] = field(default_factory=list)
    position_change: float = 0.0
    avg_price: float = 0.0
    total_fees: float = 0.0
    slippage_bps: float = 0.0
    error: Optional[str] = None


class TradingExecutor:
    """
    Executes trading decisions from the AI model.

    Handles:
    - Order routing and execution
    - Smart order splitting
    - Risk checks before execution
    - Shadow/paper/live modes
    - Execution quality tracking
    """

    def __init__(
        self,
        exchange_client: ExchangeClient,
        risk_controller: RiskController,
        config: ExecutionConfig,
    ):
        self.exchange = exchange_client
        self.risk = risk_controller
        self.config = config

        # Execution state
        self._pending_decisions: List[TradeDecision] = []
        self._execution_history: List[ExecutionResult] = []

        # Shadow mode tracking
        self._shadow_positions: Dict[str, float] = {}
        self._shadow_pnl: float = 0.0

        # Callbacks
        self._on_execution: Optional[Callable[[ExecutionResult], None]] = None

    async def execute(self, decision: TradeDecision) -> ExecutionResult:
        """
        Execute a trading decision.

        Args:
            decision: Trading decision from AI model

        Returns:
            Execution result
        """
        logger.info(
            f"Executing decision: {decision.asset} "
            f"direction={decision.direction:.2f} "
            f"size={decision.size_pct:.2%} "
            f"confidence={decision.confidence:.2f}"
        )

        # Pre-execution checks
        if not await self._pre_execution_checks(decision):
            return ExecutionResult(
                success=False,
                decision=decision,
                error="Pre-execution checks failed",
            )

        # Route based on mode
        if self.config.mode == "shadow":
            result = await self._execute_shadow(decision)
        elif self.config.mode == "paper":
            result = await self._execute_paper(decision)
        else:
            result = await self._execute_live(decision)

        # Record result
        self._execution_history.append(result)

        # Callback
        if self._on_execution:
            self._on_execution(result)

        return result

    async def _pre_execution_checks(self, decision: TradeDecision) -> bool:
        """Run pre-execution risk checks."""
        # Check confidence threshold
        if decision.confidence < 0.5:
            logger.warning(f"Low confidence: {decision.confidence:.2f}")
            return False

        # Check risk controller
        risk_check = self.risk.check_trade_allowed(
            asset=decision.asset,
            direction=1 if decision.direction > 0 else -1,
            size_pct=decision.size_pct,
        )

        if not risk_check.get("allowed", False):
            logger.warning(f"Risk check failed: {risk_check.get('reason')}")
            return False

        return True

    async def _execute_shadow(self, decision: TradeDecision) -> ExecutionResult:
        """Execute in shadow mode (simulate without real orders)."""
        # Get current market price
        # In shadow mode, we simulate at current market price

        # Calculate position change
        max_position = self.config.max_position_size_usd
        position_change = decision.direction * decision.size_pct * max_position

        # Update shadow positions
        current = self._shadow_positions.get(decision.asset, 0)
        self._shadow_positions[decision.asset] = current + position_change

        logger.info(
            f"[SHADOW] {decision.asset}: "
            f"position {current:.2f} -> {self._shadow_positions[decision.asset]:.2f}"
        )

        return ExecutionResult(
            success=True,
            decision=decision,
            position_change=position_change,
            avg_price=0.0,  # Would fetch real price in practice
            total_fees=abs(position_change) * 0.0005,  # 5bps simulated fee
            slippage_bps=0.0,
        )

    async def _execute_paper(self, decision: TradeDecision) -> ExecutionResult:
        """Execute in paper trading mode."""
        # Similar to shadow but uses exchange paper trading API
        return await self._execute_shadow(decision)

    async def _execute_live(self, decision: TradeDecision) -> ExecutionResult:
        """Execute live orders."""
        # Calculate order parameters
        order_params = await self._calculate_order_params(decision)

        if not order_params:
            return ExecutionResult(
                success=False,
                decision=decision,
                error="Failed to calculate order parameters",
            )

        # Execute orders
        orders = []
        total_filled = 0.0
        total_value = 0.0
        total_fees = 0.0

        for params in order_params:
            try:
                result = await self._execute_with_retry(params)
                if result:
                    orders.append(result)
                    total_filled += result.filled_quantity
                    if result.avg_fill_price:
                        total_value += result.filled_quantity * result.avg_fill_price
                    total_fees += result.fee

            except Exception as e:
                logger.error(f"Order execution failed: {e}")

        if not orders:
            return ExecutionResult(
                success=False,
                decision=decision,
                error="All orders failed",
            )

        # Calculate execution metrics
        avg_price = total_value / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            success=True,
            decision=decision,
            orders=orders,
            position_change=total_filled * (1 if decision.direction > 0 else -1),
            avg_price=avg_price,
            total_fees=total_fees,
        )

    async def _calculate_order_params(
        self,
        decision: TradeDecision,
    ) -> List[OrderRequest]:
        """Calculate order parameters from decision."""
        # Get current position
        position = await self.exchange.get_position(decision.asset)
        current_size = position.size if position else 0

        # Calculate target position
        max_position = self.config.max_position_size_usd
        target_size = decision.direction * decision.size_pct * max_position

        # Calculate order size
        order_size = target_size - current_size

        if abs(order_size) < 1:  # Minimum order size
            return []

        # Determine order type
        if decision.urgency > 0.8 or self.config.default_order_type == "market":
            order_type = OrderType.MARKET
        else:
            order_type = OrderType.LIMIT

        # Create order request
        order = OrderRequest(
            symbol=decision.asset,
            side=OrderSide.BUY if order_size > 0 else OrderSide.SELL,
            order_type=order_type,
            quantity=abs(order_size),
            leverage=self.config.default_leverage,
        )

        return [order]

    async def _execute_with_retry(
        self,
        order: OrderRequest,
    ) -> Optional[OrderResult]:
        """Execute order with retries."""
        for attempt in range(self.config.retry_attempts):
            try:
                result = await self.exchange.place_order(order)

                # Wait for fill if limit order
                if order.order_type == OrderType.LIMIT:
                    result = await self._wait_for_fill(
                        result.order_id,
                        order.symbol,
                        self.config.order_timeout_seconds,
                    )

                return result

            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)

        return None

    async def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: float,
    ) -> Optional[OrderResult]:
        """Wait for order to be filled."""
        from cryptoai.execution.exchange_client import OrderStatus

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            order = await self.exchange.get_order(order_id, symbol)

            if order and order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                return order

            await asyncio.sleep(0.5)

        # Cancel unfilled order
        await self.exchange.cancel_order(order_id, symbol)
        return await self.exchange.get_order(order_id, symbol)

    async def close_position(self, symbol: str) -> ExecutionResult:
        """Close position for a symbol."""
        position = await self.exchange.get_position(symbol)

        if not position:
            return ExecutionResult(
                success=True,
                decision=TradeDecision(
                    asset=symbol,
                    direction=0,
                    size_pct=0,
                    confidence=1.0,
                    urgency=1.0,
                ),
            )

        # Create close order
        order = OrderRequest(
            symbol=symbol,
            side=OrderSide.SELL if position.side == "long" else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position.size,
            reduce_only=True,
        )

        result = await self.exchange.place_order(order)

        return ExecutionResult(
            success=result.status.value in ["filled"],
            decision=TradeDecision(
                asset=symbol,
                direction=-1 if position.side == "long" else 1,
                size_pct=1.0,
                confidence=1.0,
                urgency=1.0,
            ),
            orders=[result],
            position_change=-position.size if position.side == "long" else position.size,
        )

    async def close_all_positions(self) -> List[ExecutionResult]:
        """Close all open positions."""
        positions = await self.exchange.get_positions()
        results = []

        for position in positions:
            result = await self.close_position(position.symbol)
            results.append(result)

        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {}

        successful = [e for e in self._execution_history if e.success]

        total_fees = sum(e.total_fees for e in successful)
        avg_slippage = np.mean([e.slippage_bps for e in successful]) if successful else 0

        return {
            "total_executions": len(self._execution_history),
            "successful_executions": len(successful),
            "success_rate": len(successful) / len(self._execution_history),
            "total_fees": total_fees,
            "avg_slippage_bps": avg_slippage,
            "mode": self.config.mode,
        }

    def set_callback(self, on_execution: Callable[[ExecutionResult], None]) -> None:
        """Set execution callback."""
        self._on_execution = on_execution
