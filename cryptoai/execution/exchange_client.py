"""Exchange client interfaces for order execution."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import hmac
import hashlib
import time
from loguru import logger


class ExchangeType(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderRequest:
    """Order request."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    reduce_only: bool = False
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    client_order_id: Optional[str] = None
    leverage: Optional[int] = None


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float
    price: Optional[float]
    avg_fill_price: Optional[float]
    fee: float
    fee_currency: str
    timestamp: datetime
    raw_response: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Current position."""
    symbol: str
    side: str  # long, short
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: Optional[float]
    margin: float


class ExchangeClient(ABC):
    """
    Abstract base class for exchange clients.

    Provides unified interface for different exchanges.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._session = None

    @abstractmethod
    async def connect(self) -> None:
        """Connect to exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        pass

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[OrderResult]:
        """Get order status."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Get open orders."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        pass


class BinanceClient(ExchangeClient):
    """Binance Futures client implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ):
        super().__init__(api_key, api_secret, testnet)

        self.base_url = (
            "https://testnet.binancefuture.com"
            if testnet
            else "https://fapi.binance.com"
        )
        self.ws_url = (
            "wss://stream.binancefuture.com"
            if testnet
            else "wss://fstream.binance.com"
        )

    def _sign(self, params: Dict) -> str:
        """Create signature for request."""
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def connect(self) -> None:
        """Connect to Binance."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key}
            )
            logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
        except ImportError:
            logger.error("aiohttp not installed")

    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = True,
    ) -> Dict:
        """Make API request."""
        if not self._session:
            await self.connect()

        params = params or {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)

        url = f"{self.base_url}{endpoint}"

        if method == "GET":
            async with self._session.get(url, params=params) as resp:
                return await resp.json()
        elif method == "POST":
            async with self._session.post(url, params=params) as resp:
                return await resp.json()
        elif method == "DELETE":
            async with self._session.delete(url, params=params) as resp:
                return await resp.json()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order on Binance."""
        params = {
            "symbol": request.symbol,
            "side": request.side.value.upper(),
            "type": request.order_type.value.upper(),
            "quantity": request.quantity,
        }

        if request.price:
            params["price"] = request.price
            params["timeInForce"] = request.time_in_force

        if request.stop_price:
            params["stopPrice"] = request.stop_price

        if request.reduce_only:
            params["reduceOnly"] = "true"

        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id

        response = await self._request("POST", "/fapi/v1/order", params)

        return self._parse_order_response(response)

    def _parse_order_response(self, response: Dict) -> OrderResult:
        """Parse order response from Binance."""
        status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }

        return OrderResult(
            order_id=str(response.get("orderId")),
            client_order_id=response.get("clientOrderId"),
            symbol=response.get("symbol"),
            side=OrderSide(response.get("side", "").lower()),
            order_type=OrderType(response.get("type", "").lower()),
            status=status_map.get(response.get("status"), OrderStatus.PENDING),
            quantity=float(response.get("origQty", 0)),
            filled_quantity=float(response.get("executedQty", 0)),
            price=float(response.get("price")) if response.get("price") else None,
            avg_fill_price=float(response.get("avgPrice")) if response.get("avgPrice") else None,
            fee=0.0,  # Would need to parse from fills
            fee_currency="USDT",
            timestamp=datetime.utcnow(),
            raw_response=response,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance."""
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }
        try:
            await self._request("DELETE", "/fapi/v1/order", params)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[OrderResult]:
        """Get order from Binance."""
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }
        response = await self._request("GET", "/fapi/v1/order", params)
        return self._parse_order_response(response)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Get open orders from Binance."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        response = await self._request("GET", "/fapi/v1/openOrders", params)
        return [self._parse_order_response(o) for o in response]

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position from Binance."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def get_positions(self) -> List[Position]:
        """Get all positions from Binance."""
        response = await self._request("GET", "/fapi/v2/positionRisk")

        positions = []
        for pos in response:
            if float(pos.get("positionAmt", 0)) != 0:
                positions.append(Position(
                    symbol=pos["symbol"],
                    side="long" if float(pos["positionAmt"]) > 0 else "short",
                    size=abs(float(pos["positionAmt"])),
                    entry_price=float(pos["entryPrice"]),
                    mark_price=float(pos["markPrice"]),
                    unrealized_pnl=float(pos["unRealizedProfit"]),
                    leverage=int(pos["leverage"]),
                    liquidation_price=float(pos["liquidationPrice"]) if pos["liquidationPrice"] else None,
                    margin=float(pos.get("isolatedMargin", 0)),
                ))

        return positions

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance from Binance."""
        response = await self._request("GET", "/fapi/v2/balance")
        return {
            asset["asset"]: float(asset["availableBalance"])
            for asset in response
        }

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage on Binance."""
        params = {
            "symbol": symbol,
            "leverage": leverage,
        }
        try:
            await self._request("POST", "/fapi/v1/leverage", params)
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False


def create_exchange_client(
    exchange: ExchangeType,
    api_key: str,
    api_secret: str,
    testnet: bool = True,
) -> ExchangeClient:
    """Factory function to create exchange client."""
    if exchange == ExchangeType.BINANCE:
        return BinanceClient(api_key, api_secret, testnet)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")
