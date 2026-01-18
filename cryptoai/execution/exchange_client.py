"""
Exchange client interfaces for order execution.

Supports: Binance, OKX, Bybit
Windows 11 Compatible with proper async handling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import hmac
import hashlib
import base64
import time
import json
from loguru import logger

# Platform detection
import sys
IS_WINDOWS = sys.platform == "win32"


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


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 1200
    order_requests_per_second: int = 10
    weight_per_request: int = 1


class ExchangeClient(ABC):
    """
    Abstract base class for exchange clients.

    Provides unified interface for different exchanges with:
    - Automatic rate limiting
    - Exponential backoff retry
    - Reconnection logic
    - Windows compatibility
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.rate_limit = rate_limit or RateLimitConfig()
        self._session = None
        self._ws = None
        self._ws_callbacks: Dict[str, List[Callable]] = {}
        self._last_request_time = 0.0
        self._request_count = 0
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

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

    async def _rate_limit_wait(self):
        """Wait for rate limit."""
        now = time.time()
        min_interval = 60.0 / self.rate_limit.requests_per_minute
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> Optional[Dict]:
        """Make HTTP request with exponential backoff retry."""
        await self._rate_limit_wait()

        if not self._session:
            await self.connect()

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    async with self._session.get(url, params=params, headers=headers) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            text = await resp.text()
                            logger.error(f"API error {resp.status}: {text}")
                            if resp.status >= 500:
                                await asyncio.sleep(2 ** attempt)
                            else:
                                return None

                elif method == "POST":
                    async with self._session.post(url, params=params, json=data, headers=headers) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            text = await resp.text()
                            logger.error(f"API error {resp.status}: {text}")
                            if resp.status >= 500:
                                await asyncio.sleep(2 ** attempt)
                            else:
                                return None

                elif method == "DELETE":
                    async with self._session.delete(url, params=params, headers=headers) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                        else:
                            return None

            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None


class BinanceClient(ExchangeClient):
    """Binance Futures client implementation with WebSocket support."""

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
            "wss://stream.binancefuture.com/ws"
            if testnet
            else "wss://fstream.binance.com/ws"
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
            self._connected = True
            logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
        except ImportError:
            logger.error("aiohttp not installed")

    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = True,
    ) -> Dict:
        """Make API request."""
        params = params or {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)

        url = f"{self.base_url}{endpoint}"

        return await self._request_with_retry(method, url, params=params)

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
        if not response:
            return OrderResult(
                order_id="error",
                client_order_id=None,
                symbol="",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.REJECTED,
                quantity=0,
                filled_quantity=0,
                price=None,
                avg_fill_price=None,
                fee=0.0,
                fee_currency="USDT",
                timestamp=datetime.utcnow(),
                raw_response=response or {},
            )

        status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }

        side_str = response.get("side", "BUY").lower()
        order_type_str = response.get("type", "MARKET").lower()

        return OrderResult(
            order_id=str(response.get("orderId")),
            client_order_id=response.get("clientOrderId"),
            symbol=response.get("symbol"),
            side=OrderSide(side_str) if side_str in ["buy", "sell"] else OrderSide.BUY,
            order_type=OrderType(order_type_str) if order_type_str in ["market", "limit"] else OrderType.MARKET,
            status=status_map.get(response.get("status"), OrderStatus.PENDING),
            quantity=float(response.get("origQty", 0)),
            filled_quantity=float(response.get("executedQty", 0)),
            price=float(response.get("price")) if response.get("price") else None,
            avg_fill_price=float(response.get("avgPrice")) if response.get("avgPrice") else None,
            fee=0.0,
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
        return self._parse_order_response(response) if response else None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Get open orders from Binance."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        response = await self._request("GET", "/fapi/v1/openOrders", params)
        if not response:
            return []
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

        if not response:
            return []

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
                    liquidation_price=float(pos["liquidationPrice"]) if pos.get("liquidationPrice") else None,
                    margin=float(pos.get("isolatedMargin", 0)),
                ))

        return positions

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance from Binance."""
        response = await self._request("GET", "/fapi/v2/balance")
        if not response:
            return {}
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


class OKXClient(ExchangeClient):
    """OKX exchange client implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str = "",
        testnet: bool = True,
    ):
        super().__init__(api_key, api_secret, testnet)
        self.passphrase = passphrase

        self.base_url = (
            "https://www.okx.com"
            if not testnet
            else "https://www.okx.com"  # OKX uses same URL with flag
        )
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Create OKX signature."""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        )
        return base64.b64encode(signature.digest()).decode()

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Get OKX request headers."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        signature = self._sign(timestamp, method, path, body)

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

        if self.testnet:
            headers["x-simulated-trading"] = "1"

        return headers

    async def connect(self) -> None:
        """Connect to OKX."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
            self._connected = True
            logger.info(f"Connected to OKX {'testnet' if self.testnet else 'mainnet'}")
        except ImportError:
            logger.error("aiohttp not installed")

    async def disconnect(self) -> None:
        """Disconnect from OKX."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make OKX API request."""
        body = json.dumps(data) if data else ""
        headers = self._get_headers(method, path, body)

        url = f"{self.base_url}{path}"

        if method == "GET" and params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
            headers = self._get_headers(method, f"{path}?{query}")

        result = await self._request_with_retry(
            method, url, params=None if data else params,
            data=data, headers=headers
        )

        if result and result.get("code") == "0":
            return result.get("data", result)

        logger.error(f"OKX API error: {result}")
        return None

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order on OKX."""
        data = {
            "instId": request.symbol.replace("USDT", "-USDT-SWAP"),
            "tdMode": "cross",
            "side": request.side.value,
            "ordType": "market" if request.order_type == OrderType.MARKET else "limit",
            "sz": str(request.quantity),
        }

        if request.price and request.order_type == OrderType.LIMIT:
            data["px"] = str(request.price)

        if request.reduce_only:
            data["reduceOnly"] = True

        if request.client_order_id:
            data["clOrdId"] = request.client_order_id

        response = await self._request("POST", "/api/v5/trade/order", data=data)

        if response and len(response) > 0:
            order_data = response[0]
            return OrderResult(
                order_id=order_data.get("ordId", ""),
                client_order_id=order_data.get("clOrdId"),
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.OPEN if order_data.get("sCode") == "0" else OrderStatus.REJECTED,
                quantity=request.quantity,
                filled_quantity=0,
                price=request.price,
                avg_fill_price=None,
                fee=0.0,
                fee_currency="USDT",
                timestamp=datetime.utcnow(),
                raw_response=order_data,
            )

        return OrderResult(
            order_id="error",
            client_order_id=None,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.REJECTED,
            quantity=request.quantity,
            filled_quantity=0,
            price=None,
            avg_fill_price=None,
            fee=0.0,
            fee_currency="USDT",
            timestamp=datetime.utcnow(),
            raw_response={},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on OKX."""
        data = {
            "instId": symbol.replace("USDT", "-USDT-SWAP"),
            "ordId": order_id,
        }
        response = await self._request("POST", "/api/v5/trade/cancel-order", data=data)
        return response is not None

    async def get_order(self, order_id: str, symbol: str) -> Optional[OrderResult]:
        """Get order from OKX."""
        params = {
            "instId": symbol.replace("USDT", "-USDT-SWAP"),
            "ordId": order_id,
        }
        response = await self._request("GET", "/api/v5/trade/order", params=params)
        if response and len(response) > 0:
            order_data = response[0]
            status_map = {
                "live": OrderStatus.OPEN,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "filled": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
            }
            return OrderResult(
                order_id=order_data.get("ordId", ""),
                client_order_id=order_data.get("clOrdId"),
                symbol=symbol,
                side=OrderSide(order_data.get("side", "buy")),
                order_type=OrderType.MARKET if order_data.get("ordType") == "market" else OrderType.LIMIT,
                status=status_map.get(order_data.get("state"), OrderStatus.PENDING),
                quantity=float(order_data.get("sz", 0)),
                filled_quantity=float(order_data.get("accFillSz", 0)),
                price=float(order_data.get("px")) if order_data.get("px") else None,
                avg_fill_price=float(order_data.get("avgPx")) if order_data.get("avgPx") else None,
                fee=float(order_data.get("fee", 0)),
                fee_currency="USDT",
                timestamp=datetime.utcnow(),
                raw_response=order_data,
            )
        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Get open orders from OKX."""
        params = {"instType": "SWAP"}
        if symbol:
            params["instId"] = symbol.replace("USDT", "-USDT-SWAP")

        response = await self._request("GET", "/api/v5/trade/orders-pending", params=params)
        if not response:
            return []

        orders = []
        for order_data in response:
            orders.append(OrderResult(
                order_id=order_data.get("ordId", ""),
                client_order_id=order_data.get("clOrdId"),
                symbol=order_data.get("instId", "").replace("-USDT-SWAP", "USDT"),
                side=OrderSide(order_data.get("side", "buy")),
                order_type=OrderType.MARKET if order_data.get("ordType") == "market" else OrderType.LIMIT,
                status=OrderStatus.OPEN,
                quantity=float(order_data.get("sz", 0)),
                filled_quantity=float(order_data.get("accFillSz", 0)),
                price=float(order_data.get("px")) if order_data.get("px") else None,
                avg_fill_price=None,
                fee=0.0,
                fee_currency="USDT",
                timestamp=datetime.utcnow(),
                raw_response=order_data,
            ))
        return orders

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position from OKX."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def get_positions(self) -> List[Position]:
        """Get all positions from OKX."""
        response = await self._request("GET", "/api/v5/account/positions", params={"instType": "SWAP"})
        if not response:
            return []

        positions = []
        for pos in response:
            size = float(pos.get("pos", 0))
            if size != 0:
                positions.append(Position(
                    symbol=pos.get("instId", "").replace("-USDT-SWAP", "USDT"),
                    side="long" if size > 0 else "short",
                    size=abs(size),
                    entry_price=float(pos.get("avgPx", 0)),
                    mark_price=float(pos.get("markPx", 0)),
                    unrealized_pnl=float(pos.get("upl", 0)),
                    leverage=int(float(pos.get("lever", 1))),
                    liquidation_price=float(pos.get("liqPx")) if pos.get("liqPx") else None,
                    margin=float(pos.get("margin", 0)),
                ))
        return positions

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance from OKX."""
        response = await self._request("GET", "/api/v5/account/balance")
        if not response or len(response) == 0:
            return {}

        balances = {}
        for detail in response[0].get("details", []):
            ccy = detail.get("ccy")
            available = float(detail.get("availBal", 0))
            if ccy:
                balances[ccy] = available
        return balances

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage on OKX."""
        data = {
            "instId": symbol.replace("USDT", "-USDT-SWAP"),
            "lever": str(leverage),
            "mgnMode": "cross",
        }
        response = await self._request("POST", "/api/v5/account/set-leverage", data=data)
        return response is not None


class BybitClient(ExchangeClient):
    """Bybit exchange client implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ):
        super().__init__(api_key, api_secret, testnet)

        self.base_url = (
            "https://api-testnet.bybit.com"
            if testnet
            else "https://api.bybit.com"
        )
        self.ws_url = (
            "wss://stream-testnet.bybit.com/v5/private"
            if testnet
            else "wss://stream.bybit.com/v5/private"
        )

    def _sign(self, timestamp: str, params: Dict) -> str:
        """Create Bybit signature."""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sign_str = f"{timestamp}{self.api_key}{5000}{param_str}"
        signature = hmac.new(
            self.api_secret.encode(),
            sign_str.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _get_headers(self, timestamp: str, sign: str) -> Dict[str, str]:
        """Get Bybit request headers."""
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": sign,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        """Connect to Bybit."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
            self._connected = True
            logger.info(f"Connected to Bybit {'testnet' if self.testnet else 'mainnet'}")
        except ImportError:
            logger.error("aiohttp not installed")

    async def disconnect(self) -> None:
        """Disconnect from Bybit."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make Bybit API request."""
        timestamp = str(int(time.time() * 1000))
        all_params = params or data or {}
        sign = self._sign(timestamp, all_params)
        headers = self._get_headers(timestamp, sign)

        url = f"{self.base_url}{endpoint}"

        result = await self._request_with_retry(
            method, url, params=params, data=data, headers=headers
        )

        if result and result.get("retCode") == 0:
            return result.get("result", result)

        logger.error(f"Bybit API error: {result}")
        return None

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order on Bybit."""
        data = {
            "category": "linear",
            "symbol": request.symbol,
            "side": request.side.value.capitalize(),
            "orderType": "Market" if request.order_type == OrderType.MARKET else "Limit",
            "qty": str(request.quantity),
        }

        if request.price and request.order_type == OrderType.LIMIT:
            data["price"] = str(request.price)

        if request.reduce_only:
            data["reduceOnly"] = True

        if request.client_order_id:
            data["orderLinkId"] = request.client_order_id

        response = await self._request("POST", "/v5/order/create", data=data)

        if response:
            return OrderResult(
                order_id=response.get("orderId", ""),
                client_order_id=response.get("orderLinkId"),
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.OPEN,
                quantity=request.quantity,
                filled_quantity=0,
                price=request.price,
                avg_fill_price=None,
                fee=0.0,
                fee_currency="USDT",
                timestamp=datetime.utcnow(),
                raw_response=response,
            )

        return OrderResult(
            order_id="error",
            client_order_id=None,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.REJECTED,
            quantity=request.quantity,
            filled_quantity=0,
            price=None,
            avg_fill_price=None,
            fee=0.0,
            fee_currency="USDT",
            timestamp=datetime.utcnow(),
            raw_response={},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Bybit."""
        data = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        response = await self._request("POST", "/v5/order/cancel", data=data)
        return response is not None

    async def get_order(self, order_id: str, symbol: str) -> Optional[OrderResult]:
        """Get order from Bybit."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        response = await self._request("GET", "/v5/order/realtime", params=params)
        if response and response.get("list") and len(response["list"]) > 0:
            order_data = response["list"][0]
            status_map = {
                "New": OrderStatus.OPEN,
                "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
                "Filled": OrderStatus.FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "Rejected": OrderStatus.REJECTED,
            }
            return OrderResult(
                order_id=order_data.get("orderId", ""),
                client_order_id=order_data.get("orderLinkId"),
                symbol=symbol,
                side=OrderSide(order_data.get("side", "Buy").lower()),
                order_type=OrderType.MARKET if order_data.get("orderType") == "Market" else OrderType.LIMIT,
                status=status_map.get(order_data.get("orderStatus"), OrderStatus.PENDING),
                quantity=float(order_data.get("qty", 0)),
                filled_quantity=float(order_data.get("cumExecQty", 0)),
                price=float(order_data.get("price")) if order_data.get("price") else None,
                avg_fill_price=float(order_data.get("avgPrice")) if order_data.get("avgPrice") else None,
                fee=float(order_data.get("cumExecFee", 0)),
                fee_currency="USDT",
                timestamp=datetime.utcnow(),
                raw_response=order_data,
            )
        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Get open orders from Bybit."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol

        response = await self._request("GET", "/v5/order/realtime", params=params)
        if not response or not response.get("list"):
            return []

        orders = []
        for order_data in response["list"]:
            if order_data.get("orderStatus") in ["New", "PartiallyFilled"]:
                orders.append(OrderResult(
                    order_id=order_data.get("orderId", ""),
                    client_order_id=order_data.get("orderLinkId"),
                    symbol=order_data.get("symbol", ""),
                    side=OrderSide(order_data.get("side", "Buy").lower()),
                    order_type=OrderType.MARKET if order_data.get("orderType") == "Market" else OrderType.LIMIT,
                    status=OrderStatus.OPEN,
                    quantity=float(order_data.get("qty", 0)),
                    filled_quantity=float(order_data.get("cumExecQty", 0)),
                    price=float(order_data.get("price")) if order_data.get("price") else None,
                    avg_fill_price=None,
                    fee=0.0,
                    fee_currency="USDT",
                    timestamp=datetime.utcnow(),
                    raw_response=order_data,
                ))
        return orders

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position from Bybit."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def get_positions(self) -> List[Position]:
        """Get all positions from Bybit."""
        response = await self._request("GET", "/v5/position/list", params={"category": "linear"})
        if not response or not response.get("list"):
            return []

        positions = []
        for pos in response["list"]:
            size = float(pos.get("size", 0))
            if size != 0:
                positions.append(Position(
                    symbol=pos.get("symbol", ""),
                    side=pos.get("side", "").lower(),
                    size=size,
                    entry_price=float(pos.get("avgPrice", 0)),
                    mark_price=float(pos.get("markPrice", 0)),
                    unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                    leverage=int(float(pos.get("leverage", 1))),
                    liquidation_price=float(pos.get("liqPrice")) if pos.get("liqPrice") else None,
                    margin=float(pos.get("positionIM", 0)),
                ))
        return positions

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance from Bybit."""
        response = await self._request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
        if not response or not response.get("list"):
            return {}

        balances = {}
        for account in response["list"]:
            for coin in account.get("coin", []):
                ccy = coin.get("coin")
                available = float(coin.get("availableToWithdraw", 0))
                if ccy:
                    balances[ccy] = available
        return balances

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage on Bybit."""
        data = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        response = await self._request("POST", "/v5/position/set-leverage", data=data)
        return response is not None


def create_exchange_client(
    exchange: ExchangeType,
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    passphrase: str = "",  # For OKX
) -> ExchangeClient:
    """Factory function to create exchange client."""
    if exchange == ExchangeType.BINANCE:
        return BinanceClient(api_key, api_secret, testnet)
    elif exchange == ExchangeType.OKX:
        return OKXClient(api_key, api_secret, passphrase, testnet)
    elif exchange == ExchangeType.BYBIT:
        return BybitClient(api_key, api_secret, testnet)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")


# Convenience aliases
def binance_client(api_key: str, api_secret: str, testnet: bool = True) -> BinanceClient:
    return BinanceClient(api_key, api_secret, testnet)


def okx_client(api_key: str, api_secret: str, passphrase: str, testnet: bool = True) -> OKXClient:
    return OKXClient(api_key, api_secret, passphrase, testnet)


def bybit_client(api_key: str, api_secret: str, testnet: bool = True) -> BybitClient:
    return BybitClient(api_key, api_secret, testnet)
