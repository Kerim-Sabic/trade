"""
Real-Time Price Feed API Integration.

Provides real price data for simulation mode.
Supports multiple providers: CoinGecko (free), Polygon.io, TwelveData.

Windows 11 Compatible.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import os

import aiohttp
import httpx
from loguru import logger


class PriceFeedProvider(Enum):
    """Supported price feed providers."""
    COINGECKO = "coingecko"  # Free, 10-30 calls/minute
    POLYGON = "polygon"      # Requires API key, generous free tier
    TWELVEDATA = "twelvedata"  # Requires API key
    BINANCE_PUBLIC = "binance"  # Free public API (no key needed)


@dataclass
class PriceData:
    """Real-time price data."""
    symbol: str
    price: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    source: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "source": self.source,
        }


@dataclass
class PriceFeedConfig:
    """Price feed configuration."""
    provider: PriceFeedProvider = PriceFeedProvider.COINGECKO
    api_key: Optional[str] = None
    update_interval_seconds: float = 5.0
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class PriceFeedBase(ABC):
    """Abstract base class for price feeds."""

    def __init__(self, config: PriceFeedConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # Subclasses override

    @abstractmethod
    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    async def get_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Get current prices for multiple symbols."""
        pass

    async def connect(self):
        """Connect to API."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )

    async def disconnect(self):
        """Disconnect from API."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _rate_limit(self):
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    async def _request_with_retry(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        await self._rate_limit()

        if self._session is None:
            await self.connect()

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:  # Rate limited
                        wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"API error {resp.status}: {await resp.text()}")
                        return None
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))

        return None


class CoinGeckoPriceFeed(PriceFeedBase):
    """
    CoinGecko free API price feed.

    Limits: 10-30 calls/minute (free tier).
    No API key required for basic usage.
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Symbol mapping: BTCUSDT -> bitcoin
    SYMBOL_MAP = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
        "BNBUSDT": "binancecoin",
        "XRPUSDT": "ripple",
        "ADAUSDT": "cardano",
        "AVAXUSDT": "avalanche-2",
        "DOGEUSDT": "dogecoin",
        "DOTUSDT": "polkadot",
        "LINKUSDT": "chainlink",
        "MATICUSDT": "matic-network",
        "ATOMUSDT": "cosmos",
        "UNIUSDT": "uniswap",
        "AAVEUSDT": "aave",
        "LTCUSDT": "litecoin",
    }

    def __init__(self, config: PriceFeedConfig):
        super().__init__(config)
        self._min_request_interval = 3.0  # Conservative rate limiting

    def _convert_symbol(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko ID."""
        return self.SYMBOL_MAP.get(symbol.upper(), symbol.lower())

    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        coin_id = self._convert_symbol(symbol)
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        data = await self._request_with_retry(url, params)
        if not data or coin_id not in data:
            return None

        coin_data = data[coin_id]
        return PriceData(
            symbol=symbol,
            price=coin_data.get("usd", 0),
            timestamp=datetime.utcnow(),
            volume_24h=coin_data.get("usd_24h_vol"),
            change_24h=coin_data.get("usd_24h_change"),
            source="coingecko",
        )

    async def get_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Get current prices for multiple symbols."""
        coin_ids = [self._convert_symbol(s) for s in symbols]
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        data = await self._request_with_retry(url, params)
        if not data:
            return {}

        result = {}
        for symbol in symbols:
            coin_id = self._convert_symbol(symbol)
            if coin_id in data:
                coin_data = data[coin_id]
                result[symbol] = PriceData(
                    symbol=symbol,
                    price=coin_data.get("usd", 0),
                    timestamp=datetime.utcnow(),
                    volume_24h=coin_data.get("usd_24h_vol"),
                    change_24h=coin_data.get("usd_24h_change"),
                    source="coingecko",
                )

        return result


class BinancePublicPriceFeed(PriceFeedBase):
    """
    Binance public API price feed.

    No API key required for public market data.
    Higher rate limits than CoinGecko.
    """

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, config: PriceFeedConfig):
        super().__init__(config)
        self._min_request_interval = 0.1  # 1200 req/min limit

    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        url = f"{self.BASE_URL}/ticker/24hr"
        params = {"symbol": symbol.upper()}

        data = await self._request_with_retry(url, params)
        if not data:
            return None

        return PriceData(
            symbol=symbol,
            price=float(data.get("lastPrice", 0)),
            timestamp=datetime.utcnow(),
            bid=float(data.get("bidPrice", 0)),
            ask=float(data.get("askPrice", 0)),
            volume_24h=float(data.get("quoteVolume", 0)),
            change_24h=float(data.get("priceChangePercent", 0)),
            high_24h=float(data.get("highPrice", 0)),
            low_24h=float(data.get("lowPrice", 0)),
            source="binance",
        )

    async def get_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Get current prices for multiple symbols."""
        # Binance has a bulk endpoint
        url = f"{self.BASE_URL}/ticker/24hr"

        data = await self._request_with_retry(url)
        if not data:
            return {}

        result = {}
        symbol_set = {s.upper() for s in symbols}

        for ticker in data:
            symbol = ticker.get("symbol")
            if symbol in symbol_set:
                result[symbol] = PriceData(
                    symbol=symbol,
                    price=float(ticker.get("lastPrice", 0)),
                    timestamp=datetime.utcnow(),
                    bid=float(ticker.get("bidPrice", 0)),
                    ask=float(ticker.get("askPrice", 0)),
                    volume_24h=float(ticker.get("quoteVolume", 0)),
                    change_24h=float(ticker.get("priceChangePercent", 0)),
                    high_24h=float(ticker.get("highPrice", 0)),
                    low_24h=float(ticker.get("lowPrice", 0)),
                    source="binance",
                )

        return result

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
    ) -> List[Dict]:
        """Get historical OHLCV data."""
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }

        data = await self._request_with_retry(url, params)
        if not data:
            return []

        result = []
        for kline in data:
            result.append({
                "timestamp": datetime.fromtimestamp(kline[0] / 1000),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            })

        return result


class PolygonPriceFeed(PriceFeedBase):
    """
    Polygon.io API price feed.

    Requires API key (free tier available).
    Better for stocks/forex, also has crypto.
    """

    BASE_URL = "https://api.polygon.io/v2"

    def __init__(self, config: PriceFeedConfig):
        super().__init__(config)
        self._min_request_interval = 0.2
        if not config.api_key:
            logger.warning("Polygon API key not provided, may have limited access")

    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        # Polygon uses X:BTCUSD format for crypto
        polygon_symbol = f"X:{symbol.replace('USDT', 'USD')}"
        url = f"{self.BASE_URL}/snapshot/locale/global/markets/crypto/tickers/{polygon_symbol}"
        params = {"apiKey": self.config.api_key} if self.config.api_key else {}

        data = await self._request_with_retry(url, params)
        if not data or "ticker" not in data:
            return None

        ticker = data["ticker"]
        return PriceData(
            symbol=symbol,
            price=ticker.get("lastTrade", {}).get("p", 0),
            timestamp=datetime.utcnow(),
            volume_24h=ticker.get("day", {}).get("v"),
            change_24h=ticker.get("todaysChangePerc"),
            source="polygon",
        )

    async def get_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Get current prices for multiple symbols."""
        result = {}
        for symbol in symbols:
            price = await self.get_price(symbol)
            if price:
                result[symbol] = price
        return result


class RealTimePriceFeed:
    """
    Unified real-time price feed manager.

    Supports multiple providers with automatic fallback.
    """

    def __init__(self, config: Optional[PriceFeedConfig] = None):
        self.config = config or PriceFeedConfig()
        self._feed: Optional[PriceFeedBase] = None
        self._callbacks: List[Callable[[Dict[str, PriceData]], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._latest_prices: Dict[str, PriceData] = {}

    def _create_feed(self) -> PriceFeedBase:
        """Create appropriate feed based on config."""
        provider = self.config.provider

        if provider == PriceFeedProvider.COINGECKO:
            return CoinGeckoPriceFeed(self.config)
        elif provider == PriceFeedProvider.BINANCE_PUBLIC:
            return BinancePublicPriceFeed(self.config)
        elif provider == PriceFeedProvider.POLYGON:
            return PolygonPriceFeed(self.config)
        else:
            # Default to Binance public (no API key needed)
            logger.info("Defaulting to Binance public price feed")
            return BinancePublicPriceFeed(self.config)

    async def connect(self):
        """Connect to price feed."""
        self._feed = self._create_feed()
        await self._feed.connect()
        logger.info(f"Connected to {self.config.provider.value} price feed")

    async def disconnect(self):
        """Disconnect from price feed."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._feed:
            await self._feed.disconnect()
        logger.info("Disconnected from price feed")

    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        if not self._feed:
            await self.connect()
        return await self._feed.get_price(symbol)

    async def get_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, PriceData]:
        """Get current prices for multiple symbols."""
        symbols = symbols or self.config.symbols
        if not self._feed:
            await self.connect()
        prices = await self._feed.get_prices(symbols)
        self._latest_prices.update(prices)
        return prices

    def get_latest_prices(self) -> Dict[str, PriceData]:
        """Get cached latest prices (synchronous)."""
        return self._latest_prices.copy()

    def on_price_update(self, callback: Callable[[Dict[str, PriceData]], None]):
        """Register callback for price updates."""
        self._callbacks.append(callback)

    async def start_streaming(self):
        """Start streaming price updates."""
        self._running = True
        await self.connect()

        logger.info(
            f"Starting price stream for {self.config.symbols} "
            f"(interval: {self.config.update_interval_seconds}s)"
        )

        while self._running:
            try:
                prices = await self.get_prices()

                for callback in self._callbacks:
                    try:
                        callback(prices)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price streaming error: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)

    def start_streaming_background(self) -> asyncio.Task:
        """Start streaming in background task."""
        self._task = asyncio.create_task(self.start_streaming())
        return self._task

    def stop_streaming(self):
        """Stop streaming."""
        self._running = False


# Factory function
def create_price_feed(
    provider: str = "binance",
    api_key: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    update_interval: float = 5.0,
) -> RealTimePriceFeed:
    """
    Create a real-time price feed.

    Args:
        provider: Price feed provider (binance, coingecko, polygon)
        api_key: API key (if required by provider)
        symbols: List of symbols to track
        update_interval: Update interval in seconds

    Returns:
        Configured RealTimePriceFeed instance

    Example:
        feed = create_price_feed(provider="binance", symbols=["BTCUSDT", "ETHUSDT"])
        await feed.connect()
        price = await feed.get_price("BTCUSDT")
        print(f"BTC: ${price.price:,.2f}")
    """
    provider_enum = PriceFeedProvider(provider.lower())

    config = PriceFeedConfig(
        provider=provider_enum,
        api_key=api_key or os.environ.get("PRICE_FEED_API_KEY"),
        symbols=symbols or ["BTCUSDT", "ETHUSDT"],
        update_interval_seconds=update_interval,
    )

    return RealTimePriceFeed(config)


# Synchronous wrapper for Electron/CLI usage
class SyncPriceFeed:
    """Synchronous wrapper for price feed (for non-async contexts)."""

    def __init__(self, provider: str = "binance", symbols: Optional[List[str]] = None):
        self.config = PriceFeedConfig(
            provider=PriceFeedProvider(provider.lower()),
            symbols=symbols or ["BTCUSDT", "ETHUSDT"],
        )
        self._feed: Optional[PriceFeedBase] = None

    def _get_feed(self) -> PriceFeedBase:
        if self._feed is None:
            if self.config.provider == PriceFeedProvider.BINANCE_PUBLIC:
                self._feed = BinancePublicPriceFeed(self.config)
            else:
                self._feed = CoinGeckoPriceFeed(self.config)
        return self._feed

    def get_price_sync(self, symbol: str) -> Optional[PriceData]:
        """Get price synchronously using httpx."""
        feed = self._get_feed()

        if isinstance(feed, BinancePublicPriceFeed):
            url = f"{feed.BASE_URL}/ticker/24hr"
            params = {"symbol": symbol.upper()}
        else:
            coin_id = CoinGeckoPriceFeed.SYMBOL_MAP.get(symbol.upper(), symbol.lower())
            url = f"{CoinGeckoPriceFeed.BASE_URL}/simple/price"
            params = {"ids": coin_id, "vs_currencies": "usd"}

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()

                    if isinstance(feed, BinancePublicPriceFeed):
                        return PriceData(
                            symbol=symbol,
                            price=float(data.get("lastPrice", 0)),
                            timestamp=datetime.utcnow(),
                            source="binance",
                        )
                    else:
                        coin_id = CoinGeckoPriceFeed.SYMBOL_MAP.get(symbol.upper(), symbol.lower())
                        if coin_id in data:
                            return PriceData(
                                symbol=symbol,
                                price=data[coin_id].get("usd", 0),
                                timestamp=datetime.utcnow(),
                                source="coingecko",
                            )
        except Exception as e:
            logger.error(f"Sync price fetch error: {e}")

        return None

    def get_prices_sync(self, symbols: Optional[List[str]] = None) -> Dict[str, PriceData]:
        """Get multiple prices synchronously."""
        symbols = symbols or self.config.symbols
        result = {}
        for symbol in symbols:
            price = self.get_price_sync(symbol)
            if price:
                result[symbol] = price
        return result
