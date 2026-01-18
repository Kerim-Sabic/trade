"""
Tests for real-time price feed integration.

Tests both synchronous and asynchronous price fetching.
Windows 11 compatible.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from cryptoai.data_universe.price_feed import (
    PriceFeedProvider,
    PriceFeedConfig,
    PriceData,
    CoinGeckoPriceFeed,
    BinancePublicPriceFeed,
    RealTimePriceFeed,
    SyncPriceFeed,
    create_price_feed,
)


class TestPriceData:
    """Tests for PriceData dataclass."""

    def test_price_data_creation(self):
        """Test creating a PriceData object."""
        price = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
            bid=49990.0,
            ask=50010.0,
            volume_24h=1000000000.0,
            change_24h=2.5,
            source="test",
        )

        assert price.symbol == "BTCUSDT"
        assert price.price == 50000.0
        assert price.bid == 49990.0
        assert price.ask == 50010.0

    def test_price_data_to_dict(self):
        """Test converting PriceData to dictionary."""
        price = PriceData(
            symbol="ETHUSDT",
            price=3000.0,
            timestamp=datetime.utcnow(),
            source="binance",
        )

        data = price.to_dict()

        assert data["symbol"] == "ETHUSDT"
        assert data["price"] == 3000.0
        assert data["source"] == "binance"
        assert "timestamp" in data


class TestPriceFeedConfig:
    """Tests for PriceFeedConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PriceFeedConfig()

        assert config.provider == PriceFeedProvider.COINGECKO
        assert config.update_interval_seconds == 5.0
        assert "BTCUSDT" in config.symbols
        assert "ETHUSDT" in config.symbols

    def test_custom_config(self):
        """Test custom configuration."""
        config = PriceFeedConfig(
            provider=PriceFeedProvider.BINANCE_PUBLIC,
            symbols=["BTCUSDT", "SOLUSDT"],
            update_interval_seconds=10.0,
        )

        assert config.provider == PriceFeedProvider.BINANCE_PUBLIC
        assert len(config.symbols) == 2
        assert config.update_interval_seconds == 10.0


class TestCoinGeckoPriceFeed:
    """Tests for CoinGecko price feed."""

    def test_symbol_conversion(self):
        """Test symbol conversion to CoinGecko IDs."""
        config = PriceFeedConfig()
        feed = CoinGeckoPriceFeed(config)

        assert feed._convert_symbol("BTCUSDT") == "bitcoin"
        assert feed._convert_symbol("ETHUSDT") == "ethereum"
        assert feed._convert_symbol("SOLUSDT") == "solana"

    @pytest.mark.asyncio
    async def test_get_price_mock(self):
        """Test getting price with mocked response."""
        config = PriceFeedConfig()
        feed = CoinGeckoPriceFeed(config)

        mock_response = {
            "bitcoin": {
                "usd": 50000.0,
                "usd_24h_vol": 25000000000.0,
                "usd_24h_change": 3.5,
            }
        }

        with patch.object(feed, '_request_with_retry', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await feed.connect()

            price = await feed.get_price("BTCUSDT")

            assert price is not None
            assert price.symbol == "BTCUSDT"
            assert price.price == 50000.0
            assert price.change_24h == 3.5
            assert price.source == "coingecko"


class TestBinancePublicPriceFeed:
    """Tests for Binance public price feed."""

    @pytest.mark.asyncio
    async def test_get_price_mock(self):
        """Test getting price with mocked response."""
        config = PriceFeedConfig(provider=PriceFeedProvider.BINANCE_PUBLIC)
        feed = BinancePublicPriceFeed(config)

        mock_response = {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.00",
            "bidPrice": "49990.00",
            "askPrice": "50010.00",
            "priceChangePercent": "2.50",
            "quoteVolume": "1000000000.00",
            "highPrice": "51000.00",
            "lowPrice": "49000.00",
        }

        with patch.object(feed, '_request_with_retry', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await feed.connect()

            price = await feed.get_price("BTCUSDT")

            assert price is not None
            assert price.symbol == "BTCUSDT"
            assert price.price == 50000.0
            assert price.bid == 49990.0
            assert price.ask == 50010.0
            assert price.change_24h == 2.5
            assert price.source == "binance"

    @pytest.mark.asyncio
    async def test_get_prices_bulk_mock(self):
        """Test getting multiple prices."""
        config = PriceFeedConfig(
            provider=PriceFeedProvider.BINANCE_PUBLIC,
            symbols=["BTCUSDT", "ETHUSDT"],
        )
        feed = BinancePublicPriceFeed(config)

        mock_response = [
            {
                "symbol": "BTCUSDT",
                "lastPrice": "50000.00",
                "bidPrice": "49990.00",
                "askPrice": "50010.00",
                "priceChangePercent": "2.50",
                "quoteVolume": "1000000000.00",
                "highPrice": "51000.00",
                "lowPrice": "49000.00",
            },
            {
                "symbol": "ETHUSDT",
                "lastPrice": "3000.00",
                "bidPrice": "2995.00",
                "askPrice": "3005.00",
                "priceChangePercent": "1.50",
                "quoteVolume": "500000000.00",
                "highPrice": "3100.00",
                "lowPrice": "2900.00",
            },
        ]

        with patch.object(feed, '_request_with_retry', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await feed.connect()

            prices = await feed.get_prices(["BTCUSDT", "ETHUSDT"])

            assert len(prices) == 2
            assert "BTCUSDT" in prices
            assert "ETHUSDT" in prices
            assert prices["BTCUSDT"].price == 50000.0
            assert prices["ETHUSDT"].price == 3000.0

    @pytest.mark.asyncio
    async def test_get_klines_mock(self):
        """Test getting historical OHLCV data."""
        config = PriceFeedConfig(provider=PriceFeedProvider.BINANCE_PUBLIC)
        feed = BinancePublicPriceFeed(config)

        mock_klines = [
            [1704067200000, "50000", "51000", "49500", "50500", "1000", 1704070800000, "50000000", 1000, "500", "25000000", "0"],
            [1704070800000, "50500", "51500", "50000", "51000", "1200", 1704074400000, "60000000", 1200, "600", "30000000", "0"],
        ]

        with patch.object(feed, '_request_with_retry', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_klines
            await feed.connect()

            klines = await feed.get_klines("BTCUSDT", interval="1h", limit=2)

            assert len(klines) == 2
            assert klines[0]["open"] == 50000.0
            assert klines[0]["close"] == 50500.0
            assert klines[1]["open"] == 50500.0


class TestRealTimePriceFeed:
    """Tests for unified real-time price feed."""

    def test_create_feed_binance(self):
        """Test creating Binance feed."""
        config = PriceFeedConfig(provider=PriceFeedProvider.BINANCE_PUBLIC)
        feed = RealTimePriceFeed(config)

        assert feed.config.provider == PriceFeedProvider.BINANCE_PUBLIC

    def test_create_feed_coingecko(self):
        """Test creating CoinGecko feed."""
        config = PriceFeedConfig(provider=PriceFeedProvider.COINGECKO)
        feed = RealTimePriceFeed(config)

        assert feed.config.provider == PriceFeedProvider.COINGECKO

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        config = PriceFeedConfig(provider=PriceFeedProvider.BINANCE_PUBLIC)
        feed = RealTimePriceFeed(config)

        await feed.connect()
        assert feed._feed is not None

        await feed.disconnect()

    def test_callback_registration(self):
        """Test registering price update callbacks."""
        feed = RealTimePriceFeed()

        callback_called = []

        def on_update(prices):
            callback_called.append(prices)

        feed.on_price_update(on_update)

        assert len(feed._callbacks) == 1


class TestSyncPriceFeed:
    """Tests for synchronous price feed wrapper."""

    def test_sync_feed_creation(self):
        """Test creating sync feed."""
        feed = SyncPriceFeed(provider="binance")

        assert feed.config.provider == PriceFeedProvider.BINANCE_PUBLIC

    def test_sync_feed_with_symbols(self):
        """Test sync feed with custom symbols."""
        feed = SyncPriceFeed(
            provider="binance",
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        )

        assert len(feed.config.symbols) == 3


class TestFactoryFunction:
    """Tests for create_price_feed factory function."""

    def test_create_binance_feed(self):
        """Test creating Binance feed via factory."""
        feed = create_price_feed(
            provider="binance",
            symbols=["BTCUSDT"],
            update_interval=10.0,
        )

        assert feed.config.provider == PriceFeedProvider.BINANCE_PUBLIC
        assert feed.config.update_interval_seconds == 10.0

    def test_create_coingecko_feed(self):
        """Test creating CoinGecko feed via factory."""
        feed = create_price_feed(
            provider="coingecko",
            symbols=["BTCUSDT", "ETHUSDT"],
        )

        assert feed.config.provider == PriceFeedProvider.COINGECKO


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_wait(self):
        """Test that rate limiting delays requests."""
        config = PriceFeedConfig(provider=PriceFeedProvider.BINANCE_PUBLIC)
        feed = BinancePublicPriceFeed(config)

        start = asyncio.get_event_loop().time()

        await feed._rate_limit()
        await feed._rate_limit()

        elapsed = asyncio.get_event_loop().time() - start

        # Should have some delay between requests
        assert elapsed >= feed._min_request_interval * 0.9  # Allow some tolerance


# Integration test (optional - requires network)
class TestLiveIntegration:
    """
    Live integration tests.

    These tests hit real APIs and should be run sparingly.
    Mark with @pytest.mark.integration and skip by default.
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_binance_price(self):
        """Test fetching live price from Binance."""
        config = PriceFeedConfig(
            provider=PriceFeedProvider.BINANCE_PUBLIC,
            symbols=["BTCUSDT"],
            timeout_seconds=10.0,
        )
        feed = BinancePublicPriceFeed(config)

        await feed.connect()
        try:
            price = await feed.get_price("BTCUSDT")

            # BTC should always be worth something!
            assert price is not None
            assert price.price > 1000  # BTC is definitely > $1000
            assert price.source == "binance"
        finally:
            await feed.disconnect()
