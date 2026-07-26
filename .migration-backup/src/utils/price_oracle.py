"""
Price Oracle - Fetches token prices from multiple sources

Supports multiple price sources:
- CoinGecko (free, reliable)
- Coingecko API backup
- Binance API (for current market data)
"""

import logging
import asyncio
import aiohttp
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PriceOracle:
    """
    Fetches token prices from multiple sources with caching.

    Features:
    - Multi-source redundancy
    - Automatic caching
    - Error handling and fallbacks
    - Async support
    """

    # Token ID mappings for CoinGecko
    COINGECKO_IDS = {
        "ETH": "ethereum",
        "WETH": "weth",
        "BTC": "bitcoin",
        "WBTC": "wrapped-bitcoin",
        "USDC": "usd-coin",
        "USDT": "tether",
        "SOL": "solana",
        "MATIC": "matic-network",
        "BNB": "binancecoin",
        "AVAX": "avalanche-2",
    }

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize price oracle.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self._cache: Dict[str, tuple] = {}  # {token: (price, timestamp)}
        self._cache_ttl = cache_ttl
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_price(self, token: str) -> float:
        """
        Get token price in USD.

        Args:
            token: Token symbol (e.g., "ETH", "BTC", "SOL")

        Returns:
            Price in USD
        """
        token_upper = token.upper()

        # Check cache
        if token_upper in self._cache:
            price, timestamp = self._cache[token_upper]
            age = datetime.now() - timestamp
            if age.total_seconds() < self._cache_ttl:
                logger.debug(f"Cache hit for {token_upper}: ${price:.2f}")
                return price
            else:
                # Cache expired
                del self._cache[token_upper]

        # Try CoinGecko first
        try:
            price = await self._get_coingecko_price(token_upper)
            if price and price > 0:
                self._cache[token_upper] = (price, datetime.now())
                logger.info(f"CoinGecko price: {token_upper} = ${price:.2f}")
                return price
        except Exception as e:
            logger.warning(f"CoinGecko error for {token_upper}: {e}")

        # Fallback to Binance API
        try:
            price = await self._get_binance_price(token_upper)
            if price and price > 0:
                self._cache[token_upper] = (price, datetime.now())
                logger.info(f"Binance price: {token_upper} = ${price:.2f}")
                return price
        except Exception as e:
            logger.error(f"Binance API error for {token_upper}: {e}")

        # Last resort: return hardcoded fallback
        fallback_prices = {
            "ETH": 3000.0,
            "WETH": 3000.0,
            "BTC": 95000.0,
            "WBTC": 95000.0,
            "USDC": 1.0,
            "USDT": 1.0,
            "SOL": 120.0,
            "MATIC": 0.80,
            "BNB": 580.0,
            "AVAX": 35.0,
        }

        price = fallback_prices.get(token_upper, 1.0)
        logger.warning(f"Using fallback price for {token_upper}: ${price:.2f}")
        return price

    async def get_prices(self, tokens: list) -> Dict[str, float]:
        """
        Get multiple token prices.

        Args:
            tokens: List of token symbols

        Returns:
            Dictionary of token -> price
        """
        prices = {}
        for token in tokens:
            prices[token] = await self.get_price(token)
        return prices

    async def _get_coingecko_price(self, token: str) -> Optional[float]:
        """
        Get price from CoinGecko API.

        Args:
            token: Token symbol

        Returns:
            Price in USD or None
        """
        token_id = self.COINGECKO_IDS.get(token, token.lower())

        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": token_id,
            "vs_currencies": "usd",
            "include_market_cap": "false",
            "include_24hr_vol": "false",
            "include_24hr_change": "false",
            "include_last_updated_at": "false"
        }

        session = await self._get_session()
        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get(token_id, {}).get("usd")
                elif response.status == 429:
                    logger.warning("CoinGecko rate limited")
                    return None
                else:
                    logger.warning(f"CoinGecko error: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning("CoinGecko request timeout")
            return None

    async def _get_binance_price(self, token: str) -> Optional[float]:
        """
        Get price from Binance API (current market data).

        Args:
            token: Token symbol

        Returns:
            Price in USD or None
        """
        # Map tokens to Binance symbols
        binance_symbols = {
            "ETH": "ETHUSDT",
            "WETH": "ETHUSDT",
            "BTC": "BTCUSDT",
            "WBTC": "BTCUSDT",
            "USDC": "USDCUSDT",
            "USDT": "USDTUSDT",
            "SOL": "SOLUSDT",
            "MATIC": "MATICUSDT",
            "BNB": "BNBUSDT",
            "AVAX": "AVAXUSDT",
        }

        symbol = binance_symbols.get(token)
        if not symbol:
            return None

        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}

        session = await self._get_session()
        try:
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get("price", 0))
                else:
                    logger.warning(f"Binance API error: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning("Binance request timeout")
            return None

    async def get_token_price_from_address(
        self,
        chain: str,
        token_address: str
    ) -> Optional[float]:
        """
        Get token price from address using CoinGecko.

        Args:
            chain: Chain name (e.g., "ethereum", "base")
            token_address: Token contract address

        Returns:
            Price in USD or None
        """
        url = f"https://api.coingecko.com/api/v3/token_price/{token_address}"
        params = {
            "vs_currencies": "usd",
            "include_market_cap": "false",
            "include_24hr_vol": "false",
            "include_24hr_change": "false"
        }

        session = await self._get_session()
        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    token_data = data.get(token_address.lower(), {})
                    return token_data.get("usd")
                elif response.status == 404:
                    logger.warning(f"Token not found: {token_address}")
                    return None
                else:
                    logger.warning(f"CoinGecko token API error: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning("CoinGecko token API request timeout")
            return None

    def clear_cache(self) -> None:
        """Clear all cached prices."""
        self._cache.clear()
        logger.info("Price cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached prices."""
        return len(self._cache)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance for reuse
_oracle_instance: Optional[PriceOracle] = None


def get_price_oracle() -> PriceOracle:
    """Get or create singleton price oracle instance."""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = PriceOracle()
    return _oracle_instance


async def get_token_price(token: str) -> float:
    """
    Convenience function to get token price.

    Args:
        token: Token symbol

    Returns:
        Price in USD
    """
    oracle = get_price_oracle()
    return await oracle.get_price(token)


# Example usage
if __name__ == "__main__":
    import sys

    async def main():
        oracle = PriceOracle()

        print("\n" + "="*70)
        print("PRICE ORACLE TEST")
        print("="*70)

        # Test single price
        print("\n--- Test 1: Single Token Price ---")
        eth_price = await oracle.get_price("ETH")
        print(f"ETH: ${eth_price:.2f}")

        sol_price = await oracle.get_price("SOL")
        print(f"SOL: ${sol_price:.2f}")

        # Test multiple prices
        print("\n--- Test 2: Multiple Token Prices ---")
        tokens = ["BTC", "ETH", "SOL", "MATIC", "USDC"]
        prices = await oracle.get_prices(tokens)

        for token, price in prices.items():
            print(f"{token}: ${price:.2f}")

        # Test cache
        print("\n--- Test 3: Cache Test ---")
        eth_price_2 = await oracle.get_price("ETH")
        print(f"ETH (cached): ${eth_price_2:.2f}")
        print(f"Cache size: {oracle.get_cache_size()}")

        await oracle.close()

    asyncio.run(main())
