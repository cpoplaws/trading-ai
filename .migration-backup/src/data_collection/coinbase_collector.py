"""
Coinbase Advanced Trade API Data Collector
Collects historical and real-time market data from Coinbase.
"""
import os
import time
import hmac
import hashlib
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CoinbaseCandle:
    """OHLCV candle from Coinbase."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class CoinbaseCollector:
    """
    Coinbase Advanced Trade API client.

    Features:
    - Historical OHLCV data
    - Real-time ticker prices
    - Order book snapshots
    - Recent trades
    - Product information

    API Documentation:
    https://docs.cloud.coinbase.com/advanced-trade-api/docs
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize Coinbase collector.

        Args:
            api_key: Coinbase API key (or set COINBASE_API_KEY env var)
            api_secret: Coinbase API secret (or set COINBASE_API_SECRET env var)
        """
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')

        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

        logger.info("Coinbase collector initialized")

    def _sign_request(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Sign API request for authentication.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            body: Request body (for POST requests)

        Returns:
            Headers with signature
        """
        timestamp = str(int(time.time()))
        message = timestamp + method + path + body

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp
        }

    def get_products(self) -> List[Dict]:
        """
        Get all available trading pairs.

        Returns:
            List of products with trading info
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/products",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            products = data.get('products', [])

            logger.info(f"Retrieved {len(products)} Coinbase products")
            return products

        except Exception as e:
            logger.error(f"Failed to get products: {e}")
            return []

    def get_product(self, symbol: str) -> Optional[Dict]:
        """
        Get product information.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Product information
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/products/{symbol}",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            return data

        except Exception as e:
            logger.error(f"Failed to get product {symbol}: {e}")
            return None

    def get_candles(
        self,
        symbol: str,
        granularity: str = '3600',  # 1 hour
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 300
    ) -> List[CoinbaseCandle]:
        """
        Get historical candlestick data.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            granularity: Candle size in seconds
                        '60' = 1 minute
                        '300' = 5 minutes
                        '900' = 15 minutes
                        '3600' = 1 hour
                        '21600' = 6 hours
                        '86400' = 1 day
            start: Start time
            end: End time
            limit: Max candles to return (max 300)

        Returns:
            List of OHLCV candles
        """
        try:
            params = {
                'granularity': granularity
            }

            if start:
                params['start'] = int(start.timestamp())
            if end:
                params['end'] = int(end.timestamp())

            response = self.session.get(
                f"{self.BASE_URL}/products/{symbol}/candles",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            candles_data = data.get('candles', [])

            # Parse candles
            candles = []
            for candle in candles_data:
                candles.append(CoinbaseCandle(
                    timestamp=datetime.fromtimestamp(int(candle['start'])),
                    symbol=symbol,
                    open=float(candle['open']),
                    high=float(candle['high']),
                    low=float(candle['low']),
                    close=float(candle['close']),
                    volume=float(candle['volume'])
                ))

            logger.info(f"Retrieved {len(candles)} candles for {symbol}")
            return candles

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return []

    def get_candles_range(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime
    ) -> List[CoinbaseCandle]:
        """
        Get candles for extended time range (handles API limits).

        Coinbase API limits to 300 candles per request.
        This method automatically chunks large date ranges.

        Args:
            symbol: Trading pair
            granularity: Candle size in seconds
            start: Start time
            end: End time

        Returns:
            All candles in date range
        """
        all_candles = []
        granularity_seconds = int(granularity)

        # Calculate chunk size (300 candles max per request)
        chunk_duration = timedelta(seconds=granularity_seconds * 300)

        current_start = start
        while current_start < end:
            current_end = min(current_start + chunk_duration, end)

            candles = self.get_candles(
                symbol=symbol,
                granularity=granularity,
                start=current_start,
                end=current_end
            )

            all_candles.extend(candles)
            current_start = current_end

            # Rate limiting - be nice to API
            time.sleep(0.5)

        logger.info(f"Retrieved {len(all_candles)} total candles for {symbol}")
        return all_candles

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get current ticker price.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Ticker data with price, volume, etc.
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/products/{symbol}/ticker",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            return data

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    def get_order_book(
        self,
        symbol: str,
        level: int = 2
    ) -> Optional[Dict]:
        """
        Get order book snapshot.

        Args:
            symbol: Trading pair
            level: Order book depth
                   1 = Best bid/ask only
                   2 = Top 50 bids/asks
                   3 = Full order book

        Returns:
            Order book with bids and asks
        """
        try:
            params = {'level': level}

            response = self.session.get(
                f"{self.BASE_URL}/products/{symbol}/book",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            return data

        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None

    def get_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)

        Returns:
            List of recent trades
        """
        try:
            params = {'limit': min(limit, 1000)}

            response = self.session.get(
                f"{self.BASE_URL}/products/{symbol}/trades",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            trades = data.get('trades', [])

            logger.info(f"Retrieved {len(trades)} trades for {symbol}")
            return trades

        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            return []

    def candles_to_dataframe(
        self,
        candles: List[CoinbaseCandle]
    ) -> pd.DataFrame:
        """
        Convert candles to pandas DataFrame.

        Args:
            candles: List of candles

        Returns:
            DataFrame with OHLCV data
        """
        if not candles:
            return pd.DataFrame()

        data = {
            'timestamp': [c.timestamp for c in candles],
            'symbol': [c.symbol for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles]
        }

        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)

        return df

    def save_candles_to_db(
        self,
        candles: List[CoinbaseCandle],
        db_manager
    ) -> int:
        """
        Save candles to database.

        Args:
            candles: List of candles
            db_manager: DatabaseManager instance

        Returns:
            Number of candles saved
        """
        if not candles:
            return 0

        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'exchange': 'coinbase',
                'symbol': candle.symbol,
                'interval': '1h',  # You may want to make this dynamic
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })

        count = db_manager.insert_ohlcv(data)
        logger.info(f"Saved {count} candles to database")
        return count


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("📊 Coinbase Collector Demo")
    print("=" * 60)

    # Initialize collector (no API key needed for public endpoints)
    collector = CoinbaseCollector()

    # Get all products
    print("\n1. Getting available products...")
    products = collector.get_products()
    print(f"✓ Found {len(products)} trading pairs")

    # Show top 5 by volume
    if products:
        print("\nTop 5 products by volume:")
        for i, product in enumerate(products[:5], 1):
            print(f"  {i}. {product.get('product_id')} - "
                  f"Volume: ${float(product.get('volume_24h', 0)):,.0f}")

    # Get BTC-USD ticker
    print("\n2. Getting BTC-USD ticker...")
    ticker = collector.get_ticker('BTC-USD')
    if ticker:
        print(f"✓ BTC-USD Price: ${float(ticker.get('price', 0)):,.2f}")

    # Get recent candles
    print("\n3. Getting recent BTC-USD candles (1 hour)...")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    candles = collector.get_candles(
        symbol='BTC-USD',
        granularity='3600',  # 1 hour
        start=start_time,
        end=end_time
    )

    if candles:
        print(f"✓ Retrieved {len(candles)} candles")
        print("\nLatest 3 candles:")
        for candle in candles[:3]:
            print(f"  {candle.timestamp}: "
                  f"O={candle.open:,.2f} H={candle.high:,.2f} "
                  f"L={candle.low:,.2f} C={candle.close:,.2f} "
                  f"V={candle.volume:.4f}")

        # Convert to DataFrame
        df = collector.candles_to_dataframe(candles)
        print(f"\n✓ Converted to DataFrame: {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Get order book
    print("\n4. Getting BTC-USD order book...")
    order_book = collector.get_order_book('BTC-USD', level=2)
    if order_book:
        bids = order_book.get('pricebook', {}).get('bids', [])
        asks = order_book.get('pricebook', {}).get('asks', [])
        print(f"✓ Order book: {len(bids)} bids, {len(asks)} asks")

        if bids and asks:
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            print(f"  Best bid: ${best_bid:,.2f}")
            print(f"  Best ask: ${best_ask:,.2f}")
            print(f"  Spread: ${spread:.2f} ({spread_pct:.3f}%)")

    # Get recent trades
    print("\n5. Getting recent BTC-USD trades...")
    trades = collector.get_trades('BTC-USD', limit=5)
    if trades:
        print(f"✓ Retrieved {len(trades)} trades")
        print("\nLatest 3 trades:")
        for trade in trades[:3]:
            print(f"  {trade.get('time')}: "
                  f"${float(trade.get('price', 0)):,.2f} x "
                  f"{float(trade.get('size', 0)):.4f} BTC "
                  f"[{trade.get('side')}]")

    print("\n✅ Coinbase collector demo complete!")
    print("\nNext steps:")
    print("1. Set COINBASE_API_KEY and COINBASE_API_SECRET in .env")
    print("2. Use collector.save_candles_to_db() to store data")
    print("3. Run scripts/collect_coinbase_data.py for automated collection")
