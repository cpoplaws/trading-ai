"""
Historical crypto data fetcher for backtesting blockchain assets.

Fetches and processes historical price data for crypto assets across multiple
sources and chains for use in backtesting and analysis.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class HistoricalCryptoDataFetcher:
    """
    Fetch historical crypto data for backtesting.
    
    Supports multiple data sources:
    - Binance API for spot and futures data
    - CoinGecko for historical prices
    - On-chain data simulation
    """
    
    def __init__(self, binance_client=None, coingecko_client=None):
        """
        Initialize data fetcher.
        
        Args:
            binance_client: Optional Binance client instance
            coingecko_client: Optional CoinGecko client instance
        """
        self.binance_client = binance_client
        self.coingecko_client = coingecko_client
        
        logger.info("Initialized HistoricalCryptoDataFetcher")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1h',
        source: str = 'binance',
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a crypto asset.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'bitcoin')
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)
            source: Data source ('binance', 'coingecko', 'simulated')
            
        Returns:
            DataFrame with OHLCV data
        """
        if source == 'binance' and self.binance_client:
            return self._fetch_from_binance(symbol, start_date, end_date, interval)
        elif source == 'coingecko' and self.coingecko_client:
            return self._fetch_from_coingecko(symbol, start_date, end_date)
        elif source == 'simulated':
            return self._simulate_historical_data(symbol, start_date, end_date, interval)
        else:
            logger.warning(f"Source {source} not available, using simulated data")
            return self._simulate_historical_data(symbol, start_date, end_date, interval)
    
    def _fetch_from_binance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Binance API."""
        try:
            # Calculate number of candles needed
            days = (end_date - start_date).days
            
            # Fetch klines in chunks (Binance has 1000 limit per request)
            all_klines = []
            current_date = start_date
            
            while current_date < end_date:
                klines = self.binance_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Get last timestamp and move forward
                last_timestamp = klines[-1]['timestamp']
                current_date = datetime.fromtimestamp(last_timestamp / 1000)
            
            if not all_klines:
                logger.warning(f"No data retrieved from Binance for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter to date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Rename columns for consistency
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            })
            
            logger.info(f"Fetched {len(df)} candles from Binance for {symbol}")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error fetching from Binance: {e}")
            return None
    
    def _fetch_from_coingecko(
        self,
        coin_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from CoinGecko API."""
        try:
            # Calculate days
            days = (end_date - start_date).days
            
            # Fetch historical data
            data = self.coingecko_client.get_market_chart(
                coin_id=coin_id,
                vs_currency='usd',
                days=days
            )
            
            if not data or 'prices' not in data:
                logger.warning(f"No data retrieved from CoinGecko for {coin_id}")
                return None
            
            # Convert to DataFrame
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            prices_df.set_index('timestamp', inplace=True)
            
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
            volumes_df.set_index('timestamp', inplace=True)
            
            # Merge price and volume
            df = prices_df.join(volumes_df)
            
            # CoinGecko doesn't provide OHLC, so we approximate
            df['Open'] = df['Close']
            df['High'] = df['Close'] * 1.01  # Approximate
            df['Low'] = df['Close'] * 0.99  # Approximate
            
            logger.info(f"Fetched {len(df)} data points from CoinGecko for {coin_id}")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            return None
    
    def _simulate_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """
        Simulate realistic historical crypto data.
        
        Uses geometric Brownian motion with realistic parameters for crypto.
        """
        # Determine frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
        }
        freq = freq_map.get(interval, '1H')
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Crypto-specific parameters
        base_prices = {
            'BTC': 50000,
            'ETH': 3000,
            'BNB': 400,
            'SOL': 100,
            'MATIC': 1.0,
            'AVAX': 30,
        }
        
        # Extract base symbol (e.g., 'BTC' from 'BTCUSDT')
        base_symbol = symbol[:3] if len(symbol) > 3 else symbol
        base_price = base_prices.get(base_symbol, 1000)
        
        # Simulate with realistic crypto volatility
        np.random.seed(hash(symbol) % (2**32))
        
        # Higher volatility for crypto
        daily_volatility = 0.04  # 4% daily volatility
        interval_volatility = daily_volatility / np.sqrt(24)  # Adjust for hourly
        
        # Generate returns
        returns = np.random.normal(0.0001, interval_volatility, len(dates))
        
        # Add some trending behavior
        trend = np.linspace(0, 0.1, len(dates))
        returns += trend * 0.001
        
        # Calculate prices
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        df = pd.DataFrame(index=dates)
        df['Close'] = price_series
        
        # Simulate intrabar movement
        df['Open'] = df['Close'].shift(1).fillna(base_price)
        df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.001, 1.005, len(df))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.995, 0.999, len(df))
        
        # Simulate volume (higher volume on larger price moves)
        price_change_pct = df['Close'].pct_change().abs()
        base_volume = 1000000
        df['Volume'] = base_volume * (1 + price_change_pct * 10)
        df['Volume'] = df['Volume'].fillna(base_volume)
        
        logger.info(f"Simulated {len(df)} candles for {symbol}")
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def fetch_multi_asset_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1h',
        source: str = 'simulated',
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple assets.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            source: Data source
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            df = self.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                source=source,
            )
            
            if df is not None and not df.empty:
                data[symbol] = df
        
        logger.info(f"Fetched data for {len(data)}/{len(symbols)} symbols")
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        df = df.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        return df


if __name__ == "__main__":
    # Example usage
    fetcher = HistoricalCryptoDataFetcher()
    
    print("=== Historical Crypto Data Fetcher Test ===\n")
    
    # Fetch data for single asset
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    df = fetcher.fetch_historical_data(
        symbol='BTCUSDT',
        start_date=start_date,
        end_date=end_date,
        interval='1h',
        source='simulated',
    )
    
    print(f"Fetched {len(df)} candles for BTC")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Add technical indicators
    df_with_indicators = fetcher.add_technical_indicators(df)
    print(f"\nColumns with indicators: {list(df_with_indicators.columns)}")
    
    # Fetch multiple assets
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    multi_data = fetcher.fetch_multi_asset_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='simulated',
    )
    
    print(f"\nFetched data for {len(multi_data)} assets:")
    for symbol, data in multi_data.items():
        print(f"  {symbol}: {len(data)} candles, price range ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    print("\n✅ Historical data fetcher test completed!")
