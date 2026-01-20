"""
Binance API client for spot and futures market data.
"""
import os
import logging
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Binance exchange client for spot and futures data.
    """
    
    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.secret_key = secret_key or os.getenv('BINANCE_SECRET_KEY')
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-MBX-APIKEY': self.api_key})
        
        logger.info("Binance client initialized")
    
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get current ticker price.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Current price
        """
        try:
            url = f"{self.BASE_URL}/api/v3/ticker/price"
            params = {'symbol': symbol}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting ticker price for {symbol}: {e}")
            return None
    
    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000)
            
        Returns:
            List of OHLCV candles
        """
        try:
            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Format klines
            klines = []
            for k in data:
                klines.append({
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': k[8]
                })
            
            return klines
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """
        Get 24-hour statistics.
        
        Args:
            symbol: Trading pair
            
        Returns:
            24-hour stats dictionary
        """
        try:
            url = f"{self.BASE_URL}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price_change': float(data['priceChange']),
                'price_change_percent': float(data['priceChangePercent']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'volume': float(data['volume']),
                'quote_volume': float(data['quoteVolume']),
                'trades': int(data['count'])
            }
        except Exception as e:
            logger.error(f"Error getting 24h stats for {symbol}: {e}")
            return None
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """
        Get order book depth.
        
        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Order book with bids and asks
        """
        try:
            url = f"{self.BASE_URL}/api/v3/depth"
            params = {
                'symbol': symbol,
                'limit': limit
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                'timestamp': data['lastUpdateId']
            }
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return None
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate for perpetual futures.
        
        Args:
            symbol: Futures pair (e.g., 'BTCUSDT')
            
        Returns:
            Funding rate info
        """
        try:
            url = f"{self.FUTURES_URL}/fapi/v1/fundingRate"
            params = {
                'symbol': symbol,
                'limit': 1
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                latest = data[0]
                return {
                    'symbol': latest['symbol'],
                    'funding_rate': float(latest['fundingRate']),
                    'funding_time': latest['fundingTime'],
                    'mark_price': float(latest.get('markPrice', 0))
                }
            return None
        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {e}")
            return None
    
    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """
        Get open interest for futures.
        
        Args:
            symbol: Futures pair
            
        Returns:
            Open interest data
        """
        try:
            url = f"{self.FUTURES_URL}/fapi/v1/openInterest"
            params = {'symbol': symbol}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'open_interest': float(data['openInterest']),
                'timestamp': data['time']
            }
        except Exception as e:
            logger.error(f"Error getting open interest for {symbol}: {e}")
            return None
    
    def get_top_traders_long_short_ratio(self, symbol: str, period: str = '5m') -> Optional[Dict]:
        """
        Get top traders long/short ratio.
        
        Args:
            symbol: Futures pair
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            
        Returns:
            Long/short ratio data
        """
        try:
            url = f"{self.FUTURES_URL}/futures/data/topLongShortPositionRatio"
            params = {
                'symbol': symbol,
                'period': period,
                'limit': 1
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                latest = data[0]
                return {
                    'symbol': latest['symbol'],
                    'long_short_ratio': float(latest['longShortRatio']),
                    'long_account': float(latest['longAccount']),
                    'short_account': float(latest['shortAccount']),
                    'timestamp': latest['timestamp']
                }
            return None
        except Exception as e:
            logger.error(f"Error getting long/short ratio for {symbol}: {e}")
            return None
    
    def get_liquidation_orders(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent liquidation orders.
        
        Args:
            symbol: Futures pair
            limit: Number of orders (max 1000)
            
        Returns:
            List of liquidation orders
        """
        try:
            url = f"{self.FUTURES_URL}/fapi/v1/allForceOrders"
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            liquidations = []
            for order in data:
                liquidations.append({
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'order_type': order['origType'],
                    'price': float(order['price']),
                    'quantity': float(order['origQty']),
                    'executed_qty': float(order['executedQty']),
                    'time': order['time']
                })
            
            return liquidations
        except Exception as e:
            logger.error(f"Error getting liquidations for {symbol}: {e}")
            return []


if __name__ == "__main__":
    # Test Binance client
    client = BinanceClient()
    
    print("=== Binance Client Test ===")
    
    # Get BTC price
    btc_price = client.get_ticker_price('BTCUSDT')
    print(f"BTC Price: ${btc_price:,.2f}")
    
    # Get 24h stats
    stats = client.get_24h_stats('BTCUSDT')
    if stats:
        print(f"24h Change: {stats['price_change_percent']:.2f}%")
        print(f"24h Volume: {stats['volume']:,.2f} BTC")
    
    # Get funding rate
    funding = client.get_funding_rate('BTCUSDT')
    if funding:
        print(f"Funding Rate: {funding['funding_rate']*100:.4f}%")
    
    # Get open interest
    oi = client.get_open_interest('BTCUSDT')
    if oi:
        print(f"Open Interest: {oi['open_interest']:,.2f} BTC")
    
    print("\n✅ Binance client test completed!")
