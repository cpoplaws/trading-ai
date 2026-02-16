"""
Binance Trading Client

Complete Binance API client with trading capabilities, authentication, and rate limiting.
"""
import os
import time
import hmac
import hashlib
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlencode
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class TimeInForce(str, Enum):
    """Time in force."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class BinanceTradingClient:
    """
    Binance trading client with full API support.

    Features:
    - Market and limit orders
    - Account management
    - Position tracking
    - HMAC-SHA256 authentication
    - Rate limiting (1200 req/min)
    - Testnet and mainnet support
    """

    # API URLs
    MAINNET_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"

    # Rate limits (requests per minute)
    RATE_LIMIT = 1200
    RATE_LIMIT_ORDER = 50  # For order placement

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize Binance trading client.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (default: True for safety)
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet

        if not self.api_key or not self.api_secret:
            logger.warning("No API credentials provided. Read-only mode.")

        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-MBX-APIKEY': self.api_key})

        # Rate limiting
        self.request_count = 0
        self.request_window_start = time.time()
        self.order_count = 0
        self.order_window_start = time.time()

        env = "TESTNET" if testnet else "MAINNET"
        logger.info(f"Binance Trading Client initialized ({env})")

        if testnet:
            logger.warning("⚠️  TESTNET MODE - No real money at risk")
        else:
            logger.warning("🚨 MAINNET MODE - REAL MONEY AT RISK!")

    def _generate_signature(self, params: Dict) -> str:
        """
        Generate HMAC SHA256 signature.

        Args:
            params: Request parameters

        Returns:
            Signature string
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _check_rate_limit(self, order_endpoint: bool = False):
        """Check and enforce rate limits."""
        current_time = time.time()

        # Check general rate limit
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time

        if self.request_count >= self.RATE_LIMIT:
            wait_time = 60 - (current_time - self.request_window_start)
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.request_count = 0
                self.request_window_start = time.time()

        # Check order rate limit
        if order_endpoint:
            if current_time - self.order_window_start >= 60:
                self.order_count = 0
                self.order_window_start = current_time

            if self.order_count >= self.RATE_LIMIT_ORDER:
                wait_time = 60 - (current_time - self.order_window_start)
                if wait_time > 0:
                    logger.warning(f"Order rate limit reached. Waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    self.order_count = 0
                    self.order_window_start = time.time()

        self.request_count += 1
        if order_endpoint:
            self.order_count += 1

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False
    ) -> Optional[Dict]:
        """
        Make API request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request requires signature

        Returns:
            Response JSON
        """
        self._check_rate_limit(order_endpoint='/order' in endpoint)

        params = params or {}

        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("API credentials required for signed requests")

            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=10)
            elif method == "POST":
                response = self.session.post(url, params=params, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    # ============================================================
    # Account Management
    # ============================================================

    def get_account_info(self) -> Dict:
        """
        Get account information including balances.

        Returns:
            Account info dictionary
        """
        return self._make_request("GET", "/api/v3/account", signed=True)

    def get_balances(self) -> List[Dict]:
        """
        Get all non-zero balances.

        Returns:
            List of balances
        """
        account = self.get_account_info()
        balances = []

        for balance in account.get('balances', []):
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                balances.append({
                    'asset': balance['asset'],
                    'free': free,
                    'locked': locked,
                    'total': free + locked
                })

        return balances

    def get_balance(self, asset: str) -> Dict:
        """
        Get balance for specific asset.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')

        Returns:
            Balance info
        """
        balances = self.get_balances()
        for balance in balances:
            if balance['asset'] == asset:
                return balance

        return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}

    # ============================================================
    # Market Data
    # ============================================================

    def get_ticker_price(self, symbol: str) -> float:
        """Get current ticker price."""
        data = self._make_request("GET", "/api/v3/ticker/price", {'symbol': symbol})
        return float(data['price'])

    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book."""
        params = {'symbol': symbol, 'limit': limit}
        data = self._make_request("GET", "/api/v3/depth", params)

        return {
            'bids': [[float(p), float(q)] for p, q in data['bids']],
            'asks': [[float(p), float(q)] for p, q in data['asks']]
        }

    # ============================================================
    # Order Management
    # ============================================================

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> Dict:
        """
        Place market order.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: BUY or SELL
            quantity: Order quantity

        Returns:
            Order response
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.MARKET.value,
            'quantity': quantity
        }

        logger.info(f"Placing MARKET {side.value} order: {quantity} {symbol}")
        result = self._make_request("POST", "/api/v3/order", params, signed=True)
        logger.info(f"Order placed: {result.get('orderId')}")

        return result

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        time_in_force: TimeInForce = TimeInForce.GTC
    ) -> Dict:
        """
        Place limit order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            price: Limit price
            time_in_force: Time in force (GTC, IOC, FOK)

        Returns:
            Order response
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.LIMIT.value,
            'quantity': quantity,
            'price': price,
            'timeInForce': time_in_force.value
        }

        logger.info(f"Placing LIMIT {side.value} order: {quantity} {symbol} @ ${price}")
        result = self._make_request("POST", "/api/v3/order", params, signed=True)
        logger.info(f"Order placed: {result.get('orderId')}")

        return result

    def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float
    ) -> Dict:
        """
        Place stop-loss order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            stop_price: Stop price

        Returns:
            Order response
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.STOP_LOSS.value,
            'quantity': quantity,
            'stopPrice': stop_price
        }

        logger.info(f"Placing STOP LOSS {side.value} order: {quantity} {symbol} @ ${stop_price}")
        result = self._make_request("POST", "/api/v3/order", params, signed=True)
        logger.info(f"Order placed: {result.get('orderId')}")

        return result

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """
        Cancel an order.

        Args:
            symbol: Trading pair
            order_id: Order ID

        Returns:
            Cancel response
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }

        logger.info(f"Canceling order {order_id} for {symbol}")
        result = self._make_request("DELETE", "/api/v3/order", params, signed=True)
        logger.info(f"Order canceled: {order_id}")

        return result

    def cancel_all_orders(self, symbol: str) -> List[Dict]:
        """
        Cancel all open orders for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            List of canceled orders
        """
        params = {'symbol': symbol}

        logger.info(f"Canceling all orders for {symbol}")
        result = self._make_request("DELETE", "/api/v3/openOrders", params, signed=True)
        logger.info(f"Canceled {len(result)} orders")

        return result

    def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """
        Get order status.

        Args:
            symbol: Trading pair
            order_id: Order ID

        Returns:
            Order status
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }

        return self._make_request("GET", "/api/v3/order", params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders.

        Args:
            symbol: Trading pair (optional, all if None)

        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request("GET", "/api/v3/openOrders", params, signed=True)

    def get_order_history(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get order history.

        Args:
            symbol: Trading pair
            limit: Max number of orders (max 1000)

        Returns:
            List of historical orders
        """
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000)
        }

        return self._make_request("GET", "/api/v3/allOrders", params, signed=True)

    def get_my_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get trade history.

        Args:
            symbol: Trading pair
            limit: Max number of trades (max 1000)

        Returns:
            List of trades
        """
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000)
        }

        return self._make_request("GET", "/api/v3/myTrades", params, signed=True)

    # ============================================================
    # Helper Methods
    # ============================================================

    def get_trading_fees(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get trading fees.

        Args:
            symbol: Trading pair (optional)

        Returns:
            Fee information
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request("GET", "/api/v3/tradeFee", params, signed=True)

    def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules."""
        return self._make_request("GET", "/api/v3/exchangeInfo")

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get trading rules for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Symbol info
        """
        exchange_info = self.get_exchange_info()

        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s

        return None

    def test_connectivity(self) -> bool:
        """Test API connectivity."""
        try:
            self._make_request("GET", "/api/v3/ping")
            return True
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False

    def get_server_time(self) -> int:
        """Get server time (Unix timestamp ms)."""
        data = self._make_request("GET", "/api/v3/time")
        return data['serverTime']


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("BINANCE TRADING CLIENT TEST")
    print("=" * 70)

    # Initialize client (TESTNET by default)
    client = BinanceTradingClient(testnet=True)

    print(f"\n🔗 Testing connectivity...")
    if client.test_connectivity():
        print("✅ Connected to Binance")
    else:
        print("❌ Connection failed")
        exit(1)

    # Get server time
    server_time = client.get_server_time()
    print(f"📅 Server Time: {datetime.fromtimestamp(server_time/1000)}")

    # Get ticker price
    print(f"\n💰 Getting BTC price...")
    btc_price = client.get_ticker_price('BTCUSDT')
    print(f"BTC Price: ${btc_price:,.2f}")

    # Test account access (if credentials provided)
    if client.api_key and client.api_secret:
        print(f"\n👤 Testing account access...")

        try:
            # Get balances
            balances = client.get_balances()
            print(f"✅ Account accessible")
            print(f"Non-zero balances: {len(balances)}")

            for balance in balances[:5]:
                print(f"  {balance['asset']}: {balance['total']:.8f}")

            # Get open orders
            open_orders = client.get_open_orders()
            print(f"\n📋 Open orders: {len(open_orders)}")

        except Exception as e:
            print(f"❌ Account access failed: {e}")
            print("Make sure BINANCE_API_KEY and BINANCE_API_SECRET are set")

    else:
        print("\n⚠️  No API credentials provided")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET to test trading features")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
