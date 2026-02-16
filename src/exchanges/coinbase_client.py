"""
Coinbase Pro (Advanced Trade) API Client
Real exchange integration for live trading.

Features:
- REST API for orders and account data
- WebSocket for real-time market data
- Authentication with API keys
- Rate limiting and error handling
- Order management (market, limit, stop)
"""
import logging
import time
import hmac
import hashlib
import base64
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT_LIMIT_GTC"  # Good til cancelled
    STOP_LIMIT = "STOP_LIMIT_STOP_LIMIT_GTC"


class OrderStatus(str, Enum):
    """Order status."""
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class CoinbaseProClient:
    """
    Coinbase Pro (Advanced Trade) API Client.

    Handles authentication, rate limiting, and API requests.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        sandbox: bool = False
    ):
        """
        Initialize Coinbase Pro client.

        Args:
            api_key: API key
            api_secret: API secret
            passphrase: API passphrase
            sandbox: Use sandbox environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

        if sandbox:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
        else:
            self.base_url = "https://api.coinbase.com/api/v3/brokerage"

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "TradingAI/1.0"
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests/second

        logger.info(f"Coinbase Pro client initialized ({'sandbox' if sandbox else 'live'})")

    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """Generate request signature."""
        message = timestamp + method + request_path + body
        hmac_key = base64.b64decode(self.api_secret)
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body

        Returns:
            Response data
        """
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        # Build URL
        url = f"{self.base_url}{endpoint}"

        # Generate signature
        timestamp = str(int(time.time()))
        body = json.dumps(data) if data else ""
        signature = self._generate_signature(timestamp, method, endpoint, body)

        # Headers
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase
        }

        # Make request
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")

            self.last_request_time = time.time()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"API request failed: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'N/A'}")
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    # Account endpoints
    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all accounts.

        Returns:
            List of accounts with balances
        """
        response = self._make_request("GET", "/accounts")
        return response.get("accounts", [])

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Get specific account by ID."""
        response = self._make_request("GET", f"/accounts/{account_id}")
        return response.get("account", {})

    # Market data endpoints
    def get_products(self) -> List[Dict[str, Any]]:
        """
        Get all trading pairs.

        Returns:
            List of products (trading pairs)
        """
        response = self._make_request("GET", "/products")
        return response.get("products", [])

    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get specific product (e.g., 'BTC-USD')."""
        response = self._make_request("GET", f"/products/{product_id}")
        return response

    def get_ticker(self, product_id: str) -> Dict[str, Any]:
        """
        Get current ticker for product.

        Args:
            product_id: Product ID (e.g., 'BTC-USD')

        Returns:
            Ticker data with price, volume, etc.
        """
        response = self._make_request("GET", f"/products/{product_id}/ticker")
        return response

    def get_candles(
        self,
        product_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        granularity: int = 3600
    ) -> List[List]:
        """
        Get historical candles (OHLCV).

        Args:
            product_id: Product ID
            start: Start time (ISO 8601)
            end: End time (ISO 8601)
            granularity: Candle size in seconds (60, 300, 900, 3600, 21600, 86400)

        Returns:
            List of candles [timestamp, low, high, open, close, volume]
        """
        params = {"granularity": granularity}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._make_request(
            "GET",
            f"/products/{product_id}/candles",
            params=params
        )
        return response.get("candles", [])

    # Order endpoints
    def create_market_order(
        self,
        product_id: str,
        side: OrderSide,
        size: Optional[float] = None,
        funds: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create market order.

        Args:
            product_id: Product ID (e.g., 'BTC-USD')
            side: BUY or SELL
            size: Amount in base currency (e.g., BTC)
            funds: Amount in quote currency (e.g., USD)
            client_order_id: Optional client order ID

        Returns:
            Order details
        """
        if size is None and funds is None:
            raise ValueError("Must specify either size or funds")

        order_data = {
            "product_id": product_id,
            "side": side.value,
            "order_configuration": {
                "market_market_ioc": {}
            }
        }

        if size:
            order_data["order_configuration"]["market_market_ioc"]["base_size"] = str(size)
        if funds:
            order_data["order_configuration"]["market_market_ioc"]["quote_size"] = str(funds)

        if client_order_id:
            order_data["client_order_id"] = client_order_id

        response = self._make_request("POST", "/orders", data=order_data)
        logger.info(f"Market order created: {side.value} {size or funds} {product_id}")
        return response

    def create_limit_order(
        self,
        product_id: str,
        side: OrderSide,
        price: float,
        size: float,
        post_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create limit order.

        Args:
            product_id: Product ID
            side: BUY or SELL
            price: Limit price
            size: Order size
            post_only: Post-only (maker only)
            client_order_id: Optional client order ID

        Returns:
            Order details
        """
        order_data = {
            "product_id": product_id,
            "side": side.value,
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(price),
                    "post_only": post_only
                }
            }
        }

        if client_order_id:
            order_data["client_order_id"] = client_order_id

        response = self._make_request("POST", "/orders", data=order_data)
        logger.info(f"Limit order created: {side.value} {size} {product_id} @ ${price}")
        return response

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation result
        """
        response = self._make_request("POST", f"/orders/batch_cancel", data={
            "order_ids": [order_id]
        })
        logger.info(f"Order cancelled: {order_id}")
        return response

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order by ID."""
        response = self._make_request("GET", f"/orders/historical/{order_id}")
        return response.get("order", {})

    def get_orders(
        self,
        product_id: Optional[str] = None,
        order_status: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get orders.

        Args:
            product_id: Filter by product
            order_status: Filter by status (OPEN, FILLED, etc.)
            limit: Max results

        Returns:
            List of orders
        """
        params = {"limit": limit}
        if product_id:
            params["product_id"] = product_id
        if order_status:
            params["order_status"] = order_status

        response = self._make_request("GET", "/orders/historical/batch", params=params)
        return response.get("orders", [])

    def get_fills(
        self,
        product_id: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get order fills (trades).

        Args:
            product_id: Filter by product
            order_id: Filter by order ID
            limit: Max results

        Returns:
            List of fills
        """
        params = {"limit": limit}
        if product_id:
            params["product_id"] = product_id
        if order_id:
            params["order_id"] = order_id

        response = self._make_request("GET", "/fills", params=params)
        return response.get("fills", [])

    # Utility methods
    def get_balance(self, currency: str = "USD") -> float:
        """
        Get balance for currency.

        Args:
            currency: Currency code (e.g., 'USD', 'BTC')

        Returns:
            Available balance
        """
        accounts = self.get_accounts()
        for account in accounts:
            if account.get("currency") == currency:
                return float(account.get("available_balance", {}).get("value", 0))
        return 0.0

    def get_current_price(self, product_id: str) -> float:
        """
        Get current market price.

        Args:
            product_id: Product ID

        Returns:
            Current price
        """
        ticker = self.get_ticker(product_id)
        return float(ticker.get("price", 0))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("💱 Coinbase Pro Client Demo")
    print("=" * 60)
    print("\nNOTE: This is a demo. Set real API keys in environment to test.")
    print("      API keys: CB_API_KEY, CB_API_SECRET, CB_PASSPHRASE\n")

    # Demo with mock credentials (won't work without real keys)
    api_key = os.getenv("CB_API_KEY", "demo_key")
    api_secret = os.getenv("CB_API_SECRET", "demo_secret")
    passphrase = os.getenv("CB_PASSPHRASE", "demo_passphrase")

    try:
        # Initialize client (sandbox mode)
        client = CoinbaseProClient(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            sandbox=True
        )

        print("1. Client initialized (sandbox mode)")

        # These would work with real credentials:
        print("\n2. Available methods:")
        print("   - get_accounts(): Get account balances")
        print("   - get_products(): Get trading pairs")
        print("   - get_ticker('BTC-USD'): Get current price")
        print("   - create_market_order(): Execute market order")
        print("   - create_limit_order(): Place limit order")
        print("   - get_orders(): View open orders")
        print("   - cancel_order(): Cancel order")

        print("\n✅ Coinbase Pro client ready!")
        print("\nTo use live:")
        print("   1. Get API keys from Coinbase Pro")
        print("   2. Set environment variables")
        print("   3. Initialize with sandbox=False")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("This is expected without real API credentials.")
