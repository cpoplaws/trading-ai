"""
CEX Connector - Unified interface for centralized exchanges

Provides a common interface for Binance, Coinbase, and other CEX clients.
Handles order placement, status checking, balance management, and price feeds.

Responsibilities:
1. Unified interface across all CEX platforms
2. Order placement with retry logic
3. Balance checking before trades
4. Real-time price data
5. Error handling for rate limits
6. Order status tracking
"""

from dataclasses import dataclass
from typing import Dict, Optional, Literal, List
from enum import Enum
import logging
import time
import os

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"


@dataclass
class OrderRequest:
    """Unified order request"""
    exchange: Exchange
    symbol: str  # "BTC/USDT" or "BTCUSDT" - will normalize
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: float
    price: Optional[float] = None  # For limit orders
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    """Unified order result"""
    success: bool
    order_id: str
    exchange: Exchange
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    price: Optional[float]
    average_fill_price: Optional[float]
    status: str  # "open", "filled", "cancelled", "failed"
    fees: float
    timestamp: str
    error: Optional[str] = None


@dataclass
class Balance:
    """Account balance"""
    exchange: Exchange
    asset: str
    free: float  # Available
    locked: float  # In orders
    total: float


@dataclass
class Price:
    """Current price"""
    exchange: Exchange
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: str


class CEXConnector:
    """
    Unified connector for centralized exchanges.

    Supports: Binance, Coinbase
    Future: Kraken, Bybit, OKX

    Features:
    - Unified interface across all exchanges
    - Automatic symbol normalization (BTC/USDT <-> BTCUSDT)
    - Balance checks before trading
    - Retry logic for failed orders
    - Rate limit handling
    """

    def __init__(
        self,
        binance_api_key: Optional[str] = None,
        binance_api_secret: Optional[str] = None,
        coinbase_api_key: Optional[str] = None,
        coinbase_api_secret: Optional[str] = None,
        coinbase_passphrase: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize CEX connector.

        Args:
            binance_api_key: Binance API key
            binance_api_secret: Binance API secret
            coinbase_api_key: Coinbase API key
            coinbase_api_secret: Coinbase API secret
            coinbase_passphrase: Coinbase passphrase
            testnet: Use testnet/sandbox (default: True for safety)
        """
        self.testnet = testnet
        self.clients = {}

        # Initialize Binance client
        if binance_api_key and binance_api_secret:
            try:
                from src.exchanges.binance_trading_client import BinanceTradingClient
                self.clients[Exchange.BINANCE] = BinanceTradingClient(
                    api_key=binance_api_key,
                    api_secret=binance_api_secret,
                    testnet=testnet
                )
                logger.info(f"✅ Binance client initialized ({'testnet' if testnet else 'mainnet'})")
            except Exception as e:
                logger.error(f"Failed to initialize Binance client: {e}")
        else:
            logger.warning("Binance credentials not provided - Binance trading disabled")

        # Initialize Coinbase client
        if coinbase_api_key and coinbase_api_secret and coinbase_passphrase:
            try:
                from src.exchanges.coinbase_client import CoinbaseProClient
                self.clients[Exchange.COINBASE] = CoinbaseProClient(
                    api_key=coinbase_api_key,
                    api_secret=coinbase_api_secret,
                    passphrase=coinbase_passphrase,
                    sandbox=testnet
                )
                logger.info(f"✅ Coinbase client initialized ({'sandbox' if testnet else 'live'})")
            except Exception as e:
                logger.error(f"Failed to initialize Coinbase client: {e}")
        else:
            logger.warning("Coinbase credentials not provided - Coinbase trading disabled")

        logger.info(f"CEX Connector initialized: {len(self.clients)} exchanges available")
        if testnet:
            logger.warning("⚠️  TESTNET/SANDBOX MODE - No real money at risk")
        else:
            logger.warning("🚨 MAINNET MODE - REAL MONEY AT RISK!")

    def _normalize_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        Normalize symbol format for exchange.

        Unified format: "BTC/USDT"
        Binance format: "BTCUSDT" (no separator)
        Coinbase format: "BTC-USDT" (hyphen separator)

        Args:
            symbol: Symbol in any format
            exchange: Target exchange

        Returns:
            Exchange-specific symbol format
        """
        # Remove separators to get base format
        clean = symbol.replace("/", "").replace("-", "")

        if exchange == Exchange.BINANCE:
            # Binance uses no separator: BTCUSDT
            return clean
        elif exchange == Exchange.COINBASE:
            # Coinbase uses hyphen: BTC-USDT
            # Common pairs
            bases = ["BTC", "ETH", "SOL", "AVAX", "MATIC", "LINK", "UNI", "AAVE"]
            quotes = ["USDT", "USDC", "USD", "BTC", "ETH"]

            for base in bases:
                for quote in quotes:
                    if clean == base + quote:
                        return f"{base}-{quote}"

            # Fallback: assume first 3-4 chars are base
            if len(clean) >= 6:
                base = clean[:3]
                quote = clean[3:]
                return f"{base}-{quote}"

            return clean
        else:
            return symbol

    def get_balance(self, exchange: Exchange, asset: str) -> Balance:
        """
        Get balance for asset on exchange.

        Args:
            exchange: Exchange to query
            asset: Asset symbol (e.g., "BTC", "USDT")

        Returns:
            Balance information
        """
        if exchange not in self.clients:
            raise ValueError(f"Exchange {exchange.value} not initialized")

        client = self.clients[exchange]

        try:
            if exchange == Exchange.BINANCE:
                balance_data = client.get_balance(asset)
                return Balance(
                    exchange=exchange,
                    asset=asset,
                    free=balance_data.get("free", 0.0),
                    locked=balance_data.get("locked", 0.0),
                    total=balance_data.get("total", 0.0)
                )
            elif exchange == Exchange.COINBASE:
                balance_value = client.get_balance(asset)
                return Balance(
                    exchange=exchange,
                    asset=asset,
                    free=balance_value,
                    locked=0.0,  # Coinbase doesn't separate locked
                    total=balance_value
                )
        except Exception as e:
            logger.error(f"Error getting balance from {exchange.value}: {e}")
            return Balance(
                exchange=exchange,
                asset=asset,
                free=0.0,
                locked=0.0,
                total=0.0
            )

    def get_price(self, exchange: Exchange, symbol: str) -> Price:
        """
        Get current price for symbol.

        Args:
            exchange: Exchange to query
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Price information
        """
        if exchange not in self.clients:
            raise ValueError(f"Exchange {exchange.value} not initialized")

        client = self.clients[exchange]
        normalized_symbol = self._normalize_symbol(symbol, exchange)

        try:
            if exchange == Exchange.BINANCE:
                price = client.get_ticker_price(normalized_symbol)
                # Binance ticker doesn't have bid/ask, use same for all
                return Price(
                    exchange=exchange,
                    symbol=symbol,
                    bid=price,
                    ask=price,
                    last=price,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                )
            elif exchange == Exchange.COINBASE:
                ticker = client.get_ticker(normalized_symbol)
                price = float(ticker.get("price", 0))
                return Price(
                    exchange=exchange,
                    symbol=symbol,
                    bid=price,  # TODO: Get actual bid/ask from orderbook
                    ask=price,
                    last=price,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                )
        except Exception as e:
            logger.error(f"Error getting price from {exchange.value}: {e}")
            raise

    def place_order(
        self,
        order_request: OrderRequest,
        max_retries: int = 3,
        check_balance: bool = True
    ) -> OrderResult:
        """
        Place order on exchange with retry logic.

        Args:
            order_request: Order details
            max_retries: Max retry attempts
            check_balance: Check balance before placing order

        Returns:
            Order result
        """
        exchange = order_request.exchange

        if exchange not in self.clients:
            return OrderResult(
                success=False,
                order_id="",
                exchange=exchange,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                filled_quantity=0.0,
                price=order_request.price,
                average_fill_price=None,
                status="failed",
                fees=0.0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error=f"Exchange {exchange.value} not initialized"
            )

        client = self.clients[exchange]
        normalized_symbol = self._normalize_symbol(order_request.symbol, exchange)

        # Check balance if requested
        if check_balance:
            try:
                if order_request.side == "buy":
                    # Check quote currency (USDT for BTC/USDT)
                    quote = order_request.symbol.split("/")[-1]
                    required = order_request.quantity * (order_request.price or self.get_price(exchange, order_request.symbol).last)
                    balance = self.get_balance(exchange, quote)

                    if balance.free < required:
                        return OrderResult(
                            success=False,
                            order_id="",
                            exchange=exchange,
                            symbol=order_request.symbol,
                            side=order_request.side,
                            order_type=order_request.order_type,
                            quantity=order_request.quantity,
                            filled_quantity=0.0,
                            price=order_request.price,
                            average_fill_price=None,
                            status="failed",
                            fees=0.0,
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            error=f"Insufficient balance: {balance.free} {quote} < {required} {quote}"
                        )
                else:  # sell
                    # Check base currency (BTC for BTC/USDT)
                    base = order_request.symbol.split("/")[0]
                    balance = self.get_balance(exchange, base)

                    if balance.free < order_request.quantity:
                        return OrderResult(
                            success=False,
                            order_id="",
                            exchange=exchange,
                            symbol=order_request.symbol,
                            side=order_request.side,
                            order_type=order_request.order_type,
                            quantity=order_request.quantity,
                            filled_quantity=0.0,
                            price=order_request.price,
                            average_fill_price=None,
                            status="failed",
                            fees=0.0,
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            error=f"Insufficient balance: {balance.free} {base} < {order_request.quantity} {base}"
                        )
            except Exception as e:
                logger.warning(f"Balance check failed: {e}, proceeding with order")

        # Attempt order placement with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Placing order (attempt {attempt + 1}/{max_retries}): {order_request.side} {order_request.quantity} {order_request.symbol} on {exchange.value}")

                if exchange == Exchange.BINANCE:
                    from src.exchanges.binance_trading_client import OrderSide

                    side_enum = OrderSide.BUY if order_request.side == "buy" else OrderSide.SELL

                    if order_request.order_type == "market":
                        result = client.place_market_order(
                            symbol=normalized_symbol,
                            side=side_enum,
                            quantity=order_request.quantity
                        )
                    else:  # limit
                        result = client.place_limit_order(
                            symbol=normalized_symbol,
                            side=side_enum,
                            quantity=order_request.quantity,
                            price=order_request.price
                        )

                    # Parse Binance response
                    return OrderResult(
                        success=True,
                        order_id=str(result.get("orderId")),
                        exchange=exchange,
                        symbol=order_request.symbol,
                        side=order_request.side,
                        order_type=order_request.order_type,
                        quantity=order_request.quantity,
                        filled_quantity=float(result.get("executedQty", 0)),
                        price=order_request.price,
                        average_fill_price=float(result.get("price", 0)) if result.get("price") else None,
                        status=result.get("status", "").lower(),
                        fees=0.0,  # TODO: Calculate from fills
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                    )

                elif exchange == Exchange.COINBASE:
                    from src.exchanges.coinbase_client import OrderSide as CBOrderSide

                    side_enum = CBOrderSide.BUY if order_request.side == "buy" else CBOrderSide.SELL

                    if order_request.order_type == "market":
                        result = client.create_market_order(
                            product_id=normalized_symbol,
                            side=side_enum,
                            size=order_request.quantity
                        )
                    else:  # limit
                        result = client.create_limit_order(
                            product_id=normalized_symbol,
                            side=side_enum,
                            price=order_request.price,
                            size=order_request.quantity
                        )

                    # Parse Coinbase response
                    order_id = result.get("success_response", {}).get("order_id", "")
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        exchange=exchange,
                        symbol=order_request.symbol,
                        side=order_request.side,
                        order_type=order_request.order_type,
                        quantity=order_request.quantity,
                        filled_quantity=0.0,  # Check later with get_order_status
                        price=order_request.price,
                        average_fill_price=None,
                        status="pending",
                        fees=0.0,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                    )

            except Exception as e:
                logger.error(f"Order placement failed (attempt {attempt + 1}): {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return OrderResult(
                        success=False,
                        order_id="",
                        exchange=exchange,
                        symbol=order_request.symbol,
                        side=order_request.side,
                        order_type=order_request.order_type,
                        quantity=order_request.quantity,
                        filled_quantity=0.0,
                        price=order_request.price,
                        average_fill_price=None,
                        status="failed",
                        fees=0.0,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        error=str(e)
                    )

        # Should not reach here
        return OrderResult(
            success=False,
            order_id="",
            exchange=exchange,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            filled_quantity=0.0,
            price=order_request.price,
            average_fill_price=None,
            status="failed",
            fees=0.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            error="Unknown error"
        )

    def get_order_status(self, exchange: Exchange, order_id: str, symbol: str) -> OrderResult:
        """
        Get order status.

        Args:
            exchange: Exchange to query
            order_id: Order ID
            symbol: Trading pair

        Returns:
            Order status
        """
        if exchange not in self.clients:
            raise ValueError(f"Exchange {exchange.value} not initialized")

        client = self.clients[exchange]
        normalized_symbol = self._normalize_symbol(symbol, exchange)

        try:
            if exchange == Exchange.BINANCE:
                result = client.get_order_status(normalized_symbol, int(order_id))

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    exchange=exchange,
                    symbol=symbol,
                    side=result.get("side", "").lower(),
                    order_type=result.get("type", "").lower(),
                    quantity=float(result.get("origQty", 0)),
                    filled_quantity=float(result.get("executedQty", 0)),
                    price=float(result.get("price", 0)) if result.get("price") else None,
                    average_fill_price=float(result.get("avgPrice", 0)) if result.get("avgPrice") else None,
                    status=result.get("status", "").lower(),
                    fees=0.0,  # TODO: Get from fills
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                )

            elif exchange == Exchange.COINBASE:
                result = client.get_order(order_id)

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    exchange=exchange,
                    symbol=symbol,
                    side=result.get("side", "").lower(),
                    order_type=result.get("type", "").lower(),
                    quantity=float(result.get("size", 0)),
                    filled_quantity=float(result.get("filled_size", 0)),
                    price=float(result.get("price", 0)) if result.get("price") else None,
                    average_fill_price=float(result.get("executed_value", 0)) / float(result.get("filled_size", 1)),
                    status=result.get("status", "").lower(),
                    fees=float(result.get("fill_fees", 0)),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                )

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            raise

    def cancel_order(self, exchange: Exchange, order_id: str, symbol: str) -> bool:
        """
        Cancel order.

        Args:
            exchange: Exchange
            order_id: Order ID
            symbol: Trading pair

        Returns:
            Success status
        """
        if exchange not in self.clients:
            raise ValueError(f"Exchange {exchange.value} not initialized")

        client = self.clients[exchange]
        normalized_symbol = self._normalize_symbol(symbol, exchange)

        try:
            if exchange == Exchange.BINANCE:
                client.cancel_order(normalized_symbol, int(order_id))
                logger.info(f"Order {order_id} cancelled on {exchange.value}")
                return True
            elif exchange == Exchange.COINBASE:
                client.cancel_order(order_id)
                logger.info(f"Order {order_id} cancelled on {exchange.value}")
                return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_available_exchanges(self) -> List[Exchange]:
        """Get list of initialized exchanges."""
        return list(self.clients.keys())

    def is_exchange_available(self, exchange: Exchange) -> bool:
        """Check if exchange is initialized and available."""
        return exchange in self.clients


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("CEX CONNECTOR TEST")
    print("="*70)

    # Initialize connector (testnet mode)
    connector = CEXConnector(
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_secret=os.getenv("BINANCE_API_SECRET"),
        coinbase_api_key=os.getenv("CB_API_KEY"),
        coinbase_api_secret=os.getenv("CB_API_SECRET"),
        coinbase_passphrase=os.getenv("CB_PASSPHRASE"),
        testnet=True
    )

    print(f"\nAvailable exchanges: {[e.value for e in connector.get_available_exchanges()]}")

    # Test symbol normalization
    print("\n--- Symbol Normalization Test ---")
    test_symbols = ["BTC/USDT", "BTCUSDT", "BTC-USDT"]
    for sym in test_symbols:
        binance = connector._normalize_symbol(sym, Exchange.BINANCE)
        coinbase = connector._normalize_symbol(sym, Exchange.COINBASE)
        print(f"  {sym} -> Binance: {binance}, Coinbase: {coinbase}")

    # Test price fetching
    if Exchange.BINANCE in connector.clients:
        print("\n--- Price Test (Binance) ---")
        try:
            price = connector.get_price(Exchange.BINANCE, "BTC/USDT")
            print(f"  BTC/USDT: ${price.last:,.2f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Test balance fetching
    if Exchange.BINANCE in connector.clients and os.getenv("BINANCE_API_KEY"):
        print("\n--- Balance Test (Binance) ---")
        try:
            balance = connector.get_balance(Exchange.BINANCE, "USDT")
            print(f"  USDT: {balance.free:.2f} (free), {balance.total:.2f} (total)")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*70)
    print("✅ CEX Connector ready for integration!")
    print("="*70)
