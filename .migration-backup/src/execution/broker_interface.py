"""
Abstract base class for broker interfaces and implementations.

This module provides a standard interface for connecting to different brokers
(Alpaca, Interactive Brokers, etc.) for paper and live trading.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv

from ..utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class OrderSide(Enum):
    """Order side (buy/sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"


class TimeInForce(Enum):
    """Time in force for orders."""

    DAY = "day"
    GTC = "gtc"  # Good till canceled
    OPG = "opg"  # Market on open
    CLS = "cls"  # Market on close
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


@dataclass
class Order:
    """Order data class."""

    order_id: str
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce
    status: OrderStatus
    created_at: datetime
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    def __getitem__(self, item):
        """Enable dict-style access for compatibility with existing call sites."""
        return getattr(self, item)


@dataclass
class Position:
    """Position data class."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float  # Unrealized profit/loss percentage
    side: str  # "long" or "short"


@dataclass
class Account:
    """Account data class."""

    account_number: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    last_equity: float
    long_market_value: float
    short_market_value: float


class BrokerInterface(ABC):
    """
    Abstract base class for broker integrations.
    
    All broker implementations must implement these methods.
    """

    def check_connection(self) -> bool:
        """Simple wrapper to verify connection status."""
        try:
            return self.connect()
        except Exception as exc:
            logger.error(f"Connection check failed: {exc}")
            return False

    def get_portfolio_value(self) -> float:
        """Helper to fetch portfolio value if available."""
        account = self.get_account_info()
        if isinstance(account, dict):
            return float(account.get("portfolio_value", 0.0))
        return getattr(account, "portfolio_value", 0.0)

    def get_buying_power(self) -> float:
        """Helper to fetch buying power if available."""
        account = self.get_account_info()
        if isinstance(account, dict):
            return float(account.get("buying_power", 0.0))
        return getattr(account, "buying_power", 0.0)

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker API.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass

    @abstractmethod
    def get_account_info(self) -> Account:
        """
        Get account information.
        
        Returns:
            Account object with account details
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
        """
        pass

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Returns:
            Position object or None if no position exists
        """
        logger.warning("get_position not implemented")
        return None

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Place an order.
        
        Args:
            symbol: Stock ticker symbol
            qty: Quantity of shares
            side: Buy or sell
            order_type: Market, limit, stop, etc.
            time_in_force: Day, GTC, etc.
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order object or None if failed
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if canceled successfully, False otherwise
        """
        pass

    def modify_order(self, order_id: str, new_params: Dict) -> bool:
        """
        Modify an existing order with new parameters.

        Args:
            order_id: Order ID to modify
            new_params: Dictionary of parameters to update

        Returns:
            True if modification successful, False otherwise
        """
        logger.warning("modify_order not implemented")
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        logger.warning("get_order not implemented")
        return None

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get orders, optionally filtered by status.
        
        Args:
            status: Filter by order status (optional)
            
        Returns:
            List of Order objects
        """
        logger.warning("get_orders not implemented")
        return []

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Current price or None if unavailable
        """
        logger.warning("get_current_price not implemented")
        return None


class MockBroker(BrokerInterface):
    """
    Mock broker for testing without real API credentials.
    
    Simulates broker operations for development and testing.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize mock broker.
        
        Args:
            initial_capital: Starting cash balance
        """
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.connected = False
        logger.info(f"MockBroker initialized with ${initial_capital:,.2f}")

    def connect(self) -> bool:
        """Establish mock connection."""
        self.connected = True
        logger.info("✅ MockBroker connected successfully")
        return True

    def disconnect(self) -> None:
        """Disconnect from mock broker."""
        self.connected = False
        logger.info("MockBroker disconnected")

    def get_account_info(self) -> Account:
        """Get mock account information."""
        portfolio_value = self.cash
        long_market_value = 0.0

        for pos in self.positions.values():
            long_market_value += pos.market_value

        portfolio_value += long_market_value

        return Account(
            account_number="MOCK123456",
            cash=self.cash,
            portfolio_value=portfolio_value,
            buying_power=self.cash,
            equity=portfolio_value,
            last_equity=self.initial_capital,
            long_market_value=long_market_value,
            short_market_value=0.0,
        )

    def get_positions(self) -> List[Position]:
        """Get all mock positions."""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get mock position for symbol."""
        return self.positions.get(symbol)

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> Optional[Order]:
        """Place mock order."""
        if not self.connected:
            logger.error("Cannot place order: not connected")
            return None

        qty = qty if qty is not None else quantity
        if qty is None:
            logger.error("Quantity is required to place order")
            return None

        self.order_counter += 1
        order_id = f"MOCK_{self.order_counter:06d}"

        # Simulate immediate fill for market orders
        current_price = self.get_current_price(symbol) or 100.0
        filled_qty = qty
        filled_avg_price = current_price

        order = Order(
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            status=OrderStatus.FILLED,
            created_at=datetime.now(),
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        self.orders[order_id] = order

        # Update positions and cash
        if side == OrderSide.BUY:
            cost = filled_qty * filled_avg_price
            if cost > self.cash:
                logger.error(f"Insufficient funds: need ${cost:,.2f}, have ${self.cash:,.2f}")
                order.status = OrderStatus.REJECTED
                return order

            self.cash -= cost

            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos.qty + filled_qty
                total_cost = (pos.qty * pos.avg_entry_price) + cost
                new_avg_price = total_cost / total_qty
                pos.qty = total_qty
                pos.avg_entry_price = new_avg_price
                pos.current_price = current_price
                pos.market_value = total_qty * current_price
                pos.unrealized_pl = (current_price - new_avg_price) * total_qty
                pos.unrealized_plpc = (current_price / new_avg_price - 1) * 100
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=filled_qty,
                    avg_entry_price=filled_avg_price,
                    current_price=current_price,
                    market_value=filled_qty * current_price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    side="long",
                )

            logger.info(f"✅ BUY {filled_qty} {symbol} @ ${filled_avg_price:.2f}")

        elif side == OrderSide.SELL:
            if symbol not in self.positions:
                logger.error(f"Cannot sell: no position in {symbol}")
                order.status = OrderStatus.REJECTED
                return order

            pos = self.positions[symbol]
            if pos.qty < filled_qty:
                logger.error(f"Cannot sell {filled_qty}: only have {pos.qty} shares")
                order.status = OrderStatus.REJECTED
                return order

            proceeds = filled_qty * filled_avg_price
            self.cash += proceeds

            pos.qty -= filled_qty
            if pos.qty == 0:
                del self.positions[symbol]
            else:
                pos.current_price = current_price
                pos.market_value = pos.qty * current_price
                pos.unrealized_pl = (current_price - pos.avg_entry_price) * pos.qty
                pos.unrealized_plpc = (current_price / pos.avg_entry_price - 1) * 100

            logger.info(f"✅ SELL {filled_qty} {symbol} @ ${filled_avg_price:.2f}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.NEW, OrderStatus.ACCEPTED]:
                order.status = OrderStatus.CANCELED
                logger.info(f"✅ Order {order_id} canceled")
                return True
        logger.warning(f"Cannot cancel order {order_id}")
        return False

    def modify_order(self, order_id: str, new_params: Dict) -> bool:
        """Modify an existing mock order."""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for modification")
            return False
        order = self.orders[order_id]
        for key, value in new_params.items():
            if hasattr(order, key):
                setattr(order, key, value)
        order.status = OrderStatus.REPLACED
        logger.info(f"✅ Order {order_id} modified")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get mock order."""
        return self.orders.get(order_id)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get mock orders."""
        if status:
            return [o for o in self.orders.values() if o.status == status]
        return list(self.orders.values())

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get mock current price (random walk simulation)."""
        import random

        # Simple mock: random price between 50-150
        base_price = 100.0
        if symbol in self.positions:
            base_price = self.positions[symbol].current_price

        # Add small random variation
        variation = random.uniform(-2, 2)
        return max(1.0, base_price + variation)


def create_broker(paper_trading: bool = True, mock: bool = False):
    """
    Factory helper to create broker instances.
    """
    if mock:
        broker = MockBroker()
        broker.connect()
        return broker

    try:
        from execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper_trading=paper_trading)
        broker.connect()
        return broker
    except Exception as exc:
        logger.error(f"Falling back to MockBroker: {exc}")
        broker = MockBroker()
        broker.connect()
        return broker
