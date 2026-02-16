"""
Order manager for trade execution and lifecycle management.

Handles order validation, execution, tracking, and deduplication.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from .broker_interface import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OrderManager:
    """
    Manages order lifecycle, validation, and tracking.
    
    Ensures orders are executed safely with proper validation,
    deduplication, and logging.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        trades_log_path: str = "logs/trades.log",
        max_order_value: float = 10000.0,
        max_position_size: float = 0.2,
    ):
        """
        Initialize order manager.
        
        Args:
            broker: Broker interface instance
            trades_log_path: Path to trades log file
            max_order_value: Maximum value per order in USD
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.broker = broker
        self.trades_log_path = trades_log_path
        self.max_order_value = max_order_value
        self.max_position_size = max_position_size

        # Track recent orders for deduplication
        self.recent_orders: Set[str] = set()
        self.order_history: List[Dict] = []

        # Ensure logs directory exists
        os.makedirs(os.path.dirname(trades_log_path), exist_ok=True)

        logger.info(f"OrderManager initialized (max_order: ${max_order_value:,.2f})")

    def place_order(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType,
        side: OrderSide,
        limit_price: float = None,
        stop_price: float = None,
    ) -> Dict:
        """
        Place a single order with minimal validation and normalized output.

        This is a lightweight helper for testing/integration. It performs a basic
        order value check (against ``max_order_value``) then forwards to
        ``broker.place_order``. It does not run the broader validation/deduping
        done by :meth:`execute_trade`.

        Returns a dict with normalized fields: ``status`` (str), ``symbol``,
        ``order_id`` (if provided), and ``timestamp``. On validation/broker
        failure it returns a rejected status with a message.
        """
        estimated_price = limit_price or self.broker.get_current_price(symbol) or 0
        order_value = quantity * estimated_price

        if order_value > self.max_order_value:
            return {
                "status": OrderStatus.REJECTED.value,
                "message": "Order value exceeds maximum allowed"
            }

        try:
            broker_order = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                side=side,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        except TypeError:
            broker_order = self.broker.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type=order_type,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                stop_price=stop_price,
            )

        if not broker_order:
            return {
                "status": OrderStatus.REJECTED.value,
                "message": "Broker order failed"
            }

        # Ensure status field exists as string for tests
        status = broker_order.get("status") if isinstance(broker_order, dict) else getattr(broker_order, "status", None)
        if isinstance(broker_order, dict):
            order_record = dict(broker_order)
        elif hasattr(broker_order, "__dict__"):
            order_record = dict(broker_order.__dict__)
        else:
            order_record = {}
        if status is None:
            order_record["status"] = OrderStatus.NEW.value
        elif hasattr(status, "value"):
            order_record["status"] = status.value
        else:
            order_record["status"] = status
        order_record.setdefault("symbol", symbol)
        order_record.setdefault("order_id", broker_order.get("order_id") if isinstance(broker_order, dict) else "")
        order_record.setdefault("timestamp", datetime.now().isoformat())

        self.order_history.append(order_record)
        try:
            with open(self.trades_log_path, "a") as f:
                f.write(json.dumps(order_record) + "\n")
        except Exception as exc:  # pragma: no cover - logging only
            logger.exception(
                "Failed to write order record to trades log file '%s'. Order record: %s",
                self.trades_log_path,
                order_record,
            )
        return order_record

    def execute_trade(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        position_size: float = 0.1,
        order_type: OrderType = OrderType.MARKET,
    ) -> Optional[Order]:
        """
        Execute a trading signal with validation.
        
        Args:
            symbol: Stock ticker symbol
            signal: 'BUY' or 'SELL'
            confidence: Signal confidence (0-1)
            position_size: Target position size as fraction of portfolio
            order_type: Order type (market, limit, etc.)
            
        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Validate signal
            if signal not in ["BUY", "SELL"]:
                logger.error(f"Invalid signal: {signal}")
                return None

            if not 0 <= confidence <= 1:
                logger.error(f"Invalid confidence: {confidence}")
                return None

            # Check for duplicate orders
            order_key = f"{symbol}_{signal}_{datetime.now().strftime('%Y%m%d%H%M')}"
            if order_key in self.recent_orders:
                logger.warning(f"Duplicate order detected: {order_key}")
                return None

            self.recent_orders.add(order_key)

            # Clean old order keys (older than 1 hour)
            if len(self.recent_orders) > 100:
                self.recent_orders.clear()

            # Get account info for validation
            account = self.broker.get_account_info()

            if signal == "BUY":
                return self._execute_buy(
                    symbol, confidence, position_size, order_type, account
                )
            elif signal == "SELL":
                return self._execute_sell(symbol, confidence, order_type)

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return None

    def _execute_buy(
        self,
        symbol: str,
        confidence: float,
        position_size: float,
        order_type: OrderType,
        account,
    ) -> Optional[Order]:
        """Execute buy order with validation."""
        # Calculate position size adjusted by confidence
        adjusted_size = position_size * confidence

        # Calculate maximum investment
        max_investment = min(
            account.buying_power * adjusted_size,
            account.portfolio_value * self.max_position_size,
            self.max_order_value,
        )

        if max_investment < 1.0:
            logger.warning(f"Insufficient funds for {symbol}: ${max_investment:.2f}")
            return None

        # Get current price
        current_price = self.broker.get_current_price(symbol)
        if not current_price or current_price <= 0:
            logger.error(f"Could not get valid price for {symbol}")
            return None

        # Calculate quantity
        qty = int(max_investment / current_price)

        if qty < 1:
            logger.warning(
                f"Insufficient funds to buy 1 share of {symbol} @ ${current_price:.2f}"
            )
            return None

        # Validate order value
        order_value = qty * current_price
        if order_value > self.max_order_value:
            qty = int(self.max_order_value / current_price)
            logger.warning(f"Order value capped: buying {qty} shares instead")

        # Place order
        logger.info(f"Placing BUY order: {qty} {symbol} @ ~${current_price:.2f}")
        order = self.broker.place_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            order_type=order_type,
            time_in_force=TimeInForce.DAY,
        )

        if order:
            self._log_trade(order, confidence)

        return order

    def _execute_sell(
        self, symbol: str, confidence: float, order_type: OrderType
    ) -> Optional[Order]:
        """Execute sell order with validation."""
        # Get current position
        position = self.broker.get_position(symbol)

        if not position or position.qty <= 0:
            logger.warning(f"No position to sell for {symbol}")
            return None

        # Calculate quantity to sell based on confidence
        qty_to_sell = max(1, int(position.qty * confidence))

        logger.info(
            f"Placing SELL order: {qty_to_sell} {symbol} @ ~${position.current_price:.2f}"
        )
        order = self.broker.place_order(
            symbol=symbol,
            qty=qty_to_sell,
            side=OrderSide.SELL,
            order_type=order_type,
            time_in_force=TimeInForce.DAY,
        )

        if order:
            self._log_trade(order, confidence)

        return order

    def _log_trade(self, order: Order, confidence: float) -> None:
        """Log trade to file."""
        try:
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "order_type": order.order_type.value,
                "status": order.status.value,
                "confidence": confidence,
                "filled_qty": order.filled_qty,
                "filled_avg_price": order.filled_avg_price,
            }

            self.order_history.append(trade_record)

            # Append to log file
            with open(self.trades_log_path, "a") as f:
                f.write(json.dumps(trade_record) + "\n")

            logger.info(f"Trade logged: {order.side.value} {order.qty} {order.symbol}")

        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if canceled successfully
        """
        success = self.broker.cancel_order(order_id)
        if success:
            logger.info(f"Order {order_id} canceled")
        return success

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderStatus or None if not found
        """
        order = self.broker.get_order(order_id)
        return order.status if order else None

    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders.
        
        Returns:
            List of open Order objects
        """
        return self.broker.get_orders(status=OrderStatus.NEW)

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders canceled
        """
        open_orders = self.get_open_orders()
        canceled = 0

        for order in open_orders:
            if self.cancel_order(order.order_id):
                canceled += 1

        logger.info(f"Canceled {canceled} open orders")
        return canceled

    def get_order_history(self, days: int = 7) -> List[Dict]:
        """
        Get order history from log file.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of trade records
        """
        if self.order_history:
            return list(self.order_history)

        if not os.path.exists(self.trades_log_path):
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        history = []

        try:
            with open(self.trades_log_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_date = datetime.fromisoformat(record["timestamp"])
                        if record_date >= cutoff_date:
                            history.append(record)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

            return history

        except Exception as e:
            logger.error(f"Error reading order history: {str(e)}")
            return []

    def get_trade_statistics(self, days: int = 7) -> Dict:
        """
        Get trading statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trade statistics
        """
        history = self.get_order_history(days)

        if not history:
            return {
                "total_trades": 0,
                "buy_orders": 0,
                "sell_orders": 0,
                "filled_orders": 0,
                "rejected_orders": 0,
            }

        stats = {
            "total_trades": len(history),
            "buy_orders": sum(1 for t in history if t["side"] == "buy"),
            "sell_orders": sum(1 for t in history if t["side"] == "sell"),
            "filled_orders": sum(1 for t in history if t["status"] == "filled"),
            "rejected_orders": sum(1 for t in history if t["status"] == "rejected"),
            "total_shares_bought": sum(
                t["qty"] for t in history if t["side"] == "buy" and t["status"] == "filled"
            ),
            "total_shares_sold": sum(
                t["qty"] for t in history if t["side"] == "sell" and t["status"] == "filled"
            ),
            "avg_confidence": sum(t["confidence"] for t in history) / len(history),
        }

        return stats


if __name__ == "__main__":
    from execution.broker_interface import MockBroker

    print("🎯 Testing OrderManager...")

    # Use mock broker for testing
    broker = MockBroker(initial_capital=100000)
    broker.connect()

    order_manager = OrderManager(broker, max_order_value=5000, max_position_size=0.1)

    # Test buy order
    print("\n📈 Testing BUY order...")
    order = order_manager.execute_trade(
        symbol="AAPL", signal="BUY", confidence=0.8, position_size=0.05
    )
    if order:
        print(f"  ✅ Order placed: {order.order_id}")

    # Test duplicate detection
    print("\n🔄 Testing duplicate detection...")
    duplicate = order_manager.execute_trade(
        symbol="AAPL", signal="BUY", confidence=0.8, position_size=0.05
    )
    if not duplicate:
        print("  ✅ Duplicate detected and prevented")

    # Test sell order
    print("\n📉 Testing SELL order...")
    sell_order = order_manager.execute_trade(
        symbol="AAPL", signal="SELL", confidence=0.6
    )
    if sell_order:
        print(f"  ✅ Sell order placed: {sell_order.order_id}")

    # Test statistics
    print("\n📊 Trading Statistics:")
    stats = order_manager.get_trade_statistics(days=1)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ OrderManager test complete!")
