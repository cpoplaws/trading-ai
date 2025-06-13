"""
Enhanced Order Manager with risk management integration.
Part of Phase 2: Trading System completion.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str  # 'market', 'limit', 'stop'
    price: Optional[float]
    time_in_force: str  # 'day', 'gtc', 'ioc'
    status: OrderStatus
    created_at: str
    filled_at: Optional[str] = None
    filled_price: Optional[float] = None
    filled_qty: int = 0
    broker_order_id: Optional[str] = None
    strategy: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['status'] = self.status.value
        return result

class EnhancedOrderManager:
    """
    Enhanced order management system with risk integration and tracking.
    """
    
    def __init__(self, broker, risk_manager, portfolio_tracker, save_path: str = './logs/'):
        """
        Initialize enhanced order manager.
        
        Args:
            broker: Broker interface instance
            risk_manager: Risk manager instance
            portfolio_tracker: Portfolio tracker instance
            save_path: Directory to save order logs
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.portfolio_tracker = portfolio_tracker
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        self.orders = {}  # order_id -> Order
        self.order_history = []
        self.next_order_id = 1
        
        logger.info("Enhanced order manager initialized")
    
    def create_market_order(self, symbol: str, side: str, quantity: int,
                          strategy: str = "", confidence: float = 1.0) -> Optional[str]:
        """
        Create a market order with risk checking.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            strategy: Strategy that generated the signal
            confidence: Signal confidence (0-1)
            
        Returns:
            Order ID if successful, None if rejected
        """
        try:
            # Calculate position size as fraction of portfolio
            portfolio_value = self.portfolio_tracker.calculate_portfolio_metrics().total_value
            estimated_price = 100  # In real implementation, get current price
            position_value = quantity * estimated_price
            position_size = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Risk check
            risk_check = self.risk_manager.check_trade_risk(
                symbol, side, position_size, confidence, estimated_price
            )
            
            if not risk_check.approved:
                logger.warning(f"Order rejected by risk manager: {risk_check.reason}")
                return None
            
            # Adjust quantity based on risk recommendation
            if risk_check.recommended_size < position_size:
                adjusted_quantity = int(quantity * (risk_check.recommended_size / position_size))
                if adjusted_quantity < quantity:
                    logger.info(f"Quantity adjusted by risk manager: {quantity} -> {adjusted_quantity}")
                    quantity = adjusted_quantity
            
            # Create order
            order_id = self._generate_order_id()
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",
                price=None,
                time_in_force="day",
                status=OrderStatus.PENDING,
                created_at=datetime.now().isoformat(),
                strategy=strategy,
                confidence=confidence
            )
            
            # Submit to broker
            if self._submit_order(order):
                self.orders[order_id] = order
                self._save_order(order)
                logger.info(f"Market order created: {order_id} - {side.upper()} {quantity} {symbol}")
                return order_id
            else:
                logger.error(f"Failed to submit order to broker")
                return None
                
        except Exception as e:
            logger.error(f"Error creating market order: {str(e)}")
            return None
    
    def create_limit_order(self, symbol: str, side: str, quantity: int, limit_price: float,
                         time_in_force: str = "day", strategy: str = "", 
                         confidence: float = 1.0) -> Optional[str]:
        """
        Create a limit order with risk checking.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            limit_price: Limit price
            time_in_force: Time in force
            strategy: Strategy that generated the signal
            confidence: Signal confidence (0-1)
            
        Returns:
            Order ID if successful, None if rejected
        """
        try:
            # Calculate position size
            portfolio_value = self.portfolio_tracker.calculate_portfolio_metrics().total_value
            position_value = quantity * limit_price
            position_size = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Risk check
            risk_check = self.risk_manager.check_trade_risk(
                symbol, side, position_size, confidence, limit_price
            )
            
            if not risk_check.approved:
                logger.warning(f"Limit order rejected by risk manager: {risk_check.reason}")
                return None
            
            # Create order
            order_id = self._generate_order_id()
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="limit",
                price=limit_price,
                time_in_force=time_in_force,
                status=OrderStatus.PENDING,
                created_at=datetime.now().isoformat(),
                strategy=strategy,
                confidence=confidence
            )
            
            # Submit to broker
            if self._submit_order(order):
                self.orders[order_id] = order
                self._save_order(order)
                logger.info(f"Limit order created: {order_id} - {side.upper()} {quantity} {symbol} @ ${limit_price}")
                return order_id
            else:
                logger.error(f"Failed to submit limit order to broker")
                return None
                
        except Exception as e:
            logger.error(f"Error creating limit order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled
        """
        try:
            if order_id not in self.orders:
                logger.error(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
                logger.warning(f"Cannot cancel order {order_id} - status: {order.status}")
                return False
            
            # Cancel with broker
            success = self._cancel_broker_order(order)
            
            if success:
                order.status = OrderStatus.CANCELLED
                self._save_order(order)
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order with broker: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def update_order_status(self, order_id: str) -> bool:
        """
        Update order status from broker.
        
        Args:
            order_id: Order ID to update
            
        Returns:
            True if successfully updated
        """
        try:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            
            # Get status from broker
            broker_status = self._get_broker_order_status(order)
            
            if broker_status:
                old_status = order.status
                order.status = broker_status['status']
                
                if 'filled_price' in broker_status:
                    order.filled_price = broker_status['filled_price']
                if 'filled_qty' in broker_status:
                    order.filled_qty = broker_status['filled_qty']
                if 'filled_at' in broker_status:
                    order.filled_at = broker_status['filled_at']
                
                # Log trade if filled
                if order.status == OrderStatus.FILLED and old_status != OrderStatus.FILLED:
                    self.portfolio_tracker.log_trade(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.filled_qty,
                        price=order.filled_price,
                        order_id=order_id,
                        strategy=order.strategy
                    )
                
                self._save_order(order)
                return True
                
        except Exception as e:
            logger.error(f"Error updating order status {order_id}: {str(e)}")
            
        return False
    
    def update_all_orders(self) -> None:
        """
        Update status of all active orders.
        """
        active_orders = [
            order_id for order_id, order in self.orders.items()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        ]
        
        for order_id in active_orders:
            self.update_order_status(order_id)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get orders for a specific symbol.
        
        Args:
            symbol: Stock symbol
            status: Optional status filter
            
        Returns:
            List of matching orders
        """
        orders = [order for order in self.orders.values() if order.symbol == symbol]
        
        if status:
            orders = [order for order in orders if order.status == status]
        
        return orders
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders.
        
        Returns:
            List of active orders
        """
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        ]
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        order_id = f"ORD_{self.next_order_id:06d}"
        self.next_order_id += 1
        return order_id
    
    def _submit_order(self, order: Order) -> bool:
        """
        Submit order to broker.
        
        Args:
            order: Order to submit
            
        Returns:
            True if successfully submitted
        """
        try:
            if order.order_type == "market":
                result = self.broker.place_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    order_type="market",
                    time_in_force=order.time_in_force
                )
            elif order.order_type == "limit":
                # Note: broker.place_order might need to be extended for limit orders
                result = self.broker.place_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    order_type="limit",
                    time_in_force=order.time_in_force
                )
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return False
            
            if result:
                order.status = OrderStatus.SUBMITTED
                order.broker_order_id = result.get('id', '')
                return True
            else:
                order.status = OrderStatus.REJECTED
                return False
                
        except Exception as e:
            logger.error(f"Error submitting order to broker: {str(e)}")
            order.status = OrderStatus.REJECTED
            return False
    
    def _cancel_broker_order(self, order: Order) -> bool:
        """
        Cancel order with broker.
        
        Args:
            order: Order to cancel
            
        Returns:
            True if successfully cancelled
        """
        try:
            # Implementation depends on broker API
            # For now, assume success
            logger.info(f"Cancelled order with broker: {order.broker_order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order with broker: {str(e)}")
            return False
    
    def _get_broker_order_status(self, order: Order) -> Optional[Dict]:
        """
        Get order status from broker.
        
        Args:
            order: Order to check
            
        Returns:
            Status dictionary or None
        """
        try:
            # Implementation depends on broker API
            # For mock broker, simulate filled orders
            if hasattr(self.broker, '__class__') and 'Mock' in self.broker.__class__.__name__:
                return {
                    'status': OrderStatus.FILLED,
                    'filled_price': 100.0,  # Mock price
                    'filled_qty': order.quantity,
                    'filled_at': datetime.now().isoformat()
                }
            else:
                # Real broker implementation would go here
                return None
                
        except Exception as e:
            logger.error(f"Error getting order status from broker: {str(e)}")
            return None
    
    def _save_order(self, order: Order) -> None:
        """
        Save order to file.
        
        Args:
            order: Order to save
        """
        try:
            # Add to history if status changed to final state
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                if order.to_dict() not in self.order_history:
                    self.order_history.append(order.to_dict())
            
            # Save all orders to file
            orders_file = os.path.join(self.save_path, 'orders.json')
            all_orders = [order.to_dict() for order in self.orders.values()]
            
            with open(orders_file, 'w') as f:
                json.dump(all_orders, f, indent=2)
            
            # Save order history
            history_file = os.path.join(self.save_path, 'order_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.order_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving order: {str(e)}")
    
    def print_order_status(self) -> None:
        """
        Print formatted order status to console.
        """
        try:
            active_orders = self.get_active_orders()
            
            print("\n" + "="*60)
            print("📋 ORDER STATUS")
            print("="*60)
            
            if not active_orders:
                print("No active orders")
            else:
                print(f"{'ID':<12} {'Symbol':<6} {'Side':<4} {'Qty':<6} {'Type':<6} {'Status':<10}")
                print("-" * 60)
                for order in active_orders:
                    print(f"{order.id:<12} {order.symbol:<6} {order.side:<4} {order.quantity:<6} "
                          f"{order.order_type:<6} {order.status.value:<10}")
            
            # Show recent fills
            recent_fills = [
                order for order in self.order_history[-5:]
                if order['status'] == OrderStatus.FILLED.value
            ]
            
            if recent_fills:
                print(f"\n📈 RECENT FILLS:")
                print("-" * 60)
                for order in recent_fills:
                    print(f"{order['symbol']} {order['side'].upper()} {order['quantity']} @ "
                          f"${order.get('filled_price', 0):.2f}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing order status: {str(e)}")

if __name__ == "__main__":
    # Test enhanced order manager
    from broker_interface import create_broker
    from portfolio_tracker import PortfolioTracker
    from risk_manager import RiskManager
    
    print("📋 Testing Enhanced Order Manager...")
    
    # Create components
    broker = create_broker(mock=True)
    tracker = PortfolioTracker(broker, initial_capital=10000)
    risk_mgr = RiskManager(tracker)
    order_mgr = EnhancedOrderManager(broker, risk_mgr, tracker)
    
    # Test market order
    order_id = order_mgr.create_market_order('AAPL', 'buy', 10, 'test_strategy', 0.8)
    print(f"Market order created: {order_id}")
    
    # Test limit order
    limit_order_id = order_mgr.create_limit_order('MSFT', 'buy', 5, 250.0, 'day', 'test_strategy', 0.9)
    print(f"Limit order created: {limit_order_id}")
    
    # Update orders and print status
    order_mgr.update_all_orders()
    order_mgr.print_order_status()
    
    print("\n🎉 Enhanced order manager test complete!")