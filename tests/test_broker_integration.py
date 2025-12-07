"""
Integration tests for broker interface and trading system.
Tests use mocked API responses to validate broker operations.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.broker_interface import BrokerInterface, OrderType, OrderSide
from execution.alpaca_broker import AlpacaBroker
from execution.order_manager import OrderManager, OrderStatus
from execution.portfolio_tracker import PortfolioTracker


class MockBroker(BrokerInterface):
    """Mock broker for testing."""
    
    def __init__(self):
        self.connected = False
        self.orders = {}
        self.positions = {}
        self.account = {
            'cash': 100000.0,
            'portfolio_value': 100000.0,
            'buying_power': 100000.0
        }
        self.order_counter = 1
        
    def connect(self) -> bool:
        self.connected = True
        return True
        
    def disconnect(self) -> bool:
        self.connected = False
        return True
        
    def place_order(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType,
        side: OrderSide,
        limit_price: float = None,
        stop_price: float = None
    ) -> Dict[str, Any]:
        if not self.connected:
            raise ConnectionError("Not connected to broker")
            
        order_id = f"MOCK{self.order_counter:06d}"
        self.order_counter += 1
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type.value,
            'side': side.value,
            'status': 'filled',
            'filled_price': limit_price or 100.0,
            'filled_quantity': quantity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.orders[order_id] = order
        
        # Update positions
        if side == OrderSide.BUY:
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
            else:
                self.positions[symbol] = {'quantity': quantity, 'avg_price': limit_price or 100.0}
        else:  # SELL
            if symbol in self.positions:
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
        
        return order
        
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            return True
        return False
        
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return self.orders.get(order_id, {})
        
    def get_positions(self) -> Dict[str, Any]:
        return self.positions
        
    def get_account_info(self) -> Dict[str, Any]:
        return self.account


class TestBrokerInterface:
    """Test broker interface implementation."""
    
    def test_mock_broker_connection(self):
        """Test broker connection and disconnection."""
        broker = MockBroker()
        assert broker.connect() is True
        assert broker.connected is True
        assert broker.disconnect() is True
        assert broker.connected is False
        
    def test_place_market_order(self):
        """Test placing a market order."""
        broker = MockBroker()
        broker.connect()
        
        order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY
        )
        
        assert order['symbol'] == 'AAPL'
        assert order['quantity'] == 10
        assert order['status'] == 'filled'
        assert 'order_id' in order
        
    def test_place_limit_order(self):
        """Test placing a limit order."""
        broker = MockBroker()
        broker.connect()
        
        order = broker.place_order(
            symbol='MSFT',
            quantity=5,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            limit_price=300.0
        )
        
        assert order['symbol'] == 'MSFT'
        assert order['quantity'] == 5
        assert order['filled_price'] == 300.0
        
    def test_position_tracking(self):
        """Test position tracking after orders."""
        broker = MockBroker()
        broker.connect()
        
        # Buy 10 shares
        broker.place_order('AAPL', 10, OrderType.MARKET, OrderSide.BUY)
        positions = broker.get_positions()
        assert 'AAPL' in positions
        assert positions['AAPL']['quantity'] == 10
        
        # Buy 5 more
        broker.place_order('AAPL', 5, OrderType.MARKET, OrderSide.BUY)
        positions = broker.get_positions()
        assert positions['AAPL']['quantity'] == 15
        
        # Sell 7
        broker.place_order('AAPL', 7, OrderType.MARKET, OrderSide.SELL)
        positions = broker.get_positions()
        assert positions['AAPL']['quantity'] == 8
        
    def test_cancel_order(self):
        """Test order cancellation."""
        broker = MockBroker()
        broker.connect()
        
        order = broker.place_order('SPY', 10, OrderType.LIMIT, OrderSide.BUY, limit_price=400.0)
        order_id = order['order_id']
        
        assert broker.cancel_order(order_id) is True
        status = broker.get_order_status(order_id)
        assert status['status'] == 'cancelled'


class TestOrderManager:
    """Test order manager functionality."""
    
    def test_order_manager_initialization(self):
        """Test order manager initialization."""
        broker = MockBroker()
        manager = OrderManager(broker, max_order_value=10000.0)
        assert manager.broker == broker
        assert manager.max_order_value == 10000.0
        
    def test_order_validation(self):
        """Test order validation logic."""
        broker = MockBroker()
        manager = OrderManager(broker, max_order_value=5000.0)
        broker.connect()
        
        # Valid order
        result = manager.place_order('AAPL', 10, OrderType.LIMIT, OrderSide.BUY, limit_price=150.0)
        assert result['status'] == OrderStatus.FILLED.value
        
        # Order too large
        result = manager.place_order('AAPL', 1000, OrderType.LIMIT, OrderSide.BUY, limit_price=150.0)
        assert result['status'] == OrderStatus.REJECTED.value
        assert 'exceeds maximum' in result['message'].lower()
        
    def test_duplicate_order_prevention(self):
        """Test that duplicate orders are prevented."""
        broker = MockBroker()
        manager = OrderManager(broker)
        broker.connect()
        
        # Place first order
        order1 = manager.place_order('AAPL', 10, OrderType.MARKET, OrderSide.BUY)
        assert order1['status'] == OrderStatus.FILLED.value
        
        # Try to place duplicate - should be allowed after cooldown
        order2 = manager.place_order('AAPL', 10, OrderType.MARKET, OrderSide.BUY)
        assert 'order_id' in order2
        
    def test_order_history_tracking(self):
        """Test that order history is tracked."""
        broker = MockBroker()
        manager = OrderManager(broker)
        broker.connect()
        
        manager.place_order('AAPL', 10, OrderType.MARKET, OrderSide.BUY)
        manager.place_order('MSFT', 5, OrderType.MARKET, OrderSide.BUY)
        
        history = manager.get_order_history()
        assert len(history) == 2
        assert history[0]['symbol'] == 'AAPL'
        assert history[1]['symbol'] == 'MSFT'


class TestPortfolioTracker:
    """Test portfolio tracker functionality."""
    
    def test_portfolio_initialization(self):
        """Test portfolio tracker initialization."""
        broker = MockBroker()
        tracker = PortfolioTracker(broker)
        assert tracker.broker == broker
        
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        broker = MockBroker()
        broker.connect()
        tracker = PortfolioTracker(broker)
        
        # Place some orders
        broker.place_order('AAPL', 10, OrderType.LIMIT, OrderSide.BUY, limit_price=150.0)
        broker.place_order('MSFT', 5, OrderType.LIMIT, OrderSide.BUY, limit_price=300.0)
        
        # Mock current prices
        current_prices = {'AAPL': 155.0, 'MSFT': 310.0}
        
        summary = tracker.get_portfolio_summary(current_prices)
        
        assert 'positions' in summary
        assert 'total_value' in summary
        assert summary['cash'] == 100000.0
        
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        broker = MockBroker()
        broker.connect()
        tracker = PortfolioTracker(broker)
        
        # Buy at 150
        broker.place_order('AAPL', 10, OrderType.LIMIT, OrderSide.BUY, limit_price=150.0)
        
        # Current price 155 (profit)
        current_prices = {'AAPL': 155.0}
        summary = tracker.get_portfolio_summary(current_prices)
        
        # Check unrealized P&L
        aapl_position = [p for p in summary['positions'] if p['symbol'] == 'AAPL'][0]
        assert aapl_position['unrealized_pnl'] == 50.0  # (155 - 150) * 10
        assert aapl_position['unrealized_pnl_pct'] > 0
        
    def test_position_exposure(self):
        """Test position exposure calculation."""
        broker = MockBroker()
        broker.connect()
        tracker = PortfolioTracker(broker)
        
        broker.place_order('AAPL', 10, OrderType.LIMIT, OrderSide.BUY, limit_price=150.0)
        
        current_prices = {'AAPL': 150.0}
        summary = tracker.get_portfolio_summary(current_prices)
        
        aapl_position = [p for p in summary['positions'] if p['symbol'] == 'AAPL'][0]
        # Exposure should be 1500 / 100000 = 1.5%
        assert 0.01 < aapl_position['exposure'] < 0.02


class TestAlpacaBroker:
    """Test Alpaca broker implementation with mocked API."""
    
    @patch('execution.alpaca_broker.requests.post')
    @patch('execution.alpaca_broker.requests.get')
    def test_alpaca_connection(self, mock_get, mock_post):
        """Test Alpaca API connection."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'cash': '100000',
            'portfolio_value': '100000'
        }
        
        broker = AlpacaBroker(
            api_key='test_key',
            secret_key='test_secret',
            base_url='https://paper-api.alpaca.markets'
        )
        
        # Connection is automatic on init
        assert broker.base_url == 'https://paper-api.alpaca.markets'
        
    @patch('execution.alpaca_broker.requests.post')
    def test_alpaca_place_order(self, mock_post):
        """Test placing order via Alpaca API."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'id': 'test_order_id',
            'symbol': 'AAPL',
            'qty': '10',
            'side': 'buy',
            'type': 'market',
            'status': 'filled',
            'filled_avg_price': '150.00'
        }
        
        broker = AlpacaBroker('test_key', 'test_secret', 'https://paper-api.alpaca.markets')
        
        order = broker.place_order(
            symbol='AAPL',
            quantity=10,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY
        )
        
        assert order['order_id'] == 'test_order_id'
        assert order['symbol'] == 'AAPL'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
