"""
Alpaca broker integration for paper trading.
"""
import os
import requests
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class AlpacaBroker:
    """
    Alpaca broker interface for paper trading.
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca broker connection.
        
        Args:
            paper_trading: Whether to use paper trading (True) or live trading (False)
        """
        self.paper_trading = paper_trading
        
        # API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not found in environment variables")
            self.api_key = "YOUR_API_KEY_HERE"
            self.secret_key = "YOUR_SECRET_KEY_HERE"
        
        # Base URLs
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        
        # Headers for API requests
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info(f"Alpaca broker initialized (Paper Trading: {paper_trading})")
    
    def check_connection(self) -> bool:
        """
        Test the connection to Alpaca API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/v2/account")
            if response.status_code == 200:
                account_data = response.json()
                logger.info(f"✅ Connected to Alpaca. Account: {account_data.get('account_number', 'N/A')}")
                return True
            else:
                logger.error(f"❌ Connection failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return False
    
    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        try:
            response = self.session.get(f"{self.base_url}/v2/account")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get account info: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.
        
        Returns:
            Current portfolio value in USD
        """
        account = self.get_account_info()
        return float(account.get('portfolio_value', 0))
    
    def get_buying_power(self) -> float:
        """
        Get available buying power.
        
        Returns:
            Available buying power in USD
        """
        account = self.get_account_info()
        return float(account.get('buying_power', 0))
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            response = self.session.get(f"{self.base_url}/v2/positions")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get positions: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position dictionary or None if not found
        """
        try:
            response = self.session.get(f"{self.base_url}/v2/positions/{symbol}")
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None
    
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market",
                   time_in_force: str = "day") -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity of shares
            side: 'buy' or 'sell'
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force ('day', 'gtc', etc.)
            
        Returns:
            Order response dictionary or None if failed
        """
        try:
            order_data = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            response = self.session.post(f"{self.base_url}/v2/orders", json=order_data)
            
            if response.status_code == 201:
                order = response.json()
                logger.info(f"✅ Order placed: {side.upper()} {qty} {symbol}")
                return order
            else:
                logger.error(f"❌ Order failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def buy_shares(self, symbol: str, qty: int) -> Optional[Dict]:
        """
        Buy shares using market order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares to buy
            
        Returns:
            Order response or None
        """
        return self.place_order(symbol, qty, 'buy')
    
    def sell_shares(self, symbol: str, qty: int) -> Optional[Dict]:
        """
        Sell shares using market order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares to sell
            
        Returns:
            Order response or None
        """
        return self.place_order(symbol, qty, 'sell')
    
    def execute_signal(self, symbol: str, signal: str, confidence: float, 
                      position_size: float = 0.1) -> bool:
        """
        Execute a trading signal.
        
        Args:
            symbol: Stock symbol
            signal: 'BUY' or 'SELL'
            confidence: Signal confidence (0-1)
            position_size: Fraction of portfolio to allocate
            
        Returns:
            True if executed successfully, False otherwise
        """
        try:
            # Adjust position size by confidence
            adjusted_size = position_size * confidence
            
            if signal == 'BUY':
                # Calculate how many shares to buy
                buying_power = self.get_buying_power()
                max_investment = buying_power * adjusted_size
                
                # Get current price (simplified - use a reasonable estimate)
                current_price = 100  # Placeholder - in real implementation, get from API
                shares_to_buy = int(max_investment / current_price)
                
                if shares_to_buy > 0:
                    order = self.buy_shares(symbol, shares_to_buy)
                    return order is not None
                else:
                    logger.warning(f"Insufficient buying power for {symbol}")
                    return False
                    
            elif signal == 'SELL':
                # Sell partial position based on confidence
                position = self.get_position(symbol)
                if position and int(position['qty']) > 0:
                    current_shares = int(position['qty'])
                    shares_to_sell = max(1, int(current_shares * confidence))
                    order = self.sell_shares(symbol, shares_to_sell)
                    return order is not None
                else:
                    logger.warning(f"No position to sell for {symbol}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {str(e)}")
            return False

class MockBroker:
    """
    Mock broker for testing when Alpaca credentials are not available.
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.orders = []
        logger.info("Mock broker initialized for testing")
    
    def check_connection(self) -> bool:
        logger.info("✅ Mock broker connection successful")
        return True
    
    def get_account_info(self) -> Dict:
        return {
            'account_number': 'MOCK123',
            'portfolio_value': self.cash + sum(pos['qty'] * 100 for pos in self.positions.values()),
            'buying_power': self.cash,
            'cash': self.cash
        }
    
    def get_portfolio_value(self) -> float:
        return self.get_account_info()['portfolio_value']
    
    def get_buying_power(self) -> float:
        return self.cash
    
    def execute_signal(self, symbol: str, signal: str, confidence: float, 
                      position_size: float = 0.1) -> bool:
        # Mock implementation
        logger.info(f"MOCK: {signal} signal for {symbol} (confidence: {confidence:.2f})")
        return True

def create_broker(paper_trading: bool = True, mock: bool = False) -> Union[AlpacaBroker, MockBroker]:
    """
    Factory function to create a broker instance.
    
    Args:
        paper_trading: Whether to use paper trading
        mock: Whether to use mock broker for testing
        
    Returns:
        Broker instance
    """
    if mock:
        return MockBroker()
    else:
        return AlpacaBroker(paper_trading=paper_trading)

# Legacy class for backward compatibility
class BrokerInterface:
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.broker = AlpacaBroker(paper_trading=True)
        logger.info("Legacy BrokerInterface created")

    def connect(self):
        return self.broker.check_connection()

if __name__ == "__main__":
    # Test broker functionality
    print("🏦 Testing Alpaca Broker Integration...")
    
    # Try real broker first
    broker = create_broker(paper_trading=True, mock=False)
    
    if not broker.check_connection():
        print("⚠️  Alpaca connection failed, using mock broker")
        broker = create_broker(mock=True)
        broker.check_connection()
    
    # Test basic functionality
    account = broker.get_account_info()
    print(f"Account Info: {account}")
    
    # Test signal execution
    success = broker.execute_signal('AAPL', 'BUY', 0.8, 0.1)
    print(f"Signal execution: {'✅ Success' if success else '❌ Failed'}")
    
    print("🎉 Broker integration test complete!")