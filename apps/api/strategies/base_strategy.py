"""
Base Strategy Class
All trading strategies inherit from this
"""
from abc import ABC, abstractmethod
from typing import Dict, List
from enum import Enum


class Signal(Enum):
    """Trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BaseStrategy(ABC):
    """Base class for all trading strategies"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.position = 0  # Current position size
        self.entry_price = 0.0
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.15  # 15% take profit

    @abstractmethod
    def generate_signal(self, market_data: Dict) -> Signal:
        """
        Generate a trading signal based on market data

        Args:
            market_data: Dictionary with symbol -> price data

        Returns:
            Signal: BUY, SELL, or HOLD
        """
        pass

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if self.position > 0 and self.entry_price > 0:
            loss_pct = (current_price - self.entry_price) / self.entry_price
            if loss_pct <= -self.stop_loss:
                return True
        return False

    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is hit"""
        if self.position > 0 and self.entry_price > 0:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= self.take_profit:
                return True
        return False
