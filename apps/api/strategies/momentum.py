"""
Momentum Strategy - Trend Following
Buys when momentum is strong upward, sells when downward
"""
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using moving average crossover and momentum indicators"""

    def __init__(self, symbols: List[str], fast_period: int = 10, slow_period: int = 30):
        super().__init__(symbols)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = []

    def calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return 0.0
        return np.mean(prices[-period:])

    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate momentum (rate of change)"""
        if len(prices) < period:
            return 0.0
        return (prices[-1] - prices[-period]) / prices[-period] * 100

    def generate_signal(self, market_data: Dict) -> Signal:
        """
        Generate trading signal

        BUY: Fast MA crosses above Slow MA AND positive momentum
        SELL: Fast MA crosses below Slow MA OR negative momentum
        HOLD: Otherwise
        """
        symbol = self.symbols[0]
        if symbol not in market_data:
            return Signal.HOLD

        current_price = market_data[symbol]["price"]

        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]

        # Need enough history
        if len(self.price_history) < self.slow_period:
            return Signal.HOLD

        prices = np.array(self.price_history)

        # Calculate moving averages
        fast_ma = self.calculate_sma(prices, self.fast_period)
        slow_ma = self.calculate_sma(prices, self.slow_period)

        # Calculate momentum
        momentum = self.calculate_momentum(prices, 10)

        # Check for stop loss/take profit on existing position
        if self.position > 0:
            if self.check_stop_loss(current_price):
                self.position = 0
                self.entry_price = 0
                return Signal.SELL
            if self.check_take_profit(current_price):
                self.position = 0
                self.entry_price = 0
                return Signal.SELL

        # Generate new signals if not in position
        if self.position == 0:
            # Bullish crossover + positive momentum
            if fast_ma > slow_ma and momentum > 2:
                self.entry_price = current_price
                self.position = 1
                return Signal.BUY

            # Bearish crossover + negative momentum
            elif fast_ma < slow_ma and momentum < -2:
                self.entry_price = current_price
                self.position = -1
                return Signal.SELL

        # Exit conditions for existing position
        if self.position > 0:
            # Exit long if momentum turns negative
            if momentum < -1:
                self.position = 0
                self.entry_price = 0
                return Signal.SELL

        elif self.position < 0:
            # Exit short if momentum turns positive
            if momentum > 1:
                self.position = 0
                self.entry_price = 0
                return Signal.BUY

        return Signal.HOLD
