"""
RSI Strategy - Overbought/Oversold
Trades based on RSI levels
"""
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, Signal


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy"""

    def __init__(self, symbols: List[str], rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(symbols)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.price_history = []

    def calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI"""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(prices[-self.rsi_period-1:])
        gains = deltas.copy()
        losses = deltas.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signal(self, market_data: Dict) -> Signal:
        """
        Generate trading signal

        BUY: RSI < oversold (default 30)
        SELL: RSI > overbought (default 70)
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
        if len(self.price_history) < self.rsi_period + 1:
            return Signal.HOLD

        prices = np.array(self.price_history)

        # Calculate RSI
        rsi = self.calculate_rsi(prices)

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
            # Oversold - BUY
            if rsi < self.oversold:
                self.entry_price = current_price
                self.position = 1
                return Signal.BUY

            # Overbought - SELL
            elif rsi > self.overbought:
                self.entry_price = current_price
                self.position = -1
                return Signal.SELL

        # Exit conditions for existing position
        if self.position > 0:
            # Exit long if RSI reaches neutral or overbought
            if rsi > 50:
                self.position = 0
                self.entry_price = 0
                return Signal.SELL

        elif self.position < 0:
            # Exit short if RSI reaches neutral or oversold
            if rsi < 50:
                self.position = 0
                self.entry_price = 0
                return Signal.BUY

        return Signal.HOLD
