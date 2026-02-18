"""
Mean Reversion Strategy - Simplified for Live Trading
Buys when price is below mean, sells when above
"""
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Simple mean reversion strategy using Bollinger Bands and RSI"""

    def __init__(self, symbols: List[str], lookback_period: int = 20, bb_std: float = 2.0):
        super().__init__(symbols)
        self.lookback_period = lookback_period
        self.bb_std = bb_std
        self.price_history = []
        self.rsi_period = 14

    def calculate_bollinger_bands(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.lookback_period:
            return {"upper": 0, "middle": 0, "lower": 0}

        sma = np.mean(prices[-self.lookback_period:])
        std = np.std(prices[-self.lookback_period:])

        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)

        return {"upper": upper, "middle": sma, "lower": lower}

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

        BUY: Price below lower Bollinger Band AND RSI < 30 (oversold)
        SELL: Price above upper Bollinger Band AND RSI > 70 (overbought)
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
        if len(self.price_history) < self.lookback_period:
            return Signal.HOLD

        prices = np.array(self.price_history)

        # Calculate indicators
        bb = self.calculate_bollinger_bands(prices)
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
            # Oversold - potential BUY
            if current_price < bb["lower"] and rsi < 30:
                self.entry_price = current_price
                self.position = 1
                return Signal.BUY

            # Overbought - potential SELL
            elif current_price > bb["upper"] and rsi > 70:
                self.entry_price = current_price
                self.position = -1
                return Signal.SELL

        # Mean reversion exit
        if self.position != 0:
            # Price reverted to mean - close position
            if abs(current_price - bb["middle"]) < abs(self.entry_price - bb["middle"]):
                if self.position > 0:
                    self.position = 0
                    self.entry_price = 0
                    return Signal.SELL
                else:
                    self.position = 0
                    self.entry_price = 0
                    return Signal.BUY

        return Signal.HOLD
