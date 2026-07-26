"""
Reinforcement Learning Agent Strategy
Uses trained DQN model for trading decisions
"""
import numpy as np
from typing import Dict, List
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from strategies.base_strategy import BaseStrategy, Signal
from ml.model_server import get_model_server, FeatureExtractor

logger = logging.getLogger(__name__)


class RLAgentStrategy(BaseStrategy):
    """
    Reinforcement Learning strategy using DQN

    Actions:
    - 0: HOLD (do nothing)
    - 1: BUY
    - 2: SELL
    """

    def __init__(self, symbols: List[str]):
        super().__init__(symbols)
        self.price_history = []
        self.volume_history = []
        self.model_server = get_model_server()
        self.feature_extractor = FeatureExtractor()

        # Track state for RL
        self.last_action = 0  # Start with HOLD
        self.action_history = []

        logger.info("RL Agent Strategy initialized")

    def get_state(self, market_data: Dict) -> np.ndarray:
        """
        Get current state for RL agent

        State includes:
        - Market features (price, volume, indicators)
        - Position state (in position, P&L)
        - Recent action history
        """
        symbol = self.symbols[0]
        current_price = market_data[symbol]["price"]

        # Extract market features
        market_features = self.feature_extractor.extract_features(
            self.price_history,
            self.volume_history
        )

        # Position features
        if self.position != 0 and self.entry_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            in_position = 1.0
            position_type = float(self.position)  # 1 for long, -1 for short
        else:
            pnl_pct = 0.0
            in_position = 0.0
            position_type = 0.0

        position_features = np.array([in_position, position_type, pnl_pct])

        # Action history features (last 3 actions)
        recent_actions = self.action_history[-3:] if len(self.action_history) >= 3 else [0, 0, 0]
        action_features = np.array(recent_actions, dtype=np.float32)

        # Combine all features
        state = np.concatenate([market_features, position_features, action_features])

        return state

    def action_to_signal(self, action: int) -> Signal:
        """Convert RL action to trading signal"""
        if action == 1:
            return Signal.BUY
        elif action == 2:
            return Signal.SELL
        else:
            return Signal.HOLD

    def generate_signal(self, market_data: Dict) -> Signal:
        """
        Generate trading signal using RL agent

        Process:
        1. Get current state (market + position features)
        2. Query DQN model for best action
        3. Convert action to signal
        4. Apply risk management
        """
        symbol = self.symbols[0]
        if symbol not in market_data:
            return Signal.HOLD

        current_price = market_data[symbol]["price"]
        current_volume = market_data[symbol].get("volume", 0)

        # Update history
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)

        # Keep only recent history
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]
            self.volume_history = self.volume_history[-200:]

        # Need minimum history
        if len(self.price_history) < 20:
            return Signal.HOLD

        # Check for stop loss/take profit on existing position
        if self.position > 0:
            if self.check_stop_loss(current_price):
                logger.info(f"RL Agent: Stop loss triggered at ${current_price:.2f}")
                self.position = 0
                self.entry_price = 0
                self.action_history.append(2)  # SELL action
                return Signal.SELL
            if self.check_take_profit(current_price):
                logger.info(f"RL Agent: Take profit triggered at ${current_price:.2f}")
                self.position = 0
                self.entry_price = 0
                self.action_history.append(2)  # SELL action
                return Signal.SELL

        # Get state
        try:
            state = self.get_state(market_data)
            state = state.reshape(1, -1)  # Reshape for model input
        except Exception as e:
            logger.error(f"Error getting RL state: {e}")
            return Signal.HOLD

        # Get action from DQN model
        try:
            # Try to get prediction from DQN model
            action_values = self.model_server.predict("dqn_model", state)

            if action_values is None:
                # Fallback to simple policy if model not available
                logger.warning("DQN model not available, using fallback policy")
                action = self._fallback_policy(state)
            else:
                # Choose action with highest Q-value
                if isinstance(action_values, (list, np.ndarray)) and len(action_values) > 1:
                    action = int(np.argmax(action_values))
                else:
                    # Single value prediction - interpret as action probability
                    action = 1 if action_values > 0.5 else 0

        except Exception as e:
            logger.error(f"Error getting RL action: {e}")
            action = 0  # Default to HOLD

        # Record action
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

        # Convert action to signal
        signal = self.action_to_signal(action)

        # Apply position management
        if signal == Signal.BUY and self.position == 0:
            logger.info(f"RL Agent: BUY action at ${current_price:.2f}")
            self.entry_price = current_price
            self.position = 1
            return Signal.BUY

        elif signal == Signal.SELL:
            if self.position > 0:
                logger.info(f"RL Agent: SELL action (close long) at ${current_price:.2f}")
                self.position = 0
                self.entry_price = 0
                return Signal.SELL
            elif self.position == 0:
                # Open short position (if enabled)
                logger.info(f"RL Agent: SELL action (open short) at ${current_price:.2f}")
                self.entry_price = current_price
                self.position = -1
                return Signal.SELL

        elif signal == Signal.BUY and self.position < 0:
            # Close short position
            logger.info(f"RL Agent: BUY action (close short) at ${current_price:.2f}")
            self.position = 0
            self.entry_price = 0
            return Signal.BUY

        return Signal.HOLD

    def _fallback_policy(self, state: np.ndarray) -> int:
        """
        Simple fallback policy when DQN model is not available

        Uses basic rules based on state features
        """
        try:
            # Extract key features
            # State format: [market_features..., in_position, position_type, pnl_pct, recent_actions...]

            in_position = state[0, -6] if state.shape[1] > 6 else 0
            pnl_pct = state[0, -4] if state.shape[1] > 4 else 0

            # Simple rules:
            # - If in position with good profit, hold
            # - If in position with loss > 3%, sell
            # - If not in position, look at market features

            if in_position > 0:
                if pnl_pct < -0.03:
                    return 2  # SELL (stop loss)
                elif pnl_pct > 0.10:
                    return 2  # SELL (take profit)
                else:
                    return 0  # HOLD
            else:
                # Use simple technical signals
                # Extract RSI and momentum from market features
                rsi = state[0, 15] if state.shape[1] > 15 else 0.5
                momentum = state[0, 16] if state.shape[1] > 16 else 0

                if rsi < 0.3 and momentum > 0:
                    return 1  # BUY
                elif rsi > 0.7 or momentum < -0.02:
                    return 2  # SELL
                else:
                    return 0  # HOLD

        except Exception as e:
            logger.error(f"Error in fallback policy: {e}")
            return 0  # HOLD on error
