"""
Trading Environment for Reinforcement Learning Agents
Custom OpenAI Gym environment for training RL trading agents
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading actions"""
    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6


class TradingEnv(gym.Env):
    """
    Custom Trading Environment following OpenAI Gym interface.

    State space: [price, volume, position, cash, indicators...]
    Action space: Discrete(7) - Hold, Buy/Sell in different sizes
    Reward: Profit + penalties for risk
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position: float = 0.3,
        lookback_window: int = 20
    ):
        """
        Initialize trading environment.

        Args:
            data: Historical price data with OHLCV
            initial_balance: Starting cash
            commission: Commission rate per trade
            slippage: Slippage rate
            max_position: Max % of portfolio per position
            lookback_window: Number of past timesteps to include in state
        """
        super(TradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_position = max_position
        self.lookback_window = lookback_window

        # Calculate additional features
        self._prepare_features()

        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)

        # Observation space: continuous features
        # [lookback prices, indicators, position, cash, unrealized_pnl]
        n_features = len(self.feature_columns) * lookback_window + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.avg_price = 0.0
        self.trades = []
        self.total_reward = 0.0

    def _prepare_features(self):
        """Calculate technical indicators and features."""
        df = self.data.copy()

        # Returns
        df['returns'] = df['close'].pct_change()

        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Fill NaN
        df = df.fillna(method='bfill').fillna(0)

        self.data = df
        self.feature_columns = ['close', 'returns', 'sma_10', 'sma_20', 'rsi', 'volatility', 'volume_ratio']

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.avg_price = 0.0
        self.trades = []
        self.total_reward = 0.0

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Historical features (lookback window)
        start = self.current_step - self.lookback_window
        end = self.current_step

        historical_data = []
        for col in self.feature_columns:
            values = self.data[col].iloc[start:end].values
            historical_data.extend(values)

        # Current position info
        current_price = self.data['close'].iloc[self.current_step]
        unrealized_pnl = (current_price - self.avg_price) * self.position if self.position > 0 else 0

        position_info = [
            self.position / (self.initial_balance / current_price),  # Normalized position
            self.balance / self.initial_balance,  # Normalized balance
            unrealized_pnl / self.initial_balance  # Normalized PnL
        ]

        obs = np.array(historical_data + position_info, dtype=np.float32)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one timestep."""
        current_price = self.data['close'].iloc[self.current_step]

        # Execute action
        reward = self._execute_action(action, current_price)

        # Move to next timestep
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        # Get new observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        # Additional info
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'trades': len(self.trades),
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance
        }

        self.total_reward += reward

        return obs, reward, done, info

    def _execute_action(self, action: int, price: float) -> float:
        """
        Execute trading action and return reward.

        Args:
            action: Action to execute
            price: Current price

        Returns:
            reward: Immediate reward
        """
        action_enum = Action(action)

        # Calculate position sizes (as % of portfolio)
        size_map = {
            Action.HOLD: 0,
            Action.BUY_SMALL: 0.1,
            Action.BUY_MEDIUM: 0.2,
            Action.BUY_LARGE: 0.3,
            Action.SELL_SMALL: 0.1,
            Action.SELL_MEDIUM: 0.2,
            Action.SELL_LARGE: 0.3
        }

        reward = 0.0

        if action_enum == Action.HOLD:
            # Small penalty for holding to encourage action
            reward = -0.0001

        elif action_enum in [Action.BUY_SMALL, Action.BUY_MEDIUM, Action.BUY_LARGE]:
            size_pct = size_map[action_enum]
            max_shares = (self.balance * size_pct) / price

            if self.balance > price:  # Can afford at least 1 share
                # Apply slippage
                execution_price = price * (1 + self.slippage)
                shares = min(max_shares, self.balance / execution_price)
                cost = shares * execution_price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost

                if total_cost <= self.balance:
                    # Update position (weighted average price)
                    if self.position > 0:
                        total_shares = self.position + shares
                        self.avg_price = ((self.avg_price * self.position) + (execution_price * shares)) / total_shares
                        self.position = total_shares
                    else:
                        self.position = shares
                        self.avg_price = execution_price

                    self.balance -= total_cost

                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': shares,
                        'price': execution_price,
                        'cost': total_cost
                    })

                    # Reward for taking action (will be refined by future returns)
                    reward = 0.001
            else:
                # Penalty for invalid action
                reward = -0.01

        elif action_enum in [Action.SELL_SMALL, Action.SELL_MEDIUM, Action.SELL_LARGE]:
            size_pct = size_map[action_enum]
            shares_to_sell = self.position * size_pct

            if self.position > 0:
                # Apply slippage
                execution_price = price * (1 - self.slippage)
                proceeds = shares_to_sell * execution_price
                commission_cost = proceeds * self.commission
                net_proceeds = proceeds - commission_cost

                # Calculate realized P&L
                cost_basis = shares_to_sell * self.avg_price
                realized_pnl = net_proceeds - cost_basis

                self.position -= shares_to_sell
                self.balance += net_proceeds

                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'proceeds': net_proceeds,
                    'pnl': realized_pnl
                })

                # Reward based on profit
                reward = realized_pnl / self.initial_balance * 100  # Scale reward
            else:
                # Penalty for trying to sell with no position
                reward = -0.01

        # Add portfolio value change as additional reward signal
        portfolio_value = self.balance + (self.position * price if self.position > 0 else 0)
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance

        # Sharpe-like reward (returns adjusted for volatility)
        if self.current_step > self.lookback_window + 1:
            recent_returns = self.data['returns'].iloc[self.current_step - 20:self.current_step].values
            volatility = np.std(recent_returns) if len(recent_returns) > 0 else 1.0
            reward += portfolio_return / (volatility + 1e-8) * 0.1

        return reward

    def render(self, mode='human'):
        """Render the environment."""
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)

        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.2f} shares")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Return: {((portfolio_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
        print(f"Trades: {len(self.trades)}")
        print("-" * 50)


class MultiAgentTradingEnv(TradingEnv):
    """
    Multi-agent trading environment where multiple agents can coexist.
    Agents can cooperate or compete.
    """

    def __init__(self, n_agents: int = 3, **kwargs):
        """
        Initialize multi-agent environment.

        Args:
            n_agents: Number of agents
            **kwargs: Arguments for TradingEnv
        """
        super().__init__(**kwargs)
        self.n_agents = n_agents

        # Each agent has its own portfolio
        self.agent_balances = [self.initial_balance] * n_agents
        self.agent_positions = [0.0] * n_agents
        self.agent_avg_prices = [0.0] * n_agents

    def reset(self) -> List[np.ndarray]:
        """Reset for all agents."""
        super().reset()
        self.agent_balances = [self.initial_balance] * self.n_agents
        self.agent_positions = [0.0] * self.n_agents
        self.agent_avg_prices = [0.0] * self.n_agents

        return [self._get_observation() for _ in range(self.n_agents)]

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Execute actions for all agents.

        Args:
            actions: List of actions (one per agent)

        Returns:
            observations: List of observations
            rewards: List of rewards
            done: Episode done flag
            info: Additional info
        """
        current_price = self.data['close'].iloc[self.current_step]

        rewards = []
        for agent_id, action in enumerate(actions):
            # Execute action for this agent
            reward = self._execute_agent_action(agent_id, action, current_price)
            rewards.append(reward)

        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Get observations for all agents
        observations = [self._get_observation() for _ in range(self.n_agents)] if not done else [np.zeros(self.observation_space.shape)] * self.n_agents

        # Info with aggregate stats
        total_value = sum(
            self.agent_balances[i] + (self.agent_positions[i] * current_price if self.agent_positions[i] > 0 else 0)
            for i in range(self.n_agents)
        )

        info = {
            'total_portfolio_value': total_value,
            'agent_balances': self.agent_balances.copy(),
            'agent_positions': self.agent_positions.copy(),
            'swarm_return': (total_value - self.initial_balance * self.n_agents) / (self.initial_balance * self.n_agents)
        }

        return observations, rewards, done, info

    def _execute_agent_action(self, agent_id: int, action: int, price: float) -> float:
        """Execute action for specific agent."""
        # Temporarily set agent's state
        old_balance = self.balance
        old_position = self.position
        old_avg_price = self.avg_price

        self.balance = self.agent_balances[agent_id]
        self.position = self.agent_positions[agent_id]
        self.avg_price = self.agent_avg_prices[agent_id]

        # Execute action
        reward = self._execute_action(action, price)

        # Save agent's new state
        self.agent_balances[agent_id] = self.balance
        self.agent_positions[agent_id] = self.position
        self.agent_avg_prices[agent_id] = self.avg_price

        # Restore
        self.balance = old_balance
        self.position = old_position
        self.avg_price = old_avg_price

        return reward
