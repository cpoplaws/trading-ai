"""
Trading Environment for Reinforcement Learning
==============================================

OpenAI Gym-compatible environment for training RL trading agents.

Features:
- Realistic trading simulation
- Transaction costs and slippage
- Position sizing
- Risk management
- Flexible observation space
"""

import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

logger = logging.getLogger(__name__)

# Try importing gym
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gym not available. Install with: pip install gym")


class ActionType(IntEnum):
    """Trading actions."""
    SELL = 0
    HOLD = 1
    BUY = 2


@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""
    # Capital
    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # Maximum fraction of capital per trade

    # Costs
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage

    # Risk management
    max_drawdown: float = 0.20  # 20% maximum drawdown before episode ends
    stop_loss: float = 0.05  # 5% stop loss per trade

    # Reward shaping
    reward_scaling: float = 1.0
    punish_holding: bool = False  # Small penalty for holding to encourage action
    holding_penalty: float = 0.001

    # Episode
    max_steps: int = 1000
    lookback_window: int = 50  # Number of historical steps in observation


class TradingEnvironment:
    """
    Trading Environment for RL agents.

    State Space:
    - Price history (lookback_window)
    - Technical indicators
    - Current position
    - Portfolio value
    - P&L

    Action Space:
    - Discrete: SELL, HOLD, BUY
    - Or Continuous: [-1, 1] where -1=sell all, 0=hold, 1=buy max

    Reward:
    - Portfolio value change
    - Risk-adjusted (Sharpe-like)
    - Penalty for large drawdowns
    """

    def __init__(
        self,
        price_data: np.ndarray,
        config: Optional[EnvironmentConfig] = None
    ):
        """
        Initialize trading environment.

        Args:
            price_data: Historical price data (n_steps, n_features)
                       First column should be close price
            config: Environment configuration
        """
        if not GYM_AVAILABLE:
            raise ImportError("gym required. Install with: pip install gym")

        self.config = config or EnvironmentConfig()
        self.price_data = price_data
        self.n_steps, self.n_features = price_data.shape

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # SELL, HOLD, BUY

        # Observation: [lookback prices, indicators, position, cash, value]
        obs_size = self.n_features * self.config.lookback_window + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.position = 0.0  # Number of units held
        self.position_value = 0.0
        self.entry_price = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.episode_trades = 0
        self.total_commission = 0.0

        # History
        self.portfolio_history = []
        self.trade_history = []

        logger.info(f"Initialized TradingEnvironment with {self.n_steps} steps")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = self.config.lookback_window
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.episode_trades = 0
        self.total_commission = 0.0
        self.portfolio_history = []
        self.trade_history = []

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in environment.

        Args:
            action: Action to take (0=SELL, 1=HOLD, 2=BUY)

        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Get current price
        current_price = self.price_data[self.current_step, 0]

        # Execute action
        reward = 0.0
        action_taken = ActionType(action)

        if action_taken == ActionType.BUY and self.position <= 0:
            reward = self._execute_buy(current_price)

        elif action_taken == ActionType.SELL and self.position > 0:
            reward = self._execute_sell(current_price)

        elif action_taken == ActionType.HOLD:
            # Update position value
            if self.position > 0:
                self.position_value = self.position * current_price
            # Small penalty for holding (optional)
            if self.config.punish_holding:
                reward = -self.config.holding_penalty

        # Calculate portfolio value
        portfolio_value = self.balance + self.position_value
        self.portfolio_history.append(portfolio_value)

        # Update max portfolio value for drawdown calculation
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value

        # Calculate drawdown
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value

        # Check if episode should end
        done = False
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'drawdown': drawdown,
            'trades': self.episode_trades,
            'commission_paid': self.total_commission
        }

        # End conditions
        if self.current_step >= self.n_steps - 1:
            done = True
            info['reason'] = 'max_steps'

        elif drawdown > self.config.max_drawdown:
            done = True
            reward -= 10.0  # Large penalty for exceeding max drawdown
            info['reason'] = 'max_drawdown_exceeded'

        elif portfolio_value <= 0:
            done = True
            reward -= 100.0  # Huge penalty for bankruptcy
            info['reason'] = 'bankruptcy'

        # Move to next step
        self.current_step += 1

        # Get next observation
        observation = self._get_observation()

        return observation, reward, done, info

    def _execute_buy(self, price: float) -> float:
        """
        Execute buy action.

        Args:
            price: Current price

        Returns:
            Reward for this action
        """
        # Calculate maximum position we can buy
        available_cash = self.balance
        commission_rate = self.config.commission
        slippage_rate = self.config.slippage

        # Effective buy price (price + slippage)
        effective_price = price * (1 + slippage_rate)

        # Maximum units we can buy
        max_units = available_cash / (effective_price * (1 + commission_rate))

        # Apply position size limit
        max_position_value = self.config.initial_balance * self.config.max_position_size
        max_units = min(max_units, max_position_value / effective_price)

        if max_units > 0:
            # Execute buy
            units_bought = max_units
            cost = units_bought * effective_price
            commission = cost * commission_rate

            self.position += units_bought
            self.balance -= (cost + commission)
            self.entry_price = effective_price
            self.position_value = units_bought * price  # Current market value
            self.episode_trades += 1
            self.total_commission += commission

            self.trade_history.append({
                'step': self.current_step,
                'action': 'BUY',
                'price': price,
                'units': units_bought,
                'cost': cost,
                'commission': commission
            })

            # Small positive reward for taking action
            return 0.1

        return 0.0

    def _execute_sell(self, price: float) -> float:
        """
        Execute sell action.

        Args:
            price: Current price

        Returns:
            Reward for this action (P&L of trade)
        """
        if self.position <= 0:
            return 0.0

        # Effective sell price (price - slippage)
        slippage_rate = self.config.slippage
        effective_price = price * (1 - slippage_rate)

        # Calculate proceeds
        proceeds = self.position * effective_price
        commission = proceeds * self.config.commission

        # Calculate P&L
        entry_cost = self.position * self.entry_price
        pnl = proceeds - entry_cost - commission
        pnl_pct = pnl / entry_cost if entry_cost > 0 else 0

        # Update balances
        self.balance += (proceeds - commission)
        self.position = 0.0
        self.position_value = 0.0
        self.episode_trades += 1
        self.total_commission += commission

        self.trade_history.append({
            'step': self.current_step,
            'action': 'SELL',
            'price': price,
            'units': self.position,
            'proceeds': proceeds,
            'commission': commission,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })

        # Reward is the P&L percentage, scaled
        reward = pnl_pct * self.config.reward_scaling

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array
        """
        # Get historical price data
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step

        historical_data = self.price_data[start_idx:end_idx]

        # Pad if necessary
        if len(historical_data) < self.config.lookback_window:
            padding = np.zeros((self.config.lookback_window - len(historical_data), self.n_features))
            historical_data = np.vstack([padding, historical_data])

        # Flatten historical data
        historical_flat = historical_data.flatten()

        # Current state features
        current_price = self.price_data[self.current_step, 0]
        portfolio_value = self.balance + self.position_value

        state_features = np.array([
            self.position / (self.config.initial_balance / current_price),  # Normalized position
            self.balance / self.config.initial_balance,  # Normalized cash
            portfolio_value / self.config.initial_balance,  # Normalized portfolio value
            self.position_value / portfolio_value if portfolio_value > 0 else 0,  # Position ratio
            (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value  # Drawdown
        ])

        # Combine all features
        observation = np.concatenate([historical_flat, state_features])

        return observation.astype(np.float32)

    def render(self, mode='human'):
        """
        Render environment (for debugging).

        Args:
            mode: Render mode
        """
        if mode == 'human':
            portfolio_value = self.balance + self.position_value
            pnl = portfolio_value - self.config.initial_balance
            pnl_pct = (pnl / self.config.initial_balance) * 100

            print(f"\n--- Step {self.current_step} ---")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f} units (${self.position_value:.2f})")
            print(f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"Trades: {self.episode_trades}")
            print(f"Commission Paid: ${self.total_commission:.2f}")

    def get_episode_stats(self) -> Dict:
        """
        Get statistics for completed episode.

        Returns:
            Episode statistics
        """
        portfolio_values = np.array(self.portfolio_history)

        if len(portfolio_values) == 0:
            return {}

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Total return
        total_return = (portfolio_values[-1] - self.config.initial_balance) / self.config.initial_balance

        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming daily data
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        profitable_trades = sum(1 for trade in self.trade_history
                               if trade.get('action') == 'SELL' and trade.get('pnl', 0) > 0)
        sell_trades = sum(1 for trade in self.trade_history if trade.get('action') == 'SELL')
        win_rate = profitable_trades / sell_trades if sell_trades > 0 else 0.0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_portfolio_value': portfolio_values[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.episode_trades,
            'win_rate': win_rate,
            'total_commission': self.total_commission
        }


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Trading Environment Example")
    print("=" * 60)

    if not GYM_AVAILABLE:
        print("❌ gym not available. Please install: pip install gym")
        exit(1)

    # Generate synthetic price data
    np.random.seed(42)
    n_steps = 1000
    n_features = 5

    # Create trending price series
    t = np.linspace(0, 100, n_steps)
    prices = 100 + 10 * np.sin(t / 10) + np.cumsum(np.random.randn(n_steps) * 0.3)

    # Additional features
    returns = np.diff(prices, prepend=prices[0])
    volume = np.random.lognormal(10, 0.3, n_steps)
    rsi = 50 + 25 * np.sin(t / 15)
    macd = np.sin(t / 20)

    data = np.column_stack([prices, returns, volume, rsi, macd])

    # Create environment
    config = EnvironmentConfig(
        initial_balance=10000.0,
        commission=0.001,
        max_drawdown=0.20,
        lookback_window=20
    )

    env = TradingEnvironment(data, config)

    # Run random agent
    print("\nRunning random agent for 10 episodes...")
    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Random action
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            total_reward += reward

        stats = env.get_episode_stats()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Return: {stats['total_return_pct']:.2f}%")
        print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
        print(f"  Max DD: {stats['max_drawdown']:.2%}")
        print(f"  Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")

    print("\n✅ Trading Environment Example Complete!")
