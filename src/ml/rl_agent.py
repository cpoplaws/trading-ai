"""
Reinforcement Learning Trading Agent
Self-learning agent that discovers optimal trading strategies.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
import random

logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading actions."""
    BUY = 0
    SELL = 1
    HOLD = 2


@dataclass
class State:
    """Market state representation."""
    # Price features
    price_position: int  # Discretized price level (0-9)
    trend: int  # -1 (down), 0 (neutral), 1 (up)
    volatility: int  # 0 (low), 1 (medium), 2 (high)

    # Position state
    position: int  # 0 (no position), 1 (long)

    # Technical indicators
    rsi_level: int  # 0 (oversold), 1 (neutral), 2 (overbought)

    def to_tuple(self) -> Tuple:
        """Convert state to tuple for hashing."""
        return (self.price_position, self.trend, self.volatility,
                self.position, self.rsi_level)


@dataclass
class Experience:
    """Experience replay memory."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


@dataclass
class Episode:
    """Training episode results."""
    episode_number: int
    total_reward: float
    actions_taken: int
    final_balance: float
    profit_loss: float
    win_rate: float


class TradingEnvironment:
    """
    Trading environment for RL agent.

    Simulates market with price data and executes trades.
    """

    def __init__(
        self,
        prices: List[float],
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001
    ):
        """
        Initialize trading environment.

        Args:
            prices: Historical price data
            initial_balance: Starting balance
            trading_fee: Trading fee percentage
        """
        self.prices = prices
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee

        # State
        self.reset()

        logger.info(f"Trading environment initialized with {len(prices)} price points")

    def reset(self) -> State:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Number of tokens held
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trades = 0
        self.wins = 0

        return self._get_state()

    def _get_state(self) -> State:
        """Get current market state."""
        if self.current_step >= len(self.prices):
            return None

        current_price = self.prices[self.current_step]

        # Price position (discretized)
        if self.current_step > 0:
            price_change = (current_price - self.prices[0]) / self.prices[0]
            price_position = min(9, max(0, int((price_change + 0.1) * 50)))
        else:
            price_position = 5

        # Trend (based on recent price movement)
        if self.current_step >= 5:
            recent_prices = self.prices[max(0, self.current_step-5):self.current_step+1]
            if recent_prices[-1] > recent_prices[0] * 1.01:
                trend = 1  # Up
            elif recent_prices[-1] < recent_prices[0] * 0.99:
                trend = -1  # Down
            else:
                trend = 0  # Neutral
        else:
            trend = 0

        # Volatility (based on price changes)
        if self.current_step >= 10:
            recent_prices = self.prices[max(0, self.current_step-10):self.current_step+1]
            returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                      for i in range(1, len(recent_prices))]
            volatility_val = sum(abs(r) for r in returns) / len(returns)

            if volatility_val > 0.02:
                volatility = 2  # High
            elif volatility_val > 0.01:
                volatility = 1  # Medium
            else:
                volatility = 0  # Low
        else:
            volatility = 1

        # RSI (simplified)
        if self.current_step >= 14:
            recent_prices = self.prices[max(0, self.current_step-14):self.current_step+1]
            gains = sum(max(0, recent_prices[i] - recent_prices[i-1])
                       for i in range(1, len(recent_prices)))
            losses = sum(max(0, recent_prices[i-1] - recent_prices[i])
                        for i in range(1, len(recent_prices)))

            if losses == 0:
                rsi = 100
            else:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))

            if rsi > 70:
                rsi_level = 2  # Overbought
            elif rsi < 30:
                rsi_level = 0  # Oversold
            else:
                rsi_level = 1  # Neutral
        else:
            rsi_level = 1

        return State(
            price_position=price_position,
            trend=trend,
            volatility=volatility,
            position=1 if self.position > 0 else 0,
            rsi_level=rsi_level
        )

    def step(self, action: Action) -> Tuple[State, float, bool]:
        """
        Execute action and return next state, reward, done.

        Args:
            action: Action to take

        Returns:
            (next_state, reward, done) tuple
        """
        if self.current_step >= len(self.prices) - 1:
            return None, 0.0, True

        current_price = self.prices[self.current_step]
        reward = 0.0

        # Execute action
        if action == Action.BUY and self.position == 0:
            # Buy with all available balance
            cost = self.balance * (1 + self.trading_fee)
            self.position = self.balance / current_price
            self.balance = 0
            self.entry_price = current_price
            self.trades += 1
            reward = -0.01  # Small penalty for trading

        elif action == Action.SELL and self.position > 0:
            # Sell entire position
            proceeds = self.position * current_price * (1 - self.trading_fee)
            self.balance = proceeds

            # Calculate profit/loss
            profit = proceeds - self.initial_balance
            reward = profit / self.initial_balance  # Percentage return

            if profit > 0:
                self.wins += 1

            self.position = 0
            self.entry_price = 0.0
            self.trades += 1

        elif action == Action.HOLD:
            # Calculate unrealized P&L for holding
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                reward = unrealized_pnl * 0.1  # Small reward for unrealized gains
            else:
                reward = 0.0

        # Move to next step
        self.current_step += 1
        self.total_reward += reward

        next_state = self._get_state()
        done = self.current_step >= len(self.prices) - 1

        return next_state, reward, done

    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        if self.current_step >= len(self.prices):
            return self.balance

        current_price = self.prices[self.current_step]
        return self.balance + (self.position * current_price)


class QLearningAgent:
    """
    Q-Learning based trading agent.

    Learns optimal trading strategy through trial and error.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning agent.

        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action -> Q-value
        self.q_table: Dict[Tuple, Dict[Action, float]] = {}

        # Training stats
        self.episodes_trained = 0
        self.total_reward_history = []

        logger.info(
            f"Q-Learning agent initialized "
            f"(lr={learning_rate}, gamma={discount_factor}, epsilon={epsilon})"
        )

    def get_q_value(self, state: State, action: Action) -> float:
        """Get Q-value for state-action pair."""
        state_tuple = state.to_tuple()

        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {a: 0.0 for a in Action}

        return self.q_table[state_tuple][action]

    def set_q_value(self, state: State, action: Action, value: float):
        """Set Q-value for state-action pair."""
        state_tuple = state.to_tuple()

        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {a: 0.0 for a in Action}

        self.q_table[state_tuple][action] = value

    def choose_action(self, state: State, training: bool = True) -> Action:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use exploration

        Returns:
            Action to take
        """
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            return random.choice(list(Action))

        # Exploitation: choose best action
        q_values = {action: self.get_q_value(state, action) for action in Action}
        max_q = max(q_values.values())

        # Handle ties randomly
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: Optional[State]
    ):
        """
        Update Q-values using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)

        if next_state is None:
            # Terminal state
            target_q = reward
        else:
            # Best Q-value for next state
            next_q_values = [self.get_q_value(next_state, a) for a in Action]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q

        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state, action, new_q)

    def train(
        self,
        env: TradingEnvironment,
        num_episodes: int = 100
    ) -> List[Episode]:
        """
        Train agent on environment.

        Args:
            env: Trading environment
            num_episodes: Number of training episodes

        Returns:
            List of episode results
        """
        episodes = []

        logger.info(f"Starting training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            actions_taken = 0

            while not done and state is not None:
                # Choose and execute action
                action = self.choose_action(state, training=True)
                next_state, reward, done = env.step(action)

                # Update Q-values
                self.update(state, action, reward, next_state)

                episode_reward += reward
                actions_taken += 1
                state = next_state

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Record episode
            final_value = env.get_portfolio_value()
            profit_loss = final_value - env.initial_balance
            win_rate = env.wins / env.trades if env.trades > 0 else 0.0

            episode_result = Episode(
                episode_number=episode + 1,
                total_reward=episode_reward,
                actions_taken=actions_taken,
                final_balance=final_value,
                profit_loss=profit_loss,
                win_rate=win_rate
            )

            episodes.append(episode_result)
            self.total_reward_history.append(episode_reward)
            self.episodes_trained += 1

            # Log progress
            if (episode + 1) % 20 == 0:
                avg_reward = sum(self.total_reward_history[-20:]) / 20
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Final Value: ${final_value:,.2f}"
                )

        logger.info(f"Training complete! Trained for {num_episodes} episodes")

        return episodes

    def evaluate(self, env: TradingEnvironment) -> Dict:
        """
        Evaluate trained agent.

        Args:
            env: Trading environment

        Returns:
            Evaluation metrics
        """
        state = env.reset()
        done = False
        total_reward = 0.0
        actions = {Action.BUY: 0, Action.SELL: 0, Action.HOLD: 0}

        while not done and state is not None:
            action = self.choose_action(state, training=False)
            next_state, reward, done = env.step(action)

            total_reward += reward
            actions[action] += 1
            state = next_state

        final_value = env.get_portfolio_value()
        profit_loss = final_value - env.initial_balance
        roi = (profit_loss / env.initial_balance) * 100

        return {
            'final_value': final_value,
            'profit_loss': profit_loss,
            'roi': roi,
            'total_reward': total_reward,
            'trades': env.trades,
            'wins': env.wins,
            'win_rate': env.wins / env.trades if env.trades > 0 else 0.0,
            'actions': dict(actions),
            'q_table_size': len(self.q_table)
        }


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🤖 Reinforcement Learning Trading Agent Demo")
    print("=" * 60)

    # Generate sample price data
    print("\n1. Generating Price Data...")
    print("-" * 60)

    random.seed(42)
    base_price = 2000.0
    prices = [base_price]

    # Generate realistic price movement
    for i in range(200):
        # Trend + noise
        trend = math.sin(i / 20) * 0.002  # Cyclical trend
        noise = random.gauss(0, 0.01)
        return_val = trend + noise

        new_price = prices[-1] * (1 + return_val)
        prices.append(new_price)

    print(f"Generated {len(prices)} price points")
    print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")

    # Create environment and agent
    print("\n2. Training RL Agent...")
    print("-" * 60)

    env = TradingEnvironment(prices, initial_balance=10000.0)
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )

    # Train
    episodes = agent.train(env, num_episodes=100)

    # Show training progress
    print("\n3. Training Results...")
    print("-" * 60)

    final_episodes = episodes[-5:]
    print("\nLast 5 Episodes:")
    for ep in final_episodes:
        print(f"  Episode {ep.episode_number}: "
              f"Reward={ep.total_reward:.4f}, "
              f"P&L=${ep.profit_loss:+,.2f}, "
              f"Win Rate={ep.win_rate*100:.1f}%")

    # Evaluate
    print("\n4. Evaluating Trained Agent...")
    print("-" * 60)

    results = agent.evaluate(env)

    print(f"\nPerformance:")
    print(f"  Initial Balance: ${env.initial_balance:,.2f}")
    print(f"  Final Balance: ${results['final_value']:,.2f}")
    print(f"  Profit/Loss: ${results['profit_loss']:+,.2f}")
    print(f"  ROI: {results['roi']:+.2f}%")
    print(f"\nTrading Stats:")
    print(f"  Total Trades: {results['trades']}")
    print(f"  Wins: {results['wins']}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"\nActions Taken:")
    for action, count in results['actions'].items():
        print(f"  {action.name}: {count}")
    print(f"\nQ-Table Size: {results['q_table_size']} states learned")

    print("\n✅ RL agent demo complete!")
