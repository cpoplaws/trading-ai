"""
Deep Q-Network (DQN) Trading Agent
Advanced RL agent with neural network Q-function approximation.
"""
import logging
from typing import List, Tuple, Optional, Deque
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import random
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class DQNConfig:
    """DQN configuration."""
    state_size: int = 10
    action_size: int = 3  # BUY, SELL, HOLD
    hidden_size: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 10  # Update target network every N episodes


class Experience:
    """Experience tuple for replay buffer."""

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """
    Deep Q-Network.

    Neural network that approximates Q-values for state-action pairs.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128
    ):
        super(DQN, self).__init__()

        # Network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor (batch, state_size)

        Returns:
            Q-values for each action (batch, action_size)
        """
        x = self.relu(self.fc1(state))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    """
    DQN Trading Agent.

    Uses deep Q-learning with experience replay and target network.
    """

    def __init__(self, config: Optional[DQNConfig] = None):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config or DQNConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Q-networks
        self.q_network = DQN(
            self.config.state_size,
            self.config.action_size,
            self.config.hidden_size
        ).to(self.device)

        self.target_network = DQN(
            self.config.state_size,
            self.config.action_size,
            self.config.hidden_size
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.memory = ReplayBuffer(self.config.memory_size)

        # Training stats
        self.epsilon = self.config.epsilon
        self.episodes_trained = 0
        self.total_steps = 0

        logger.info(f"DQN agent initialized on device: {self.device}")

    def get_state_vector(self, state: dict) -> np.ndarray:
        """
        Convert state dict to vector.

        Args:
            state: State dictionary

        Returns:
            State vector
        """
        # Example state features:
        # - Price position (normalized)
        # - Trend indicator
        # - Volatility
        # - RSI
        # - Position (has position or not)
        # - Portfolio value (normalized)
        # - Recent returns
        # etc.

        vector = np.array([
            state.get('price_position', 0.5),
            state.get('trend', 0.0),
            state.get('volatility', 0.5),
            state.get('rsi', 0.5),
            state.get('position', 0.0),
            state.get('portfolio_value', 1.0),
            state.get('returns_1', 0.0),
            state.get('returns_5', 0.0),
            state.get('returns_10', 0.0),
            state.get('ma_diff', 0.0)
        ])

        return vector[:self.config.state_size]

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: State vector
            training: If True, use exploration

        Returns:
            Action index (0=BUY, 1=SELL, 2=HOLD)
        """
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_size - 1)

        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience)

    def learn(self):
        """
        Learn from batch of experiences.

        Implements DQN learning with experience replay and target network.
        """
        if len(self.memory) < self.config.batch_size:
            return

        # Sample batch
        experiences = self.memory.sample(self.config.batch_size)

        # Extract batch data
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def train_episode(self, env, max_steps: int = 1000) -> dict:
        """
        Train for one episode.

        Args:
            env: Trading environment
            max_steps: Maximum steps per episode

        Returns:
            Episode metrics
        """
        state_dict = env.reset()
        state = self.get_state_vector(state_dict)

        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # Choose action
            action = self.choose_action(state, training=True)

            # Take action
            next_state_dict, reward, done = env.step(action)
            next_state = self.get_state_vector(next_state_dict)

            # Store experience
            self.store_experience(state, action, reward, next_state, done)

            # Learn
            loss = self.learn()

            total_reward += reward
            steps += 1
            self.total_steps += 1

            state = next_state

            if done:
                break

        # Decay epsilon
        self.decay_epsilon()

        # Update target network
        if (self.episodes_trained + 1) % self.config.target_update_freq == 0:
            self.update_target_network()
            logger.info(f"Target network updated at episode {self.episodes_trained + 1}")

        self.episodes_trained += 1

        return {
            'episode': self.episodes_trained,
            'reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }

    def save(self, path: str):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episodes_trained = checkpoint['episodes_trained']
        logger.info(f"Model loaded from {path}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not installed!")
        print("Install with: pip install torch")
        exit(1)

    print("🤖 Deep Q-Network (DQN) Trading Agent Demo")
    print("=" * 60)

    # Simple mock environment for demo
    class MockTradingEnv:
        def __init__(self):
            self.step_count = 0
            self.max_steps = 100

        def reset(self):
            self.step_count = 0
            return {
                'price_position': 0.5,
                'trend': 0.0,
                'volatility': 0.3,
                'rsi': 0.5,
                'position': 0.0,
                'portfolio_value': 1.0,
                'returns_1': 0.0,
                'returns_5': 0.0,
                'returns_10': 0.0,
                'ma_diff': 0.0
            }

        def step(self, action):
            self.step_count += 1

            # Mock reward (random for demo)
            reward = np.random.normal(0, 0.1)

            # Mock next state
            next_state = {
                'price_position': np.random.uniform(0, 1),
                'trend': np.random.uniform(-1, 1),
                'volatility': np.random.uniform(0, 1),
                'rsi': np.random.uniform(0, 1),
                'position': float(action == 0),  # BUY
                'portfolio_value': 1.0 + reward,
                'returns_1': np.random.normal(0, 0.01),
                'returns_5': np.random.normal(0, 0.02),
                'returns_10': np.random.normal(0, 0.03),
                'ma_diff': np.random.normal(0, 0.01)
            }

            done = self.step_count >= self.max_steps

            return next_state, reward, done

    # Create agent and environment
    print("\n1. Initializing DQN agent...")
    config = DQNConfig(
        state_size=10,
        action_size=3,
        hidden_size=64,
        learning_rate=0.001,
        batch_size=32
    )

    agent = DQNAgent(config)
    env = MockTradingEnv()

    print(f"   State size: {config.state_size}")
    print(f"   Action size: {config.action_size} (BUY/SELL/HOLD)")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Memory size: {config.memory_size}")

    # Train
    print("\n2. Training DQN agent...")
    num_episodes = 50

    for episode in range(num_episodes):
        metrics = agent.train_episode(env)

        if (episode + 1) % 10 == 0:
            print(f"   Episode {metrics['episode']}: "
                  f"Reward={metrics['reward']:.2f}, "
                  f"Epsilon={metrics['epsilon']:.3f}, "
                  f"Memory={metrics['memory_size']}")

    print(f"\n3. Training complete!")
    print(f"   Episodes: {agent.episodes_trained}")
    print(f"   Total steps: {agent.total_steps}")
    print(f"   Final epsilon: {agent.epsilon:.3f}")
    print(f"   Memory size: {len(agent.memory)}")

    # Test
    print("\n4. Testing trained agent...")
    state_dict = env.reset()
    state = agent.get_state_vector(state_dict)
    action = agent.choose_action(state, training=False)

    action_names = ["BUY", "SELL", "HOLD"]
    print(f"   Test action: {action_names[action]}")

    print("\n✅ DQN agent demo complete!")
    print("\nFeatures:")
    print("- Deep neural network Q-function")
    print("- Experience replay buffer")
    print("- Target network for stability")
    print("- Epsilon-greedy exploration")
    print("- GPU support")
