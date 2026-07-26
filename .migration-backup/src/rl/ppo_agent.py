"""
Proximal Policy Optimization (PPO) Agent
=========================================

State-of-the-art policy gradient algorithm for trading.

Features:
- Clipped surrogate objective (prevents large policy updates)
- Advantage estimation (GAE)
- Multiple epochs per batch
- Stable and sample-efficient

PPO is the preferred choice for:
- Continuous learning
- Stable training
- Production deployment
"""

import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


@dataclass
class PPOConfig:
    """PPO configuration."""
    # Network architecture
    hidden_sizes: List[int] = None  # [256, 256]
    activation: str = "tanh"

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    lambda_gae: float = 0.95  # GAE parameter
    epsilon_clip: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient

    # Optimization
    batch_size: int = 64
    n_epochs: int = 10  # Number of epochs per batch
    max_grad_norm: float = 0.5  # Gradient clipping

    # Experience collection
    steps_per_episode: int = 2048  # Steps before policy update

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Actor: Policy network (outputs action probabilities)
    Critic: Value network (estimates state value)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        activation: str = "tanh"
    ):
        super(ActorCritic, self).__init__()

        # Choose activation
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.ReLU

        # Shared feature extractor
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                act_fn(),
            ])
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Linear(prev_size, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(prev_size, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Special initialization for output layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            action_logits: Action logits (batch_size, action_dim)
            value: State value (batch_size, 1)
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, state):
        """
        Sample action from policy.

        Args:
            state: State tensor

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value
        """
        action_logits, value = self(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, states, actions):
        """
        Evaluate actions (for training).

        Args:
            states: State tensor (batch_size, state_dim)
            actions: Action tensor (batch_size,)

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_logits, values = self(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """
    PPO Agent for trading.

    Learns optimal trading policy through interaction with environment.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config or PPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create actor-critic network
        self.policy = ActorCritic(
            state_dim,
            action_dim,
            self.config.hidden_sizes,
            self.config.activation
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }

        logger.info(f"Initialized PPO agent with state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state: np.ndarray) -> Tuple[int, Dict]:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            action: Selected action
            info: Additional information
        """
        self.policy.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.policy.get_action(state_tensor)

        action = action.item()
        log_prob = log_prob.item()
        value = value.item()

        return action, {'log_prob': log_prob, 'value': value}

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """
        Store transition in buffer.

        Args:
            state: State
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
            value: State value estimate
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            next_value: Value of next state (for bootstrapping)

        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.lambda_gae * last_gae * (1 - dones[t])
            advantages[t] = last_gae

        # Compute returns
        returns = advantages + values[:-1]

        return advantages, returns

    def update(self, next_value: float = 0.0) -> Dict:
        """
        Update policy using collected experience.

        Args:
            next_value: Value of next state

        Returns:
            Training statistics
        """
        if len(self.states) == 0:
            return {}

        self.policy.train()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Training loop
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        epoch_stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'total_loss': []}

        for epoch in range(self.config.n_epochs):
            # Shuffle data
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                batch_indices = indices[start_idx:end_idx]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states,
                    batch_actions
                )

                # Compute ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.epsilon_clip,
                    1.0 + self.config.epsilon_clip
                ) * batch_advantages

                # Policy loss (maximize advantage, but clip updates)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.MSELoss()(values, batch_returns)

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # Record stats
                epoch_stats['policy_loss'].append(policy_loss.item())
                epoch_stats['value_loss'].append(value_loss.item())
                epoch_stats['entropy'].append(entropy.mean().item())
                epoch_stats['total_loss'].append(total_loss.item())

        # Average stats across all epochs/batches
        stats = {
            'policy_loss': np.mean(epoch_stats['policy_loss']),
            'value_loss': np.mean(epoch_stats['value_loss']),
            'entropy': np.mean(epoch_stats['entropy']),
            'total_loss': np.mean(epoch_stats['total_loss'])
        }

        # Store stats
        for key, value in stats.items():
            self.training_stats[key].append(value)

        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        return stats

    def save(self, path: str):
        """Save agent to disk."""
        torch.save({
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Agent saved to {path}")

    @classmethod
    def load(cls, path: str, state_dim: int, action_dim: int) -> 'PPOAgent':
        """Load agent from disk."""
        checkpoint = torch.load(path)

        agent = cls(state_dim, action_dim, checkpoint['config'])
        agent.policy.load_state_dict(checkpoint['policy_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        agent.training_stats = checkpoint['training_stats']

        logger.info(f"Agent loaded from {path}")
        return agent


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("PPO Agent Example")
    print("=" * 60)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install: pip install torch")
        exit(1)

    # Create agent
    state_dim = 25
    action_dim = 3  # SELL, HOLD, BUY

    config = PPOConfig(
        hidden_sizes=[128, 128],
        learning_rate=3e-4,
        epsilon_clip=0.2
    )

    agent = PPOAgent(state_dim, action_dim, config)

    # Simulate training episode
    print("\nSimulating training episode...")
    np.random.seed(42)

    for step in range(100):
        # Random state
        state = np.random.randn(state_dim)

        # Select action
        action, info = agent.select_action(state)

        # Random reward
        reward = np.random.randn()

        # Store transition
        agent.store_transition(
            state,
            action,
            info['log_prob'],
            reward,
            info['value'],
            done=(step == 99)
        )

    # Update policy
    print("Updating policy...")
    stats = agent.update()

    print(f"\nTraining stats:")
    print(f"  Policy loss: {stats['policy_loss']:.4f}")
    print(f"  Value loss: {stats['value_loss']:.4f}")
    print(f"  Entropy: {stats['entropy']:.4f}")
    print(f"  Total loss: {stats['total_loss']:.4f}")

    # Save and load
    print("\nTesting save/load...")
    agent.save("ppo_test.pth")
    loaded_agent = PPOAgent.load("ppo_test.pth", state_dim, action_dim)

    print("\n✅ PPO Agent Example Complete!")
