"""
Autonomous Agent Swarm System for Trading
Multi-agent reinforcement learning with swarm intelligence
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from pathlib import Path
import json
from datetime import datetime

from .trading_env import TradingEnv, MultiAgentTradingEnv, Action

logger = logging.getLogger(__name__)


class TradingAgent:
    """Base RL trading agent using Stable-Baselines3."""
    
    def __init__(self, name: str, algorithm: str = "PPO", model_path: Optional[Path] = None):
        """
        Initialize trading agent.
        
        Args:
            name: Agent name
            algorithm: RL algorithm (PPO, DDPG, SAC)
            model_path: Path to pre-trained model
        """
        self.name = name
        self.algorithm = algorithm
        self.model = None
        self.env = None
        self.training_history = []
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def create_env(self, data: pd.DataFrame, **kwargs):
        """Create training environment."""
        self.env = DummyVecEnv([lambda: TradingEnv(data, **kwargs)])
        
    def train(self, timesteps: int = 100000, progress_bar: bool = True):
        """Train the agent."""
        if self.env is None:
            raise ValueError("Environment not created. Call create_env() first.")
        
        # Create model if not exists
        if self.model is None:
            if self.algorithm == "PPO":
                self.model = PPO("MlpPolicy", self.env, verbose=1)
            elif self.algorithm == "DDPG":
                self.model = DDPG("MlpPolicy", self.env, verbose=1)
            elif self.algorithm == "SAC":
                self.model = SAC("MlpPolicy", self.env, verbose=1)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        logger.info(f"Training {self.name} for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'timesteps': timesteps,
            'algorithm': self.algorithm
        })
        
        logger.info(f"{self.name} training completed!")
    
    def predict(self, observation, deterministic: bool = True):
        """Predict action given observation."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: Path):
        """Save agent model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(str(path))
        
        # Save metadata
        metadata = {
            'name': self.name,
            'algorithm': self.algorithm,
            'training_history': self.training_history
        }
        with open(path.parent / f"{path.stem}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {self.name} to {path}")
    
    def load(self, path: Path):
        """Load agent model."""
        if self.algorithm == "PPO":
            self.model = PPO.load(str(path))
        elif self.algorithm == "DDPG":
            self.model = DDPG.load(str(path))
        elif self.algorithm == "SAC":
            self.model = SAC.load(str(path))
        
        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.training_history = metadata.get('training_history', [])
        
        logger.info(f"Loaded {self.name} from {path}")


class AgentSwarm:
    """
    Swarm Intelligence Coordinator for Multiple Trading Agents.
    
    Manages:
    - ExecutionAgent: Optimizes trade execution (minimize slippage)
    - RiskAgent: Monitors and controls risk
    - ArbitrageAgent: Finds arbitrage opportunities
    - MarketMakingAgent: Provides liquidity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize agent swarm."""
        self.config = config or self._default_config()
        self.agents = {}
        self.performance = {}
        self.swarm_state = "idle"  # idle, training, trading, paused
        
        # Create specialized agents
        self._initialize_agents()
    
    def _default_config(self) -> Dict:
        """Default swarm configuration."""
        return {
            'execution_agent': {'algorithm': 'PPO', 'enabled': True},
            'risk_agent': {'algorithm': 'SAC', 'enabled': True},
            'arbitrage_agent': {'algorithm': 'DDPG', 'enabled': True},
            'market_making_agent': {'algorithm': 'PPO', 'enabled': False},
            'initial_balance': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'coordination_mode': 'voting',  # voting, hierarchical, consensus
            'min_confidence': 0.6
        }
    
    def _initialize_agents(self):
        """Initialize all agents in the swarm."""
        for agent_type, config in self.config.items():
            if agent_type.endswith('_agent') and config.get('enabled', False):
                agent_name = agent_type.replace('_agent', '').title() + 'Agent'
                self.agents[agent_type] = TradingAgent(
                    name=agent_name,
                    algorithm=config['algorithm']
                )
                logger.info(f"Initialized {agent_name} with {config['algorithm']}")
    
    def train_swarm(self, data: pd.DataFrame, timesteps: int = 100000):
        """Train all agents in the swarm."""
        logger.info(f"Training swarm with {len(self.agents)} agents...")
        self.swarm_state = "training"
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Training {agent_name}...")
            
            # Create environment for this agent
            agent.create_env(
                data,
                initial_balance=self.config['initial_balance'],
                commission=self.config['commission'],
                slippage=self.config['slippage']
            )
            
            # Train
            agent.train(timesteps=timesteps)
            
            # Evaluate
            performance = self.evaluate_agent(agent, data)
            self.performance[agent_name] = performance
            logger.info(f"{agent_name} performance: {performance}")
        
        self.swarm_state = "idle"
        logger.info("Swarm training completed!")
    
    def evaluate_agent(self, agent: TradingAgent, data: pd.DataFrame) -> Dict:
        """Evaluate agent performance."""
        env = TradingEnv(data)
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        return {
            'total_return': info['total_return'],
            'portfolio_value': info['portfolio_value'],
            'trades': info['trades'],
            'total_reward': total_reward
        }
    
    def get_swarm_decision(self, observations: List[np.ndarray]) -> Tuple[int, float]:
        """
        Get collective decision from the swarm.
        
        Args:
            observations: List of observations for each agent
        
        Returns:
            action: Consensus action
            confidence: Confidence score (0-1)
        """
        if not self.agents:
            return Action.HOLD.value, 0.0
        
        # Get predictions from all agents
        predictions = []
        for i, (agent_name, agent) in enumerate(self.agents.items()):
            try:
                action, _ = agent.predict(observations[i])
                predictions.append(action)
            except Exception as e:
                logger.warning(f"Agent {agent_name} prediction failed: {e}")
                predictions.append(Action.HOLD.value)
        
        # Coordination strategy
        if self.config['coordination_mode'] == 'voting':
            # Majority voting
            action_counts = {}
            for action in predictions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            consensus_action = max(action_counts, key=action_counts.get)
            confidence = action_counts[consensus_action] / len(predictions)
            
        elif self.config['coordination_mode'] == 'hierarchical':
            # Priority: execution > risk > arbitrage > market_making
            priority_order = ['execution_agent', 'risk_agent', 'arbitrage_agent', 'market_making_agent']
            for agent_type in priority_order:
                if agent_type in self.agents:
                    idx = list(self.agents.keys()).index(agent_type)
                    consensus_action = predictions[idx]
                    confidence = 0.8  # High confidence in hierarchical
                    break
            else:
                consensus_action = Action.HOLD.value
                confidence = 0.0
        
        elif self.config['coordination_mode'] == 'consensus':
            # All agents must agree
            if len(set(predictions)) == 1:
                consensus_action = predictions[0]
                confidence = 1.0
            else:
                consensus_action = Action.HOLD.value  # Hold if no consensus
                confidence = 0.0
        
        else:
            raise ValueError(f"Unknown coordination mode: {self.config['coordination_mode']}")
        
        return consensus_action, confidence
    
    def execute_swarm_trading(self, data: pd.DataFrame, live: bool = False) -> Dict:
        """
        Execute trading with the swarm.
        
        Args:
            data: Market data
            live: If True, trade live; if False, backtest
        
        Returns:
            results: Trading results
        """
        logger.info(f"Swarm trading {'LIVE' if live else 'BACKTEST'} mode...")
        self.swarm_state = "trading"
        
        env = TradingEnv(data)
        obs = env.reset()
        done = False
        
        trade_log = []
        
        while not done:
            # Get observations for all agents (same for now, could be specialized)
            observations = [obs] * len(self.agents)
            
            # Get swarm decision
            action, confidence = self.get_swarm_decision(observations)
            
            # Only execute if confidence meets threshold
            if confidence >= self.config['min_confidence']:
                obs, reward, done, info = env.step(action)
                
                trade_log.append({
                    'step': env.current_step,
                    'action': Action(action).name,
                    'confidence': confidence,
                    'portfolio_value': info['portfolio_value'],
                    'return': info['total_return']
                })
            else:
                # Hold if not confident
                obs, reward, done, info = env.step(Action.HOLD.value)
        
        self.swarm_state = "idle"
        
        results = {
            'final_portfolio_value': info['portfolio_value'],
            'total_return': info['total_return'],
            'total_trades': info['trades'],
            'trade_log': trade_log
        }
        
        logger.info(f"Swarm trading completed: {results['total_return']:.2%} return")
        return results
    
    def save_swarm(self, directory: Path):
        """Save all agents in the swarm."""
        directory.mkdir(parents=True, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            agent_path = directory / f"{agent_name}.zip"
            agent.save(agent_path)
        
        # Save swarm config
        config_path = directory / "swarm_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save performance
        perf_path = directory / "swarm_performance.json"
        with open(perf_path, 'w') as f:
            json.dump(self.performance, f, indent=2)
        
        logger.info(f"Swarm saved to {directory}")
    
    def load_swarm(self, directory: Path):
        """Load all agents in the swarm."""
        # Load config
        config_path = directory / "swarm_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        
        # Load agents
        for agent_name in self.agents.keys():
            agent_path = directory / f"{agent_name}.zip"
            if agent_path.exists():
                self.agents[agent_name].load(agent_path)
        
        # Load performance
        perf_path = directory / "swarm_performance.json"
        if perf_path.exists():
            with open(perf_path) as f:
                self.performance = json.load(f)
        
        logger.info(f"Swarm loaded from {directory}")
    
    def get_status(self) -> Dict:
        """Get swarm status."""
        return {
            'state': self.swarm_state,
            'n_agents': len(self.agents),
            'agents': list(self.agents.keys()),
            'coordination_mode': self.config['coordination_mode'],
            'performance': self.performance
        }
