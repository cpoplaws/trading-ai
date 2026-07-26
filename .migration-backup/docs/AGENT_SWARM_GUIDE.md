# 🤖 Agent Swarm Trading System - Complete Guide

## Overview

The Agent Swarm system enables **autonomous trading** using multiple reinforcement learning (RL) agents that work together to make intelligent trading decisions. Each agent specializes in different trading strategies and they coordinate to reach consensus decisions.

## 🎯 Key Features

- **Multi-Agent Coordination**: 4 specialized agents working together
- **Reinforcement Learning**: Agents learn from historical data and improve over time
- **Multiple Algorithms**: PPO, SAC, DDPG for different trading styles
- **Coordination Modes**: Voting, hierarchical, and consensus decision-making
- **Paper & Live Trading**: Safe testing before deploying real capital
- **Dashboard Integration**: Visual monitoring and control interface

## 🤖 Agent Types

### 1. **ExecutionAgent (PPO)**
- **Purpose**: General trade execution and market timing
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Strengths**: Stable learning, good for continuous trading
- **Use Case**: Primary trading decisions, market entries/exits

### 2. **RiskAgent (SAC)**
- **Purpose**: Risk assessment and portfolio protection
- **Algorithm**: Soft Actor-Critic (SAC)
- **Strengths**: Maximum entropy RL, exploration-exploitation balance
- **Use Case**: Position sizing, risk mitigation, stop-loss decisions

### 3. **ArbitrageAgent (DDPG)**
- **Purpose**: Finding and exploiting arbitrage opportunities
- **Algorithm**: Deep Deterministic Policy Gradient (DDPG)
- **Strengths**: Continuous action space, precise timing
- **Use Case**: Cross-exchange arbitrage, price inefficiencies

### 4. **MarketMakingAgent (PPO)**
- **Purpose**: Providing liquidity and spread capture
- **Algorithm**: PPO
- **Strengths**: Stable in range-bound markets
- **Use Case**: Market making strategies, range trading

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Swarm                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ExecutionAgent │ RiskAgent │ ArbitrageAgent │ MM │  │
│  └──────────────────────────────────────────────────┘  │
│                         ▼                                │
│              Coordination Layer                          │
│         (Voting/Hierarchical/Consensus)                  │
│                         ▼                                │
│                  Consensus Decision                      │
│                         ▼                                │
│               Order Manager / Broker                     │
└─────────────────────────────────────────────────────────┘
```

## 📊 Trading Environment

The system uses a custom OpenAI Gym environment (`TradingEnv`) that simulates realistic trading:

### State Space
- **Historical prices**: Lookback window of OHLCV data
- **Technical indicators**: SMA, RSI, volatility, volume ratios
- **Position info**: Current holdings, cash balance, unrealized P&L
- **Momentum features**: Price momentum, support/resistance levels

### Action Space (7 discrete actions)
0. **HOLD** - No action
1. **BUY_SMALL** - Buy 10% of available capital
2. **BUY_MEDIUM** - Buy 20% of available capital
3. **BUY_LARGE** - Buy 30% of available capital
4. **SELL_SMALL** - Sell 10% of position
5. **SELL_MEDIUM** - Sell 20% of position
6. **SELL_LARGE** - Sell 30% of position

### Reward Function
- **Profit-based**: Realized P&L from trades
- **Sharpe-adjusted**: Returns adjusted for volatility
- **Risk penalties**: Penalties for excessive risk-taking
- **Action incentives**: Small rewards/penalties to encourage/discourage certain behaviors

## 🚀 Getting Started

### 1. Installation

Dependencies are already installed, but if needed:

```bash
pip install stable-baselines3 gym torch tensorboard
```

### 2. Training Your First Agent

```python
from src.execution.trading_env import TradingEnv
from src.execution.agent_swarm import TradingAgent
import yfinance as yf

# Fetch training data
data = yf.Ticker('AAPL').history(period='6mo')
data.columns = [col.lower() for col in data.columns]

# Create environment
env = TradingEnv(data=data, initial_balance=100000)

# Create and train agent
agent = TradingAgent(
    agent_id="my_agent",
    env=env,
    algorithm="PPO",
    learning_rate=0.0003
)

# Train for 100k timesteps
agent.train(total_timesteps=100000)

# Save model
agent.save('./models/my_first_agent.zip')
```

### 3. Training a Full Swarm

```python
from src.execution.agent_swarm import AgentSwarm

# Configure swarm
config = {
    'coordination_mode': 'voting',
    'min_confidence': 0.6,
    'agent_configs': {
        'ExecutionAgent': {'algorithm': 'PPO', 'enabled': True},
        'RiskAgent': {'algorithm': 'SAC', 'enabled': True},
        'ArbitrageAgent': {'algorithm': 'DDPG', 'enabled': True}
    }
}

# Initialize swarm
swarm = AgentSwarm(config=config)

# Add and train agents
for agent_name, agent_config in config['agent_configs'].items():
    env = TradingEnv(data=data, initial_balance=100000)
    swarm.add_agent(agent_name, agent_config['algorithm'], env)
    swarm.train_agent(agent_name, total_timesteps=100000)

# Save swarm
swarm.save_swarm('./models/my_swarm')
```

### 4. Using the Dashboard

Launch the unified dashboard:

```bash
~/launch_command_center.sh
```

Then navigate to **🤖 Agent Swarm** page where you can:
- Monitor agent status and performance
- Train agents with custom parameters
- Deploy swarm for trading
- View real-time decision-making
- Adjust swarm configuration

## 🎮 Coordination Modes

### Voting (Majority Consensus)
```python
swarm.set_coordination_mode('voting')
```
- Each agent votes on action (BUY/SELL/HOLD)
- Majority decision wins
- **Best for**: Democratic decision-making, balanced strategies

### Hierarchical (Priority-Based)
```python
swarm.set_coordination_mode('hierarchical')
```
- Agents ranked by priority
- Higher-priority agents have more weight
- **Best for**: When certain agents are more reliable

### Consensus (Unanimous)
```python
swarm.set_coordination_mode('consensus')
```
- All agents must agree
- Very conservative approach
- **Best for**: High-confidence trades only, risk aversion

## 📈 Performance Monitoring

### Key Metrics

1. **Individual Agent Metrics**:
   - Total return
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Number of trades

2. **Swarm Metrics**:
   - Consensus strength
   - Agreement rate
   - Combined return
   - Decision latency

3. **Risk Metrics**:
   - Portfolio volatility
   - Value at Risk (VaR)
   - Position concentration
   - Exposure by asset

### Tracking Performance

```python
# Get swarm status
status = swarm.get_status()
print(f"Status: {status['state']}")
print(f"Active agents: {status['active_agents']}")

# Get individual agent performance
for agent_id, perf in swarm.performance.items():
    print(f"{agent_id}: {perf['avg_reward']:.4f}")
```

## 🔧 Configuration & Tuning

### Training Parameters

```python
# Learning rate: How fast agent learns
learning_rate = 0.0003  # Lower = more stable, Higher = faster learning

# Timesteps: How long to train
total_timesteps = 100000  # More = better performance, longer training

# Batch size: Training batch size
batch_size = 128  # 64, 128, 256 are common

# Buffer size: Experience replay buffer
buffer_size = 50000  # For SAC/DDPG
```

### Environment Parameters

```python
env = TradingEnv(
    data=data,
    initial_balance=100000,    # Starting capital
    commission=0.001,          # 0.1% commission per trade
    slippage=0.0005,           # 0.05% slippage
    max_position=0.3,          # Max 30% in single position
    lookback_window=20         # 20 periods of history
)
```

### Risk Management

```python
swarm.set_risk_parameters(
    max_position_size=0.3,     # Max 30% per position
    stop_loss=0.05,            # 5% stop loss
    take_profit=0.15,          # 15% take profit
    max_drawdown=0.10          # 10% max portfolio drawdown
)
```

## 🔴 Live Trading Integration

### Step-by-Step Live Deployment

⚠️ **WARNING**: Always test thoroughly with paper trading first!

```python
# 1. Load trained swarm
swarm = AgentSwarm()
swarm.load_swarm('./models/production_swarm')
swarm.deploy()

# 2. Connect to broker
from src.execution.alpaca_broker import AlpacaBroker
broker = AlpacaBroker(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    paper=True  # Start with paper trading!
)
broker.connect()

# 3. Trading loop
import time
while True:
    try:
        # Get latest market data
        data = fetch_latest_data('AAPL')

        # Create observation
        env = TradingEnv(data=data, initial_balance=broker.get_account_info().portfolio_value)
        obs = env.reset()
        observations = [obs] * len(swarm.agents)

        # Get swarm decision
        action, confidence = swarm.get_swarm_decision(observations)

        # Execute if confidence is high
        if confidence > 0.7:
            execute_action(broker, action, 'AAPL', confidence)

        # Log decision
        log_decision(action, confidence)

        # Wait before next check
        time.sleep(60)  # 1 minute

    except Exception as e:
        logger.error(f"Trading loop error: {e}")
        time.sleep(300)  # Wait 5 minutes on error
```

## 🧪 Backtesting & Evaluation

```python
# Backtest swarm on historical data
from src.backtesting.backtester import Backtester

backtester = Backtester()
results = backtester.run_backtest(
    swarm=swarm,
    data=historical_data,
    initial_capital=100000,
    commission=0.001
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

## 📚 Examples & Tutorials

### Complete Examples
Run the comprehensive example script:

```bash
cd /Users/silasmarkowicz/trading-ai-working
python examples/agent_swarm_example.py
```

This includes:
1. Single agent training
2. Full swarm training
3. Multi-agent cooperative environment
4. Live trading integration guide

### Example Use Cases

1. **Day Trading**: Fast execution agent with tight risk controls
2. **Swing Trading**: Longer-term agents with trend following
3. **Market Making**: Spread capture with liquidity provision
4. **Arbitrage**: Cross-exchange price exploitation

## 🐛 Troubleshooting

### Common Issues

**Issue**: Agent not learning / poor performance
- **Solution**: Increase training timesteps (100k → 500k)
- **Solution**: Adjust learning rate (try 0.0001 or 0.001)
- **Solution**: Check if data has sufficient variation

**Issue**: Swarm decisions too conservative
- **Solution**: Lower min_confidence threshold
- **Solution**: Switch from consensus to voting mode
- **Solution**: Retrain with different reward function

**Issue**: High computational cost
- **Solution**: Reduce lookback_window (20 → 10)
- **Solution**: Use fewer agents
- **Solution**: Decrease total_timesteps for training

### Performance Optimization

```python
# Use GPU for training (if available)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reduce observation space
env = TradingEnv(data=data, lookback_window=10)  # Smaller window

# Parallel training (train multiple agents simultaneously)
from multiprocessing import Pool
with Pool(processes=4) as pool:
    pool.map(train_agent, agent_configs)
```

## 🔐 Security & Risk Management

### API Key Security
- Never commit API keys to code
- Use environment variables: `export ALPACA_API_KEY='your_key'`
- Store in encrypted settings: Dashboard → Settings → API Keys

### Risk Controls
1. **Position Limits**: Max % of portfolio per trade
2. **Stop Losses**: Automatic exit on losses
3. **Daily Loss Limits**: Stop trading after X% daily loss
4. **Exposure Limits**: Max total market exposure

### Monitoring Alerts
Set up alerts for:
- Large drawdowns (> 5%)
- Unusual trading activity
- System errors
- API connection issues

## 📖 Additional Resources

- **OpenAI Gym Documentation**: https://gym.openai.com/
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
- **Reinforcement Learning Intro**: https://spinningup.openai.com/
- **Trading-AI Project**: https://github.com/cpoplaws/trading-ai

## 🚀 Next Steps

1. ✅ Review this guide completely
2. 🧪 Run `examples/agent_swarm_example.py`
3. 🎮 Train your first agent on historical data
4. 📊 Monitor training progress in TensorBoard
5. 🤖 Build your first swarm (3-4 agents)
6. 📈 Backtest swarm performance
7. 📝 Deploy to paper trading
8. 🔴 Gradually move to live trading (with caution!)

## ⚠️ Disclaimer

This system is for educational and research purposes. Trading carries significant risk. Always:
- Start with paper trading
- Never risk money you can't afford to lose
- Understand the algorithms before deploying
- Monitor your agents continuously
- Have human oversight and kill switches
- Comply with all applicable regulations

**Past performance does not guarantee future results.**

---

**Questions?** Check the main README or open an issue on GitHub.

**Happy Trading! 🚀**
