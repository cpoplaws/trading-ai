# 🤖 Agent Swarm - Quick Start Guide

## ✅ What's Been Implemented

Your autonomous trading agent swarm is now fully integrated! Here's what you have:

### 1. **Core Components**
- ✅ Custom OpenAI Gym trading environment (`src/execution/trading_env.py`)
- ✅ Reinforcement learning agent framework (`src/execution/agent_swarm.py`)
- ✅ Multi-agent coordination system (voting, hierarchical, consensus)
- ✅ Dashboard integration with full UI controls
- ✅ Paper and live trading support

### 2. **Agent Types Available**
- **ExecutionAgent** (PPO) - General trading decisions
- **RiskAgent** (SAC) - Risk assessment & protection
- **ArbitrageAgent** (DDPG) - Arbitrage opportunities
- **MarketMakingAgent** (PPO) - Liquidity provision

### 3. **Dependencies Installed**
```bash
✅ stable-baselines3==2.2.1
✅ gym==0.21.0
✅ torch>=2.0.0
✅ tensorboard
```

## 🚀 Quick Start (5 Minutes)

### Option 1: Use the Dashboard (Easiest)

```bash
# Launch the unified dashboard
~/launch_command_center.sh
```

Then in your browser:
1. Navigate to **🤖 Agent Swarm** page
2. Go to **🧠 Training** tab
3. Select symbol (e.g., AAPL)
4. Choose agents to train
5. Click **🧠 Start Training**
6. Monitor progress in **📊 Performance** tab
7. Deploy swarm in **🎮 Control Center**

### Option 2: Command Line Training

```bash
cd /Users/silasmarkowicz/trading-ai-working

# Run complete examples (recommended for first time)
python examples/agent_swarm_example.py
```

This will:
- Train a single agent on AAPL data
- Build and train a full swarm
- Demonstrate multi-agent coordination
- Show live trading integration patterns

### Option 3: Custom Python Script

```python
import sys
sys.path.insert(0, '/Users/silasmarkowicz/trading-ai-working')

from src.execution.trading_env import TradingEnv
from src.execution.agent_swarm import AgentSwarm
import yfinance as yf

# 1. Fetch data
data = yf.Ticker('AAPL').history(period='3mo')
data.columns = [col.lower() for col in data.columns]

# 2. Create swarm
swarm = AgentSwarm(config={
    'coordination_mode': 'voting',
    'min_confidence': 0.6
})

# 3. Add and train agents
for agent_name in ['ExecutionAgent', 'RiskAgent']:
    env = TradingEnv(data=data, initial_balance=100000)
    swarm.add_agent(agent_name, 'PPO', env)
    swarm.train_agent(agent_name, total_timesteps=50000)

# 4. Deploy swarm
swarm.deploy()
print(f"Swarm ready: {swarm.get_status()}")

# 5. Get trading decision
obs = env.reset()
action, confidence = swarm.get_swarm_decision([obs, obs])
print(f"Decision: Action {action} with {confidence:.2%} confidence")
```

## 📊 Monitor Training

### TensorBoard (Recommended)

```bash
# Start TensorBoard to monitor training
tensorboard --logdir=/Users/silasmarkowicz/trading-ai-working/tensorboard_logs
```

Open browser to: `http://localhost:6006`

You'll see:
- Episode rewards over time
- Learning curves
- Policy loss
- Value function loss

### Dashboard UI

Launch dashboard and go to:
- **🤖 Agent Swarm** → **📊 Performance** tab
- View real-time agent metrics
- Compare agent performance
- See cumulative returns

## 📁 Key Files & Locations

```
trading-ai-working/
├── src/execution/
│   ├── trading_env.py          # Custom Gym environment
│   ├── agent_swarm.py          # Agent swarm coordinator
│   ├── alpaca_broker.py        # Broker integration
│   └── order_manager.py        # Order execution
├── src/monitoring/
│   └── unified_dashboard.py    # Dashboard with swarm page
├── examples/
│   └── agent_swarm_example.py  # Complete examples
├── docs/
│   └── AGENT_SWARM_GUIDE.md    # Comprehensive guide
├── models/                      # Trained models saved here
│   ├── execution_agent_demo.zip
│   └── swarm_demo/
└── tensorboard_logs/           # Training logs
```

## 🎯 Training Recommendations

### For Day Trading (Short-term)
```python
config = {
    'coordination_mode': 'voting',
    'min_confidence': 0.7,
    'agent_configs': {
        'ExecutionAgent': {'algorithm': 'PPO', 'learning_rate': 0.001},
        'RiskAgent': {'algorithm': 'SAC', 'learning_rate': 0.0003}
    }
}

env = TradingEnv(
    lookback_window=10,    # Shorter window
    max_position=0.2       # Smaller positions
)
```

### For Swing Trading (Medium-term)
```python
config = {
    'coordination_mode': 'hierarchical',
    'min_confidence': 0.6,
    'agent_configs': {
        'ExecutionAgent': {'algorithm': 'PPO', 'learning_rate': 0.0003},
        'RiskAgent': {'algorithm': 'SAC', 'learning_rate': 0.0003},
        'ArbitrageAgent': {'algorithm': 'DDPG', 'learning_rate': 0.0001}
    }
}

env = TradingEnv(
    lookback_window=20,    # Standard window
    max_position=0.3       # Moderate positions
)
```

### For Conservative Trading
```python
config = {
    'coordination_mode': 'consensus',  # All agents must agree
    'min_confidence': 0.8,             # High confidence required
    'agent_configs': {
        'ExecutionAgent': {'algorithm': 'PPO', 'learning_rate': 0.0001},
        'RiskAgent': {'algorithm': 'SAC', 'learning_rate': 0.0001}
    }
}

env = TradingEnv(
    commission=0.002,      # Higher commission (conservative)
    slippage=0.001,        # Higher slippage assumption
    max_position=0.15      # Smaller positions
)
```

## ⚡ Performance Tips

### Training Speed
- **Reduce lookback_window**: 20 → 10 (faster, less context)
- **Fewer timesteps**: Start with 10k, increase to 100k-500k
- **Use GPU**: Install CUDA-enabled PyTorch
- **Smaller batch sizes**: 64 instead of 128

### Better Results
- **More data**: 6 months or 1 year of training data
- **Feature engineering**: Add custom indicators to TradingEnv
- **Hyperparameter tuning**: Test different learning rates
- **Ensemble methods**: Combine multiple trained swarms

### Memory Optimization
```python
# Reduce observation space
env = TradingEnv(
    lookback_window=10,         # Smaller window
    data=data.iloc[-1000:]      # Only recent data
)

# Limit buffer size for SAC/DDPG
agent_config = {
    'buffer_size': 10000,       # Smaller replay buffer
    'batch_size': 64            # Smaller batches
}
```

## 🔴 Live Trading Checklist

Before deploying to live trading:

- [ ] Train swarm on at least 3-6 months of data
- [ ] Backtest on out-of-sample data
- [ ] Test with paper trading for 2+ weeks
- [ ] Monitor all swarm decisions manually
- [ ] Set up stop-loss and position limits
- [ ] Configure alerts for unusual activity
- [ ] Start with small capital allocation (< 5%)
- [ ] Have kill switch / emergency stop ready
- [ ] Verify API keys are correct (paper vs live)
- [ ] Test all error handling scenarios

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'stable_baselines3'"
```bash
pip install stable-baselines3
```

### "CUDA out of memory"
```python
# Force CPU usage
import torch
device = torch.device("cpu")

# Or reduce batch size
batch_size = 32  # Instead of 128
```

### "Training too slow"
```python
# Reduce training timesteps
total_timesteps = 10000  # Instead of 100000

# Reduce lookback window
lookback_window = 10  # Instead of 20

# Use fewer agents
# Start with just ExecutionAgent
```

### "Poor agent performance"
```python
# Increase training time
total_timesteps = 500000  # More training

# Adjust learning rate
learning_rate = 0.0001  # Lower for stability

# Check reward function
# Review src/execution/trading_env.py:_execute_action()
```

## 📚 Next Steps

1. **Read the comprehensive guide**: `docs/AGENT_SWARM_GUIDE.md`
2. **Run examples**: `python examples/agent_swarm_example.py`
3. **Train your first swarm**: Use dashboard or Python
4. **Monitor with TensorBoard**: Track training progress
5. **Backtest thoroughly**: Test before live deployment
6. **Start paper trading**: Dashboard → Settings → Trading Mode
7. **Gradually go live**: Small capital first

## 🆘 Need Help?

- **Comprehensive Guide**: `docs/AGENT_SWARM_GUIDE.md`
- **Example Scripts**: `examples/agent_swarm_example.py`
- **Dashboard**: `~/launch_command_center.sh` → 🤖 Agent Swarm
- **GitHub Issues**: https://github.com/cpoplaws/trading-ai/issues

## 🎉 You're Ready!

Your autonomous trading swarm is fully set up and ready to train. Start with the examples, experiment with different configurations, and always test thoroughly before live deployment.

**Remember**: Start small, test thoroughly, and never risk more than you can afford to lose.

**Happy Autonomous Trading! 🚀**
