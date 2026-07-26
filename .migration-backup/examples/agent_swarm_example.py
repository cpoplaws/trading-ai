"""
🤖 Agent Swarm Trading Example
Complete example showing how to train and deploy the autonomous agent swarm.
"""
import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution.trading_env import TradingEnv, MultiAgentTradingEnv
from src.execution.agent_swarm import AgentSwarm, TradingAgent

def fetch_training_data(symbol='AAPL', period='6mo'):
    """Fetch historical data for training."""
    print(f"📊 Fetching training data for {symbol}...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    # Standardize column names
    data.columns = [col.lower() for col in data.columns]

    print(f"✅ Fetched {len(data)} data points")
    return data

def train_single_agent_example():
    """Example: Train a single agent"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 1: Training a Single Agent")
    print("="*60)

    # 1. Fetch data
    data = fetch_training_data('AAPL', period='3mo')

    # 2. Create trading environment
    env = TradingEnv(
        data=data,
        initial_balance=100000,
        commission=0.001,
        slippage=0.0005,
        max_position=0.3,
        lookback_window=20
    )

    print(f"📦 Created trading environment:")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.n} actions")
    print(f"   - Initial balance: ${env.initial_balance:,.2f}")

    # 3. Create and train agent
    agent = TradingAgent(
        agent_id="execution_agent",
        env=env,
        algorithm="PPO",  # Options: PPO, SAC, DDPG
        learning_rate=0.0003
    )

    print(f"\n🧠 Training agent for 10,000 timesteps...")
    agent.train(total_timesteps=10000, log_interval=1000)

    # 4. Save model
    model_path = './models/execution_agent_demo.zip'
    agent.save(model_path)
    print(f"💾 Model saved to: {model_path}")

    # 5. Evaluate agent
    print(f"\n📊 Evaluating agent performance...")
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"✅ Evaluation complete!")
    print(f"   - Total reward: {total_reward:.2f}")
    print(f"   - Final portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"   - Total return: {info['total_return']*100:.2f}%")
    print(f"   - Trades executed: {info['trades']}")

def train_swarm_example():
    """Example: Train and deploy agent swarm"""
    print("\n" + "="*60)
    print("🤖 EXAMPLE 2: Training an Agent Swarm")
    print("="*60)

    # 1. Fetch data
    data = fetch_training_data('AAPL', period='3mo')

    # 2. Create swarm configuration
    swarm_config = {
        'coordination_mode': 'voting',  # Options: voting, hierarchical, consensus
        'min_confidence': 0.6,
        'max_agents': 4,
        'agent_configs': {
            'ExecutionAgent': {
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'enabled': True
            },
            'RiskAgent': {
                'algorithm': 'SAC',
                'learning_rate': 0.0003,
                'enabled': True
            },
            'ArbitrageAgent': {
                'algorithm': 'DDPG',
                'learning_rate': 0.0001,
                'enabled': True
            },
            'MarketMakingAgent': {
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'enabled': False  # Disabled for this example
            }
        }
    }

    # 3. Initialize swarm
    swarm = AgentSwarm(config=swarm_config)

    print(f"🤖 Initialized swarm with {len(swarm_config['agent_configs'])} agent types")
    print(f"   - Coordination mode: {swarm_config['coordination_mode']}")
    print(f"   - Min confidence: {swarm_config['min_confidence']}")

    # 4. Train swarm
    print(f"\n🧠 Training swarm (this may take a few minutes)...")

    for agent_name, agent_config in swarm_config['agent_configs'].items():
        if not agent_config['enabled']:
            print(f"⏭️  Skipping {agent_name} (disabled)")
            continue

        print(f"\n📚 Training {agent_name}...")

        # Create environment for this agent
        env = TradingEnv(
            data=data,
            initial_balance=100000,
            commission=0.001,
            slippage=0.0005,
            max_position=0.3,
            lookback_window=20
        )

        # Add agent to swarm
        swarm.add_agent(
            agent_id=agent_name,
            agent_type=agent_config['algorithm'],
            env=env,
            config={
                'learning_rate': agent_config['learning_rate']
            }
        )

        # Train
        swarm.train_agent(
            agent_id=agent_name,
            total_timesteps=10000,
            log_interval=1000
        )

        print(f"✅ {agent_name} training complete")

    # 5. Save swarm
    swarm_path = './models/swarm_demo'
    swarm.save_swarm(swarm_path)
    print(f"\n💾 Swarm saved to: {swarm_path}")

    # 6. Deploy swarm for live decision-making
    print(f"\n🚀 Deploying swarm for trading decisions...")
    swarm.deploy()

    print(f"📊 Swarm status: {swarm.get_status()}")

    # 7. Get swarm decision
    print(f"\n🎯 Getting swarm trading decision...")

    # Create observations for each agent
    env = TradingEnv(data=data, initial_balance=100000, lookback_window=20)
    obs = env.reset()
    observations = [obs] * len([a for a in swarm_config['agent_configs'].values() if a['enabled']])

    # Get consensus decision
    action, confidence = swarm.get_swarm_decision(observations)

    action_names = ['HOLD', 'BUY_SMALL', 'BUY_MEDIUM', 'BUY_LARGE',
                    'SELL_SMALL', 'SELL_MEDIUM', 'SELL_LARGE']

    print(f"✅ Swarm decision:")
    print(f"   - Action: {action_names[action]}")
    print(f"   - Confidence: {confidence:.2%}")

    # 8. Performance report
    print(f"\n📊 Swarm Performance Report:")
    for agent_id, perf in swarm.performance.items():
        print(f"\n   {agent_id}:")
        print(f"      - Decisions: {perf.get('decisions', 0)}")
        print(f"      - Avg Reward: {perf.get('avg_reward', 0):.4f}")

def multi_agent_environment_example():
    """Example: Using multi-agent environment"""
    print("\n" + "="*60)
    print("👥 EXAMPLE 3: Multi-Agent Cooperative Trading")
    print("="*60)

    # 1. Fetch data
    data = fetch_training_data('AAPL', period='2mo')

    # 2. Create multi-agent environment
    n_agents = 3
    env = MultiAgentTradingEnv(
        n_agents=n_agents,
        data=data,
        initial_balance=100000,
        commission=0.001,
        slippage=0.0005,
        max_position=0.3,
        lookback_window=20
    )

    print(f"👥 Created multi-agent environment:")
    print(f"   - Number of agents: {n_agents}")
    print(f"   - Initial balance per agent: ${env.initial_balance:,.2f}")

    # 3. Reset and run simulation
    observations = env.reset()

    print(f"\n🎮 Running multi-agent simulation...")
    total_rewards = [0] * n_agents
    done = False
    step = 0

    while not done and step < 100:  # Limit to 100 steps for demo
        # Each agent takes random action (in real case, use trained agents)
        actions = [env.action_space.sample() for _ in range(n_agents)]

        observations, rewards, done, info = env.step(actions)

        for i, reward in enumerate(rewards):
            total_rewards[i] += reward

        step += 1

        if step % 20 == 0:
            print(f"   Step {step}: Swarm value = ${info['total_portfolio_value']:,.2f}")

    print(f"\n✅ Simulation complete!")
    print(f"   - Steps: {step}")
    print(f"   - Final swarm value: ${info['total_portfolio_value']:,.2f}")
    print(f"   - Swarm return: {info['swarm_return']*100:.2f}%")
    print(f"\n   Agent Performance:")
    for i, (balance, position, reward) in enumerate(zip(info['agent_balances'],
                                                        info['agent_positions'],
                                                        total_rewards)):
        value = balance + (position * data['close'].iloc[-1] if position > 0 else 0)
        print(f"      Agent {i+1}: ${value:,.2f} (return: {(value-100000)/100000*100:.2f}%)")

def live_trading_integration_example():
    """Example: Integrating swarm with live trading"""
    print("\n" + "="*60)
    print("🔴 EXAMPLE 4: Live Trading Integration (Conceptual)")
    print("="*60)

    print("""
This example shows how to integrate the agent swarm with live trading.

📝 IMPORTANT: This is for educational purposes only.
             Always test thoroughly with paper trading first!

Integration Steps:
==================

1. Train your swarm on historical data (examples above)

2. Load the trained swarm:
   ```python
   from src.execution.agent_swarm import AgentSwarm
   swarm = AgentSwarm()
   swarm.load_swarm('./models/swarm_demo')
   swarm.deploy()
   ```

3. Connect to your broker:
   ```python
   from src.execution.alpaca_broker import AlpacaBroker
   broker = AlpacaBroker(api_key='your_key', secret_key='your_secret')
   broker.connect()
   ```

4. Create a trading loop:
   ```python
   import time
   while True:
       # Fetch latest market data
       data = fetch_latest_data(symbol='AAPL')

       # Get swarm decision
       env = TradingEnv(data=data, initial_balance=broker.get_account_info().portfolio_value)
       obs = env.reset()
       observations = [obs] * len(swarm.agents)

       action, confidence = swarm.get_swarm_decision(observations)

       # Execute trade if confidence is high enough
       if confidence > 0.7:
           if action in [1, 2, 3]:  # BUY actions
               broker.place_order(symbol='AAPL', qty=100, side='buy')
           elif action in [4, 5, 6]:  # SELL actions
               broker.place_order(symbol='AAPL', qty=100, side='sell')

       # Wait before next decision
       time.sleep(60)  # Check every minute
   ```

5. Monitor and adjust:
   - Track swarm performance metrics
   - Retrain agents periodically with new data
   - Adjust risk parameters based on market conditions
   - Use the dashboard for real-time monitoring

⚠️  Risk Management:
   - Always start with paper trading
   - Set strict position limits
   - Use stop-loss orders
   - Monitor swarm decisions manually at first
   - Gradually increase capital allocation
    """)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 AUTONOMOUS AGENT SWARM - COMPLETE EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates how to use the agent swarm system.")
    print("Each example can be run independently or as a complete tutorial.\n")

    # Run examples
    try:
        # Example 1: Single agent training
        train_single_agent_example()

        # Example 2: Full swarm training and deployment
        train_swarm_example()

        # Example 3: Multi-agent cooperative environment
        multi_agent_environment_example()

        # Example 4: Live trading integration guide
        live_trading_integration_example()

        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETE!")
        print("="*60)
        print("\n📚 Next Steps:")
        print("   1. Review the trained models in ./models/")
        print("   2. Open the unified dashboard: ~/launch_command_center.sh")
        print("   3. Navigate to '🤖 Agent Swarm' page")
        print("   4. Start with paper trading to test your swarm")
        print("   5. Monitor performance and adjust parameters")
        print("\n🚀 Happy Trading!\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
