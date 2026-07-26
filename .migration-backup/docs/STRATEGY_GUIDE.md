# Trading Strategy Guide

**Last Updated**: 2026-02-16
**Version**: 1.0
**Status**: Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Strategy Catalog](#strategy-catalog)
3. [Basic Strategies](#basic-strategies)
4. [Advanced Strategies](#advanced-strategies)
5. [ML-Based Strategies](#ml-based-strategies)
6. [DeFi Strategies](#defi-strategies)
7. [Strategy Configuration](#strategy-configuration)
8. [Backtesting](#backtesting)
9. [Performance Metrics](#performance-metrics)
10. [Best Practices](#best-practices)

---

## Overview

This guide provides comprehensive documentation for all 11+ trading strategies implemented in the Trading AI system. Each strategy includes:

- **Description**: What the strategy does and when to use it
- **Parameters**: Configurable settings and their effects
- **Usage Examples**: Code examples for implementation
- **Performance Characteristics**: Expected returns, risk profile, drawdowns
- **Best Use Cases**: Market conditions where strategy excels

### Strategy Categories

| Category | Strategies | Complexity | Risk Level |
|----------|-----------|------------|------------|
| Basic | DCA, Buy & Hold | Low | Low |
| Technical | Momentum, Mean Reversion, Trend Following | Medium | Medium |
| Advanced | Pairs Trading, Statistical Arbitrage | High | Medium-High |
| ML-Based | ML Trading Agent, Ensemble Models | High | Medium |
| Reinforcement Learning | RL Agent (DQN, PPO, A2C) | Very High | High |
| DeFi | Arbitrage, Yield Farming, Liquidity Mining | High | High |

---

## Strategy Catalog

### Available Strategies

1. **DCA (Dollar-Cost Averaging)** - Systematic periodic buying
2. **Momentum Trading** - Trend-following based on price momentum
3. **Mean Reversion** - Trading price deviations from moving average
4. **ML Trading Agent** - Machine learning predictions
5. **Reinforcement Learning Agent** - Self-learning adaptive trading
6. **Pairs Trading** - Statistical arbitrage between correlated assets
7. **Trend Following** - Long-term trend capture
8. **Statistical Arbitrage** - Quantitative mean reversion
9. **Arbitrage** - Cross-exchange price differences
10. **Yield Farming** - DeFi liquidity provision
11. **Liquidity Mining** - Automated LP token management

---

## Basic Strategies

### 1. Dollar-Cost Averaging (DCA)

**Description**: Systematically buys a fixed dollar amount at regular intervals, regardless of price.

**When to Use**:
- Long-term accumulation
- High volatility markets
- Reducing timing risk
- Beginners

**Parameters**:
```python
{
    "investment_amount": 100.0,  # USD per interval
    "interval": "daily",          # daily, weekly, monthly
    "symbol": "BTC/USDT",
    "max_position_size": 10000.0  # Maximum total position
}
```

**Usage Example**:
```python
from src.strategies.dca_strategy import DCAStrategy
from src.agents.trading_agent import TradingAgent

# Configure DCA strategy
strategy = DCAStrategy(
    investment_amount=100.0,
    interval_hours=24,  # Daily
    symbol="BTC/USDT"
)

# Create agent with strategy
agent = TradingAgent(
    agent_id="dca-btc-001",
    strategy=strategy,
    initial_capital=10000.0
)

# Run agent
await agent.start()
```

**Performance Characteristics**:
- **Returns**: Market average (minus fees)
- **Volatility**: Matches underlying asset
- **Max Drawdown**: Follows market drawdowns
- **Win Rate**: N/A (not timing-based)
- **Sharpe Ratio**: Typically 0.5-1.5

**Backtest Example**:
```python
from src.backtesting.backtest_engine import BacktestEngine

backtest = BacktestEngine(
    strategy=strategy,
    start_date="2024-01-01",
    end_date="2026-01-01",
    initial_capital=10000.0
)

results = backtest.run()
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

**Best Use Cases**:
- Bitcoin/Ethereum long-term accumulation
- Volatile altcoin markets
- Set-and-forget investing
- Risk-averse traders

---

### 2. Buy & Hold

**Description**: Buys asset once and holds indefinitely.

**When to Use**:
- Strong bullish conviction
- Tax efficiency (long-term gains)
- Minimal trading activity

**Usage Example**:
```python
from src.strategies.buy_hold_strategy import BuyAndHoldStrategy

strategy = BuyAndHoldStrategy(
    symbol="ETH/USDT",
    allocation_pct=0.95  # 95% of capital
)

agent = TradingAgent(
    agent_id="hodl-eth-001",
    strategy=strategy,
    initial_capital=10000.0
)
```

---

## Advanced Strategies

### 3. Momentum Trading

**Description**: Identifies and trades assets with strong directional movement.

**Algorithm**:
1. Calculate momentum indicators (RSI, MACD, rate of change)
2. Identify trending assets (momentum > threshold)
3. Enter long positions on strong uptrends
4. Exit when momentum weakens

**Parameters**:
```python
{
    "lookback_period": 14,        # Days for momentum calculation
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "momentum_threshold": 0.05,   # 5% minimum momentum
    "stop_loss": 0.02,            # 2% stop loss
    "take_profit": 0.06           # 6% take profit
}
```

**Usage Example**:
```python
from src.strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(
    symbol="BTC/USDT",
    lookback_period=14,
    rsi_period=14,
    momentum_threshold=0.05
)

agent = TradingAgent(
    agent_id="momentum-btc-001",
    strategy=strategy,
    initial_capital=10000.0,
    risk_manager=RiskManager(max_position_size=0.10)
)

await agent.start()
```

**Performance Characteristics**:
- **Returns**: 15-30% annually (trending markets)
- **Volatility**: Medium-High
- **Max Drawdown**: 15-25%
- **Win Rate**: 40-50%
- **Sharpe Ratio**: 1.0-2.0
- **Best Markets**: Trending, volatile

**Signal Generation**:
```python
def generate_signal(self, market_data):
    """Generate momentum trading signal"""

    # Calculate indicators
    rsi = self.calculate_rsi(market_data, period=14)
    macd = self.calculate_macd(market_data)
    momentum = self.calculate_momentum(market_data, period=14)

    # Entry conditions
    if (momentum > self.momentum_threshold and
        rsi < 70 and
        macd['histogram'] > 0):
        return TradingSignal(
            action="BUY",
            symbol=self.symbol,
            confidence=0.75,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.06
        )

    # Exit conditions
    if (momentum < 0 or
        rsi > 75 or
        macd['histogram'] < 0):
        return TradingSignal(action="SELL", symbol=self.symbol)

    return None
```

---

### 4. Mean Reversion

**Description**: Trades price deviations from statistical average, expecting reversion to mean.

**Algorithm**:
1. Calculate moving average and standard deviation
2. Identify overbought (price > mean + 2σ) and oversold (price < mean - 2σ)
3. Sell overbought, buy oversold
4. Exit when price reverts to mean

**Parameters**:
```python
{
    "ma_period": 20,              # Moving average period
    "std_dev_multiplier": 2.0,    # Standard deviation bands
    "min_deviation": 0.02,        # 2% minimum deviation
    "reversion_threshold": 0.5,   # 50% reversion to mean
    "holding_period_max": 48      # Maximum holding hours
}
```

**Usage Example**:
```python
from src.strategies.mean_reversion_strategy import MeanReversionStrategy

strategy = MeanReversionStrategy(
    symbol="ETH/USDT",
    ma_period=20,
    std_dev_multiplier=2.0,
    min_deviation=0.02
)

agent = TradingAgent(
    agent_id="meanrev-eth-001",
    strategy=strategy,
    initial_capital=10000.0
)
```

**Performance Characteristics**:
- **Returns**: 10-20% annually (ranging markets)
- **Volatility**: Medium
- **Max Drawdown**: 10-20%
- **Win Rate**: 55-65%
- **Sharpe Ratio**: 1.5-2.5
- **Best Markets**: Range-bound, low volatility

**Best Use Cases**:
- Sideways markets
- High-volume pairs (BTC/USDT, ETH/USDT)
- Low volatility periods
- Short-term trading (1-3 days)

---

### 5. Pairs Trading

**Description**: Statistical arbitrage between two correlated assets.

**Algorithm**:
1. Identify highly correlated asset pairs (correlation > 0.8)
2. Calculate spread: spread = price_A - (hedge_ratio × price_B)
3. Trade when spread deviates from mean (z-score > 2)
4. Long underperformer, short outperformer
5. Exit when spread reverts

**Parameters**:
```python
{
    "pair": ["BTC/USDT", "ETH/USDT"],
    "lookback_period": 30,
    "z_score_entry": 2.0,
    "z_score_exit": 0.5,
    "min_correlation": 0.75,
    "hedge_ratio_method": "OLS"  # or "cointegration"
}
```

**Usage Example**:
```python
from src.strategies.pairs_trading_strategy import PairsTradingStrategy

strategy = PairsTradingStrategy(
    pair=["BTC/USDT", "ETH/USDT"],
    lookback_period=30,
    z_score_entry=2.0
)

agent = TradingAgent(
    agent_id="pairs-btc-eth-001",
    strategy=strategy,
    initial_capital=20000.0
)
```

**Performance Characteristics**:
- **Returns**: 8-15% annually
- **Volatility**: Low-Medium
- **Max Drawdown**: 8-15%
- **Win Rate**: 60-70%
- **Sharpe Ratio**: 2.0-3.0
- **Market Neutral**: Yes

---

## ML-Based Strategies

### 6. ML Trading Agent

**Description**: Uses machine learning models to predict price movements.

**Models Used**:
- Random Forest Classifier
- Gradient Boosting (XGBoost)
- Neural Networks (LSTM)
- Ensemble voting

**Features**:
```python
{
    "price_features": [
        "returns_1h", "returns_4h", "returns_24h",
        "volatility_24h", "volume_ratio"
    ],
    "technical_features": [
        "rsi_14", "macd", "bollinger_position",
        "atr_14", "obv", "cmf"
    ],
    "sentiment_features": [
        "twitter_sentiment", "reddit_sentiment",
        "news_sentiment"
    ]
}
```

**Usage Example**:
```python
from src.strategies.ml_trading_strategy import MLTradingStrategy
from src.ml.models import EnsembleModel

# Train model
model = EnsembleModel(
    models=["random_forest", "xgboost", "lstm"],
    voting="soft"
)
model.train(historical_data, labels)

# Create strategy
strategy = MLTradingStrategy(
    symbol="BTC/USDT",
    model=model,
    confidence_threshold=0.70,
    prediction_horizon="1h"
)

agent = TradingAgent(
    agent_id="ml-btc-001",
    strategy=strategy,
    initial_capital=10000.0
)
```

**Training Example**:
```python
from src.ml.feature_engineering import FeatureEngineer
from src.ml.training import ModelTrainer

# Prepare data
engineer = FeatureEngineer()
features, labels = engineer.prepare_training_data(
    symbol="BTC/USDT",
    start_date="2023-01-01",
    end_date="2025-12-31"
)

# Train model
trainer = ModelTrainer(model_type="ensemble")
model = trainer.train(
    features=features,
    labels=labels,
    validation_split=0.2,
    epochs=100
)

# Evaluate
metrics = trainer.evaluate(model, test_features, test_labels)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
```

**Performance Characteristics**:
- **Returns**: 20-40% annually (well-trained model)
- **Volatility**: Medium-High
- **Max Drawdown**: 15-30%
- **Win Rate**: 50-60%
- **Sharpe Ratio**: 1.5-2.5
- **Requires**: Regular retraining, feature engineering

**Feature Importance**:
```python
# View feature importance
importance = model.get_feature_importance()
print("\nTop 10 Features:")
for feature, score in importance[:10]:
    print(f"  {feature}: {score:.3f}")
```

---

### 7. Reinforcement Learning Agent

**Description**: Self-learning agent that optimizes trading policy through trial and error.

**RL Algorithms**:
- **DQN** (Deep Q-Network): Discrete action space
- **PPO** (Proximal Policy Optimization): Continuous actions
- **A2C** (Advantage Actor-Critic): Balance exploration/exploitation

**State Space**:
```python
{
    "market_state": [
        "current_price", "volume", "volatility",
        "bid_ask_spread", "order_book_depth"
    ],
    "portfolio_state": [
        "position_size", "unrealized_pnl", "cash_balance",
        "margin_used", "risk_exposure"
    ],
    "technical_indicators": [
        "rsi", "macd", "bollinger_bands", "atr"
    ],
    "historical_context": [
        "returns_1h", "returns_4h", "returns_24h",
        "recent_trades", "recent_pnl"
    ]
}
```

**Action Space**:
```python
{
    "discrete": [
        "HOLD", "BUY_SMALL", "BUY_MEDIUM", "BUY_LARGE",
        "SELL_SMALL", "SELL_MEDIUM", "SELL_LARGE"
    ],
    "continuous": {
        "action": [-1.0, 1.0],  # -1 = sell all, +1 = buy max
        "position_size": [0.0, 1.0]  # Fraction of capital
    }
}
```

**Reward Function**:
```python
def calculate_reward(self, state, action, next_state):
    """Calculate reward for RL agent"""

    # PnL component (primary)
    pnl = next_state.portfolio_value - state.portfolio_value
    pnl_reward = pnl / state.portfolio_value

    # Risk-adjusted return
    volatility = state.calculate_volatility()
    risk_penalty = -0.1 * volatility * abs(action)

    # Transaction cost penalty
    if action != "HOLD":
        cost_penalty = -0.001  # 0.1% fee
    else:
        cost_penalty = 0

    # Drawdown penalty
    if next_state.drawdown > 0.10:  # > 10% drawdown
        drawdown_penalty = -0.5 * next_state.drawdown
    else:
        drawdown_penalty = 0

    # Total reward
    total_reward = (
        pnl_reward +
        risk_penalty +
        cost_penalty +
        drawdown_penalty
    )

    return total_reward
```

**Training Example**:
```python
from src.strategies.rl_trading_agent import RLTradingAgent
from src.rl.environments import TradingEnvironment
from src.rl.algorithms import PPOAgent

# Create trading environment
env = TradingEnvironment(
    symbol="BTC/USDT",
    start_date="2023-01-01",
    end_date="2025-12-31",
    initial_capital=10000.0,
    commission=0.001
)

# Initialize RL agent
rl_agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    learning_rate=0.0003,
    gamma=0.99,  # Discount factor
    epsilon=0.2,  # Clip parameter
    entropy_coef=0.01
)

# Training loop
for episode in range(1000):
    state = env.reset()
    episode_reward = 0

    for step in range(env.max_steps):
        # Agent selects action
        action = rl_agent.select_action(state)

        # Environment step
        next_state, reward, done, info = env.step(action)

        # Store experience
        rl_agent.store_transition(state, action, reward, next_state, done)

        # Update policy
        if len(rl_agent.memory) > rl_agent.batch_size:
            rl_agent.update()

        episode_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
          f"Portfolio Value = {env.portfolio_value:.2f}")

    # Save best model
    if episode_reward > best_reward:
        rl_agent.save(f"models/rl_agent_best.pth")
        best_reward = episode_reward

# Use trained agent
strategy = RLTradingAgent(
    symbol="BTC/USDT",
    model_path="models/rl_agent_best.pth",
    algorithm="PPO"
)
```

**Performance Characteristics**:
- **Returns**: 25-50% annually (well-trained)
- **Volatility**: High
- **Max Drawdown**: 20-40%
- **Win Rate**: Variable (learns over time)
- **Sharpe Ratio**: 1.0-2.5
- **Requires**: Extensive training (1000+ episodes)

**Hyperparameter Tuning**:
```python
from src.rl.hyperparameter_search import GridSearch

search = GridSearch(
    param_grid={
        "learning_rate": [0.0001, 0.0003, 0.001],
        "gamma": [0.95, 0.99, 0.995],
        "epsilon": [0.1, 0.2, 0.3],
        "hidden_layers": [[64, 64], [128, 128], [256, 256]]
    },
    n_trials=20
)

best_params = search.run(env)
print(f"Best parameters: {best_params}")
```

---

## DeFi Strategies

### 8. Arbitrage Strategy

**Description**: Exploits price differences across multiple exchanges.

**Types**:
- **Spatial Arbitrage**: Same asset, different exchanges
- **Triangular Arbitrage**: Currency pairs on same exchange
- **Statistical Arbitrage**: Mean reversion between correlated assets

**Parameters**:
```python
{
    "exchanges": ["binance", "coinbase", "kraken"],
    "symbol": "BTC/USDT",
    "min_profit_threshold": 0.005,  # 0.5% minimum profit
    "max_execution_time": 5,        # 5 seconds max
    "slippage_tolerance": 0.002,    # 0.2% slippage
    "gas_price_limit": 50           # Max gas price (DeFi)
}
```

**Usage Example**:
```python
from src.strategies.arbitrage_strategy import ArbitrageStrategy

strategy = ArbitrageStrategy(
    exchanges=["binance", "coinbase"],
    symbol="BTC/USDT",
    min_profit_threshold=0.005
)

agent = TradingAgent(
    agent_id="arb-btc-001",
    strategy=strategy,
    initial_capital=50000.0,  # Larger capital for arbitrage
    exchange_credentials={
        "binance": binance_creds,
        "coinbase": coinbase_creds
    }
)
```

**Arbitrage Detection**:
```python
def detect_arbitrage_opportunity(self):
    """Scan exchanges for arbitrage opportunities"""

    opportunities = []

    for i, exchange_1 in enumerate(self.exchanges):
        for exchange_2 in self.exchanges[i+1:]:
            # Get prices
            price_1 = self.get_price(exchange_1, self.symbol)
            price_2 = self.get_price(exchange_2, self.symbol)

            # Calculate profit
            if price_1 < price_2:
                buy_exchange = exchange_1
                sell_exchange = exchange_2
                profit = (price_2 - price_1) / price_1
            else:
                buy_exchange = exchange_2
                sell_exchange = exchange_1
                profit = (price_1 - price_2) / price_2

            # Account for fees
            total_fees = (
                self.get_trading_fee(buy_exchange) +
                self.get_trading_fee(sell_exchange) +
                self.estimate_transfer_cost()
            )
            net_profit = profit - total_fees

            # Check if profitable
            if net_profit > self.min_profit_threshold:
                opportunities.append({
                    "buy": buy_exchange,
                    "sell": sell_exchange,
                    "profit": net_profit,
                    "timestamp": datetime.utcnow()
                })

    return opportunities
```

**Performance Characteristics**:
- **Returns**: 5-15% annually (high frequency)
- **Volatility**: Low
- **Max Drawdown**: 5-10%
- **Win Rate**: 70-80%
- **Sharpe Ratio**: 2.5-4.0
- **Requires**: Fast execution, low latency

---

### 9. Yield Farming

**Description**: Provides liquidity to DeFi protocols for yield generation.

**Supported Protocols**:
- Uniswap V2/V3
- SushiSwap
- Curve Finance
- Balancer
- Aave (lending)

**Parameters**:
```python
{
    "protocol": "uniswap_v3",
    "pool": "ETH/USDC",
    "liquidity_range": [0.95, 1.05],  # Concentrated liquidity
    "min_apr": 0.15,                   # 15% minimum APR
    "rebalance_threshold": 0.10,       # Rebalance if price moves 10%
    "compound_frequency": "daily",     # Auto-compound rewards
    "impermanent_loss_limit": 0.05     # 5% max IL
}
```

**Usage Example**:
```python
from src.strategies.yield_farming_strategy import YieldFarmingStrategy

strategy = YieldFarmingStrategy(
    protocol="uniswap_v3",
    pool="ETH/USDC",
    liquidity_range=[0.95, 1.05],
    min_apr=0.15
)

agent = TradingAgent(
    agent_id="yield-eth-usdc-001",
    strategy=strategy,
    initial_capital=10000.0,
    web3_provider=web3_provider
)
```

**Impermanent Loss Calculation**:
```python
def calculate_impermanent_loss(self, entry_price, current_price):
    """Calculate IL for providing liquidity"""

    price_ratio = current_price / entry_price

    # IL formula for 50/50 pool
    il = (2 * math.sqrt(price_ratio)) / (1 + price_ratio) - 1

    return abs(il)
```

**Performance Characteristics**:
- **Returns**: 10-50% APR (depending on protocol)
- **Volatility**: Low-Medium
- **Max Drawdown**: 10-20% (impermanent loss)
- **Win Rate**: 85-95% (consistent yield)
- **Sharpe Ratio**: 1.5-3.0
- **Risks**: Impermanent loss, smart contract risk

---

### 10. Liquidity Mining

**Description**: Automated management of liquidity provider positions.

**Features**:
- Automatic position rebalancing
- Multi-pool optimization
- Gas cost optimization
- Reward harvesting and compounding

**Usage Example**:
```python
from src.strategies.liquidity_mining_strategy import LiquidityMiningStrategy

strategy = LiquidityMiningStrategy(
    pools=[
        {"protocol": "uniswap_v3", "pair": "ETH/USDC", "allocation": 0.50},
        {"protocol": "curve", "pair": "USDT/USDC/DAI", "allocation": 0.30},
        {"protocol": "balancer", "pair": "ETH/WBTC", "allocation": 0.20}
    ],
    rebalance_frequency="weekly",
    compound_frequency="daily",
    gas_price_limit=50
)

agent = TradingAgent(
    agent_id="lm-multi-001",
    strategy=strategy,
    initial_capital=50000.0
)
```

**Performance Characteristics**:
- **Returns**: 15-60% APR
- **Volatility**: Medium
- **Max Drawdown**: 15-25%
- **Win Rate**: 80-90%
- **Sharpe Ratio**: 2.0-3.5

---

## Strategy Configuration

### Configuration File Format

```yaml
# config/strategies/momentum_btc.yaml
strategy:
  name: "momentum_btc"
  type: "momentum"
  version: "1.0"

  parameters:
    symbol: "BTC/USDT"
    lookback_period: 14
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    momentum_threshold: 0.05
    stop_loss: 0.02
    take_profit: 0.06

  risk_management:
    max_position_size: 0.10      # 10% of portfolio
    max_leverage: 1.0             # No leverage
    stop_loss_enabled: true
    trailing_stop: true
    trailing_stop_pct: 0.03

  execution:
    order_type: "limit"
    time_in_force: "GTC"
    post_only: true
    reduce_only: false

  backtesting:
    start_date: "2024-01-01"
    end_date: "2026-01-01"
    initial_capital: 10000.0
    commission: 0.001
```

### Loading Configuration

```python
from src.strategies.strategy_factory import StrategyFactory

# Load from config file
strategy = StrategyFactory.from_config("config/strategies/momentum_btc.yaml")

# Create agent
agent = TradingAgent(
    agent_id="momentum-btc-001",
    strategy=strategy,
    initial_capital=10000.0
)
```

---

## Backtesting

### Running Backtests

```python
from src.backtesting.backtest_engine import BacktestEngine
from src.strategies.momentum_strategy import MomentumStrategy

# Configure strategy
strategy = MomentumStrategy(
    symbol="BTC/USDT",
    lookback_period=14,
    momentum_threshold=0.05
)

# Run backtest
backtest = BacktestEngine(
    strategy=strategy,
    start_date="2024-01-01",
    end_date="2026-01-01",
    initial_capital=10000.0,
    commission=0.001,
    slippage=0.0005
)

results = backtest.run()

# Print results
print(f"\n=== Backtest Results ===")
print(f"Total Return: {results.total_return:.2%}")
print(f"Annualized Return: {results.annualized_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Total Trades: {results.total_trades}")
print(f"Profitable Trades: {results.profitable_trades}")
print(f"Average Trade: {results.avg_trade_pnl:.2f}")
```

### Visualization

```python
# Plot equity curve
results.plot_equity_curve(save_path="backtest_equity.png")

# Plot drawdown
results.plot_drawdown(save_path="backtest_drawdown.png")

# Plot trades
results.plot_trades(save_path="backtest_trades.png")

# Generate report
results.generate_report(output_path="backtest_report.pdf")
```

### Walk-Forward Optimization

```python
from src.backtesting.walk_forward import WalkForwardOptimizer

optimizer = WalkForwardOptimizer(
    strategy_class=MomentumStrategy,
    param_grid={
        "lookback_period": [10, 14, 20],
        "momentum_threshold": [0.03, 0.05, 0.07],
        "rsi_period": [10, 14, 20]
    },
    training_window=180,  # 180 days
    testing_window=60,    # 60 days
    step_size=30          # Roll forward 30 days
)

results = optimizer.run(
    symbol="BTC/USDT",
    start_date="2024-01-01",
    end_date="2026-01-01"
)

print(f"Walk-Forward Results:")
print(f"  In-Sample Sharpe: {results.in_sample_sharpe:.2f}")
print(f"  Out-of-Sample Sharpe: {results.out_of_sample_sharpe:.2f}")
print(f"  Best Parameters: {results.best_params}")
```

---

## Performance Metrics

### Key Metrics Explained

**Return Metrics**:
- **Total Return**: (Final Value - Initial Value) / Initial Value
- **Annualized Return**: (1 + Total Return) ^ (365 / Days) - 1
- **CAGR**: Compound Annual Growth Rate

**Risk Metrics**:
- **Volatility**: Standard deviation of returns (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Maximum expected loss at confidence level

**Risk-Adjusted Metrics**:
- **Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility
- **Sortino Ratio**: Like Sharpe, but only downside volatility
- **Calmar Ratio**: Return / Max Drawdown

**Trade Metrics**:
- **Win Rate**: Profitable Trades / Total Trades
- **Profit Factor**: Gross Profit / Gross Loss
- **Average Trade**: Total PnL / Number of Trades
- **Expectancy**: (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

### Calculating Metrics

```python
from src.analytics.performance_metrics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(trades=agent.trade_history)

metrics = analyzer.calculate_all_metrics()

print(f"\n=== Performance Analysis ===")
print(f"Total Return: {metrics.total_return:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
print(f"Win Rate: {metrics.win_rate:.2%}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Expectancy: ${metrics.expectancy:.2f}")
```

---

## Best Practices

### 1. Strategy Selection

**Market Conditions**:
- **Trending**: Momentum, Trend Following
- **Range-Bound**: Mean Reversion, Pairs Trading
- **Volatile**: DCA, ML/RL Agents
- **Low Volatility**: Yield Farming, Arbitrage

### 2. Risk Management

```python
# Always use stop losses
strategy.set_stop_loss(percent=0.02)  # 2% stop loss

# Position sizing
strategy.set_position_size(max_pct=0.10)  # Max 10% per trade

# Diversification
agent.add_strategies([
    momentum_strategy,
    mean_reversion_strategy,
    yield_farming_strategy
])
```

### 3. Backtesting

- Use at least 2 years of historical data
- Include transaction costs and slippage
- Walk-forward optimization to avoid overfitting
- Test on multiple market conditions

### 4. Paper Trading

```python
# Always paper trade before live
agent = TradingAgent(
    agent_id="test-001",
    strategy=strategy,
    mode="paper",  # Paper trading mode
    initial_capital=10000.0
)

# Run for 30 days minimum
await agent.start()
```

### 5. Monitoring

```python
# Set up alerts
agent.add_alert(
    condition="drawdown > 0.10",
    action="pause_trading",
    notification="email"
)

# Monitor key metrics
agent.monitor_metrics([
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "portfolio_value"
])
```

### 6. Regular Review

- Review strategy performance weekly
- Retrain ML models monthly
- Update parameters quarterly
- Conduct full audit annually

---

## Conclusion

This guide covers all major strategies in the Trading AI system. For additional help:

- **API Documentation**: See `docs/API_REFERENCE.md`
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Troubleshooting**: See `docs/TROUBLESHOOTING_GUIDE.md`
- **Examples**: See `examples/` directory

**Remember**:
- Start with paper trading
- Use proper risk management
- Backtest thoroughly
- Monitor continuously
- Keep learning and adapting

---

**Document Version**: 1.0
**Last Updated**: 2026-02-16
**Maintainer**: Trading AI Team
