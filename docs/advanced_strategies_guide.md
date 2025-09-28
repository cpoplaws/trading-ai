# Advanced Trading Strategies Documentation

## Overview

The trading AI system now includes comprehensive advanced strategies that combine multiple analytical approaches for sophisticated trading decisions. This system integrates:

1. **Portfolio Optimization** - Kelly Criterion & Mean Reversion
2. **Sentiment Analysis** - Multi-source sentiment aggregation
3. **Options Strategies** - Complex options analysis
4. **Enhanced ML Models** - Ensemble methods & time series forecasting
5. **Multi-timeframe Analysis** - Cross-timeframe signal generation

## Architecture

```
src/advanced_strategies/
├── __init__.py                   # Main integration module
├── portfolio_optimizer.py       # Kelly Criterion & portfolio optimization
├── sentiment_analyzer.py        # Multi-source sentiment analysis
├── options_strategies.py        # Options trading strategies
├── enhanced_ml_models.py        # Advanced ML models
└── multi_timeframe.py          # Multi-timeframe analysis
```

## Core Components

### 1. Portfolio Optimizer (`portfolio_optimizer.py`)

**Features:**

- Kelly Criterion position sizing
- Mean reversion detection
- Modern Portfolio Theory optimization
- Risk parity weighting
- Portfolio-level recommendations

**Key Methods:**

```python
optimizer = PortfolioOptimizer(['AAPL', 'MSFT', 'GOOGL'])

# Calculate Kelly Criterion position size
kelly_size = optimizer.calculate_kelly_criterion(returns, confidence=0.7)

# Detect mean reversion opportunities
mean_reversion = optimizer.detect_mean_reversion_opportunities(prices)

# Generate portfolio recommendations
recommendations = optimizer.generate_portfolio_recommendations(prices_data, signals_data)
```

**Example Output:**

```python
{
    'recommendations': {
        'AAPL': {
            'signal': 'BUY',
            'confidence': 0.75,
            'recommended_weight': 0.25,
            'kelly_size': 0.15,
            'z_score': -1.8
        }
    },
    'portfolio_summary': {
        'buy_signals': 2,
        'expected_return': 0.12,
        'sharpe_ratio': 1.45
    }
}
```

### 2. Sentiment Analyzer (`sentiment_analyzer.py`)

**Features:**

- Multi-source sentiment (Twitter, Reddit, News)
- Financial lexicon-based analysis
- Source consensus calculation
- Sentiment trend tracking
- Trading signal generation

**Key Methods:**

```python
analyzer = SentimentAnalyzer(api_keys={'twitter': 'key', 'reddit': 'key'})

# Get aggregated sentiment from all sources
sentiment = analyzer.aggregate_sentiment_signals(
    'AAPL',
    include_sources=['twitter', 'reddit', 'news']
)
```

**Example Output:**

```python
{
    'overall_sentiment': 0.35,
    'confidence': 0.8,
    'signal': 'BUY',
    'source_consensus': 0.75,
    'sources': {
        'twitter': {'sentiment_score': 0.4, 'confidence': 0.6},
        'reddit': {'sentiment_score': 0.3, 'confidence': 0.7},
        'news': {'sentiment_score': 0.35, 'confidence': 0.9}
    }
}
```

### 3. Options Strategies (`options_strategies.py`)

**Features:**

- Black-Scholes pricing and Greeks
- Bull/Bear spreads analysis
- Long straddles and strangles
- Iron condors
- Strategy screening and recommendations

**Key Methods:**

```python
options = OptionsStrategy(risk_free_rate=0.05)

# Analyze bull call spread
spread = options.bull_call_spread(150, 145, 155, 30/365, 0.25)

# Analyze long straddle
straddle = options.long_straddle(150, 150, 30/365, 0.25)

# Screen for opportunities
opportunities = options.screen_options_opportunities(
    'AAPL', 150, 0.25, 'bullish'
)
```

**Example Output:**

```python
{
    'strategy': 'bull_call_spread',
    'net_debit': 507.16,
    'max_profit': 492.84,
    'breakeven': 150.07,
    'probability_of_profit': 0.506,
    'recommendation_score': 0.85
}
```

### 4. Enhanced ML Models (`enhanced_ml_models.py`)

**Features:**

- Ensemble methods (Random Forest, Gradient Boosting, SVR)
- Prophet time series forecasting
- ARIMA-GARCH volatility modeling
- Advanced feature engineering
- Automated feature selection

**Key Methods:**

```python
ml_models = EnhancedMLModels()

# Generate comprehensive ML signals
signals = ml_models.generate_ml_signals(
    price_data,
    models_to_use=['ensemble', 'prophet', 'arima_garch']
)

# Create ensemble model
ensemble = ml_models.create_ensemble_model(features, target)
```

**Example Output:**

```python
{
    'final_signal': 'BUY',
    'confidence': 0.73,
    'expected_return': 0.025,
    'individual_signals': {
        'ensemble': {'signal': 'BUY', 'model_r2': 0.42},
        'prophet': {'signal': 'BUY', 'trend': 'increasing'}
    }
}
```

### 5. Multi-timeframe Analysis (`multi_timeframe.py`)

**Features:**

- Multiple timeframe data aggregation
- Cross-timeframe signal consensus
- Timeframe-specific feature engineering
- Weighted signal combination

**Key Methods:**

```python
mtf = MultiTimeframeAnalyzer('AAPL')

# Generate signals across timeframes
signals = mtf.generate_multi_timeframe_signals('AAPL', {
    '1min': minute_data,
    '5min': five_min_data,
    '1h': hourly_data,
    '1d': daily_data
})
```

## Integration Module (`__init__.py`)

The main integration module combines all strategies:

```python
from advanced_strategies import AdvancedTradingStrategies

# Initialize with symbols
strategies = AdvancedTradingStrategies(['AAPL', 'MSFT', 'GOOGL'])

# Get comprehensive signals
signals = strategies.get_comprehensive_signals(
    'AAPL',
    market_data,
    current_price=150.0,
    market_outlook='bullish'
)

# Generate portfolio dashboard
dashboard = strategies.get_portfolio_dashboard(market_data, current_prices)
```

## Signal Aggregation

The system uses weighted voting to combine signals from different strategies:

```python
strategy_weights = {
    'ml_models': 0.30,          # Highest weight - data-driven
    'multi_timeframe': 0.25,    # Cross-timeframe validation
    'sentiment': 0.20,          # Market psychology
    'portfolio_optimization': 0.15,  # Risk management
    'options': 0.10             # Derivatives insights
}
```

## Output Structure

### Comprehensive Signal Output

```python
{
    'symbol': 'AAPL',
    'timestamp': '2024-01-15T10:30:00',
    'current_price': 150.25,
    'individual_signals': {
        'ml_models': {...},
        'sentiment': {...},
        'multi_timeframe': {...},
        'portfolio_optimization': {...},
        'options': {...}
    },
    'aggregated_signal': {
        'signal': 'BUY',
        'confidence': 0.78,
        'expected_return': 0.035,
        'consensus': 0.85
    },
    'final_recommendations': {
        'primary_action': 'BUY',
        'position_sizing': 0.15,
        'stop_loss': 142.50,
        'take_profit': 158.75,
        'risk_assessment': 'MEDIUM',
        'rationale': [...]
    }
}
```

### Portfolio Dashboard Output

```python
{
    'portfolio_summary': {
        'total_symbols': 3,
        'buy_signals': 2,
        'sell_signals': 0,
        'hold_signals': 1,
        'average_confidence': 0.75,
        'top_opportunities': [...],
        'risk_assessment': 'MEDIUM'
    },
    'symbol_signals': {
        'AAPL': {...},
        'MSFT': {...},
        'GOOGL': {...}
    }
}
```

## Usage Examples

### Basic Usage

```python
# Initialize system
from advanced_strategies import AdvancedTradingStrategies

symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
strategies = AdvancedTradingStrategies(symbols)

# Get signals for a single symbol
signals = strategies.get_comprehensive_signals(
    'AAPL',
    market_data['AAPL'],
    current_price=150.0
)

print(f"Signal: {signals['aggregated_signal']['signal']}")
print(f"Confidence: {signals['aggregated_signal']['confidence']}")
```

### Portfolio Analysis

```python
# Get portfolio-wide analysis
dashboard = strategies.get_portfolio_dashboard(
    market_data,  # Dict of symbol -> timeframe -> data
    current_prices  # Dict of symbol -> current price
)

# Display top opportunities
for opp in dashboard['portfolio_summary']['top_opportunities']:
    print(f"{opp['symbol']}: {opp['signal']} (score: {opp['score']:.3f})")
```

### Individual Strategy Components

```python
# Use individual components
from advanced_strategies.portfolio_optimizer import PortfolioOptimizer
from advanced_strategies.sentiment_analyzer import SentimentAnalyzer

# Portfolio optimization
optimizer = PortfolioOptimizer(symbols)
portfolio_rec = optimizer.generate_portfolio_recommendations(prices, signals)

# Sentiment analysis
sentiment = SentimentAnalyzer()
sentiment_signals = sentiment.aggregate_sentiment_signals('AAPL')
```

## Configuration

### Environment Variables

Add to your `.env` file:

```
# Optional API keys for enhanced functionality
TWITTER_API_KEY=your_twitter_key
REDDIT_API_KEY=your_reddit_key
NEWS_API_KEY=your_news_key

# ML Model parameters
ML_MODEL_RETRAIN_FREQUENCY=daily
ENSEMBLE_N_ESTIMATORS=100
FEATURE_SELECTION_METHOD=rfe
```

### Custom Strategy Weights

```python
# Customize strategy weights
strategies.strategy_weights = {
    'ml_models': 0.40,          # Increase ML weight
    'sentiment': 0.30,          # Increase sentiment weight
    'multi_timeframe': 0.20,
    'portfolio_optimization': 0.05,
    'options': 0.05
}
```

## Dependencies

Core dependencies are already included in `requirements.txt`. Optional dependencies for enhanced functionality:

```bash
# Time series forecasting
pip install prophet

# Volatility modeling
pip install arch

# Statistical models
pip install statsmodels

# Advanced TA
pip install ta-lib
```

## Performance Considerations

1. **Caching**: Multi-timeframe analyzers are cached per symbol
2. **Rate Limiting**: Sentiment analysis includes rate limiting
3. **Memory Management**: Large datasets are processed in chunks
4. **Parallel Processing**: Individual strategies can run in parallel

## Error Handling

The system is designed to be robust:

- Individual strategy failures don't break the entire system
- Graceful degradation when optional dependencies are missing
- Comprehensive logging for debugging
- Default fallback values for missing data

## Testing

Run the demo to verify all components:

```bash
cd /workspaces/trading-ai
python -c "
from advanced_strategies import AdvancedTradingStrategies
# Test all components...
"
```

## Next Steps

1. **Integration**: Connect to your existing trading system
2. **Backtesting**: Test strategies on historical data
3. **Paper Trading**: Implement signals in paper trading
4. **Monitoring**: Set up performance monitoring
5. **Optimization**: Fine-tune strategy weights based on performance

The advanced strategies system provides a comprehensive framework for sophisticated trading decisions, combining multiple analytical approaches with robust risk management.
