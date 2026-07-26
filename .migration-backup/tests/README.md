# Tests

Automated tests for the Trading AI system.

## Directory Structure

```
tests/
├── unit/              # Unit tests for individual components
└── integration/       # Integration tests for system interactions
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Unit Tests Only
```bash
pytest tests/unit/
```

### Run Integration Tests Only
```bash
pytest tests/integration/
```

### Run Specific Test File
```bash
pytest tests/unit/test_backtest.py
pytest tests/integration/test_system.py
```

---

## Unit Tests

Test individual components in isolation.

- **`test_backtest.py`**: Backtesting engine tests
- **`test_neural_models.py`**: ML/Neural network model tests
- **`test_paper_trading_api.py`**: Paper trading API tests
- **`validate_crypto_transformation.py`**: Crypto data validation

**Run**: `pytest tests/unit/`

---

## Integration Tests

Test system-wide interactions and workflows.

- **`test_integration.py`**: Full system integration tests
- **`test_system.py`**: End-to-end system validation

**Run**: `pytest tests/integration/`

---

## Test Coverage

Current coverage includes:
- ✅ Backtesting engine
- ✅ Strategy execution
- ✅ Paper trading
- ✅ ML models (when dependencies available)
- ✅ Data ingestion
- ✅ Risk management
- ✅ Broker integrations

---

## Writing New Tests

### Unit Test Template
```python
import pytest
from src.strategies.momentum import MomentumStrategy

def test_momentum_signal():
    strategy = MomentumStrategy()
    signal = strategy.generate_signal(price_data)
    assert signal in ['buy', 'sell', 'hold']
```

### Integration Test Template
```python
import pytest
from src.trading_engine import TradingEngine

def test_full_trading_workflow():
    engine = TradingEngine(mode='paper')
    engine.start()
    # Test complete workflow
    assert engine.is_running()
```

---

## CI/CD Integration

Tests run automatically via GitHub Actions on:
- Every push
- Every pull request
- Scheduled daily runs

See `.github/workflows/` for CI configuration.

---

**Note**: Some tests may require API keys or specific dependencies. Configure via environment variables.
