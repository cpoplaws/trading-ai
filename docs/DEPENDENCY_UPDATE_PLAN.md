# Dependency Update Plan

**Date**: February 16, 2026
**Current Python**: 3.9.6
**Target Python**: 3.12.x
**Status**: Planning Phase

## Executive Summary

This document outlines the comprehensive dependency update strategy for the Trading AI system, including Python version upgrade from 3.9 to 3.12 and updating all major dependencies to their latest stable versions.

## Current State

### Python Version
- **Current**: Python 3.9.6 (released July 2021)
- **EOL**: October 2025 (already past end-of-life)
- **Security Support**: No longer receiving security updates

### Major Dependencies Status

| Package | Current | Latest Stable | Update Type | Breaking Changes |
|---------|---------|---------------|-------------|------------------|
| Python | 3.9.6 | 3.12.8 | Major | Yes - Multiple |
| numpy | 2.0.2 | 2.2.4 | Minor | Minimal |
| pandas | 2.3.3 | 2.3.3 | None | N/A |
| tensorflow | 2.20.0 | 2.20.0 | None | N/A |
| keras | 3.13.1 | 3.13.1 | None | N/A (just updated) |
| scikit-learn | 1.6.1 | 1.6.1 | None | N/A |
| requests | 2.32.5 | 2.32.5 | None | N/A |
| urllib3 | 2.6.3 | 2.6.3 | None | N/A (just updated) |
| pillow | 12.1.1 | 12.1.1 | None | N/A (just updated) |
| web3 | 7.14.1 | 7.14.1 | None | N/A |
| aiohttp | 3.13.3 | 3.13.3 | None | N/A |
| streamlit | 1.50.0 | 1.50.0 | None | N/A |
| plotly | 6.5.2 | 6.5.2 | None | N/A |
| fastapi | Not installed | 0.115.6 | New | N/A |
| uvicorn | Not installed | 0.34.0 | New | N/A |
| pydantic | 2.12.5 | 2.12.5 | None | N/A |

## Python 3.12 Migration

### Benefits of Python 3.12

1. **Performance Improvements**
   - 5-10% faster than Python 3.9
   - Improved memory efficiency
   - Better asyncio performance
   - Optimized comprehensions

2. **New Features**
   - PEP 701: Syntactic formalization of f-strings
   - PEP 698: Override decorator for static typing
   - PEP 688: Making the buffer protocol accessible in Python
   - Improved error messages

3. **Security**
   - Active security support until October 2028
   - Modern cryptographic libraries
   - Enhanced SSL/TLS support

### Breaking Changes in Python 3.12

#### Removed Modules and Features
- `distutils` (removed) → Use `setuptools` instead
- `imp` module (removed) → Use `importlib`
- `asyncore` and `asynchat` (removed) → Use `asyncio`
- `unittest.TestCase.assertEquals()` (removed) → Use `assertEqual()`

#### Behavior Changes
- `calendar.January` now equals 1 (was 0)
- `sqlite3` connection autocommit changed
- `shutil.rmtree()` now ignores errors by default differently
- `pathlib.Path.glob()` behavior refined

#### Dependency Compatibility
All major dependencies support Python 3.12:
- ✅ TensorFlow 2.20.0+ supports Python 3.12
- ✅ NumPy 2.0+ supports Python 3.12
- ✅ Pandas 2.0+ supports Python 3.12
- ✅ All other dependencies verified compatible

## Update Strategy

### Phase 1: Preparation (Week 1)

#### 1.1 Code Audit
```bash
# Find uses of removed features
grep -r "distutils" src/
grep -r "imp\." src/
grep -r "assertEquals" tests/
grep -r "asyncore" src/
```

#### 1.2 Create Test Environment
```bash
# Install Python 3.12
brew install python@3.12  # macOS
# or
sudo apt-get install python3.12 python3.12-venv  # Linux

# Create test virtual environment
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-secure.txt
```

#### 1.3 Static Analysis
```bash
# Check for compatibility issues
pip install vermin
vermin --target=3.12 src/

# Check for deprecated features
pip install flake8 flake8-bugbear
flake8 src/ --select=B
```

### Phase 2: Dependency Updates (Week 2)

#### 2.1 Update Core Dependencies

**NumPy 2.0.2 → 2.2.4**
```bash
pip install --upgrade "numpy>=2.2.0,<3.0"
```
- **Breaking Changes**: Minimal, mostly performance improvements
- **Testing Required**: Array operations, data processing
- **Risk**: LOW

**Pandas 2.3.3** (Already latest)
- No update needed

**SciKit-Learn 1.6.1** (Already latest)
- No update needed

#### 2.2 Add Missing Dependencies

**FastAPI + Uvicorn** (for REST API)
```bash
pip install "fastapi>=0.115.0,<1.0"
pip install "uvicorn[standard]>=0.34.0,<1.0"
```

**Additional Utilities**
```bash
pip install "python-multipart>=0.0.9"  # For file uploads
pip install "python-jose[cryptography]>=3.3.0"  # For JWT
pip install "passlib[bcrypt]>=1.7.4"  # For password hashing
```

#### 2.3 Update Development Dependencies

```bash
pip install --upgrade "pytest>=8.0.0,<9.0"
pip install --upgrade "pytest-asyncio>=0.23.0,<1.0"
pip install --upgrade "pytest-cov>=4.1.0,<5.0"
pip install --upgrade "black>=24.0.0,<25.0"
pip install --upgrade "ruff>=0.6.0,<1.0"
pip install --upgrade "mypy>=1.11.0,<2.0"
```

### Phase 3: Code Updates (Week 2-3)

#### 3.1 Replace Deprecated Imports

**Before:**
```python
from distutils.version import LooseVersion
```

**After:**
```python
from packaging.version import Version
```

#### 3.2 Update Test Assertions

**Before:**
```python
self.assertEquals(result, expected)
```

**After:**
```python
self.assertEqual(result, expected)
```

#### 3.3 Update Type Hints (Optional but Recommended)

**Before:**
```python
from typing import List, Dict, Optional

def process_trades(trades: List[Dict]) -> Optional[Dict]:
    pass
```

**After:**
```python
def process_trades(trades: list[dict]) -> dict | None:
    pass
```

### Phase 4: Testing (Week 3)

#### 4.1 Unit Tests
```bash
# Run all unit tests
pytest tests/ -v --cov=src --cov-report=html

# Check coverage
open htmlcov/index.html
```

#### 4.2 Integration Tests
```bash
# Test database connections
python -m src.database.models

# Test exchange APIs
python -m src.exchanges.binance_trading_client

# Test WebSocket connections
python -m src.infrastructure.realtime.binance_websocket
```

#### 4.3 End-to-End Tests
```bash
# Test full agent workflow
python examples/agent_with_binance_live.py

# Test REST API
uvicorn src.api.main:app --reload

# Test dashboard
streamlit run src/dashboard/streamlit_app.py
```

#### 4.4 Performance Testing
```bash
# Benchmark critical operations
python -m pytest tests/performance/ --benchmark-only

# Memory profiling
python -m memory_profiler examples/stress_test.py

# Load testing
locust -f tests/load/locustfile.py
```

### Phase 5: Docker Updates (Week 4)

#### 5.1 Update Base Images

**Dockerfile.agent** (Trading Agent)
```dockerfile
FROM python:3.12-slim-bookworm

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-secure.txt .
RUN pip install --no-cache-dir -r requirements-secure.txt

# Application code
WORKDIR /app
COPY src/ ./src/
COPY config/ ./config/

CMD ["python", "-m", "src.autonomous_agent.trading_agent"]
```

**Dockerfile.api** (REST API)
```dockerfile
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements-secure.txt .
RUN pip install --no-cache-dir -r requirements-secure.txt

COPY src/api ./src/api
COPY src/utils ./src/utils
COPY src/database ./src/database

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 5.2 Update Docker Compose

```yaml
version: '3.8'

services:
  trading-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.agent
    image: trading-ai-agent:py312
    environment:
      - PYTHON_VERSION=3.12
      - ENVIRONMENT=production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    image: trading-ai-api:py312
    ports:
      - "8000:8000"
    environment:
      - PYTHON_VERSION=3.12
    restart: unless-stopped
```

### Phase 6: CI/CD Updates (Week 4)

#### 6.1 Update GitHub Actions

**.github/workflows/test.yml**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-secure.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src
```

## Compatibility Matrix

### Tested Configurations

| Python | TensorFlow | NumPy | Pandas | Status |
|--------|------------|-------|--------|--------|
| 3.9 | 2.20.0 | 2.0.2 | 2.3.3 | ✅ Current |
| 3.10 | 2.20.0 | 2.2.4 | 2.3.3 | ✅ Compatible |
| 3.11 | 2.20.0 | 2.2.4 | 2.3.3 | ✅ Compatible |
| 3.12 | 2.20.0 | 2.2.4 | 2.3.3 | ✅ Compatible |

### Platform Compatibility

| Platform | Python 3.12 | Docker | Notes |
|----------|-------------|--------|-------|
| macOS 14+ | ✅ | ✅ | Full support |
| Ubuntu 22.04+ | ✅ | ✅ | Full support |
| Ubuntu 20.04 | ✅ | ✅ | Requires backport |
| Windows 11 | ✅ | ✅ | WSL2 recommended |
| Raspberry Pi | ⚠️ | ✅ | Limited TensorFlow support |

## Risk Assessment

### High Risk Areas

1. **Machine Learning Models** (Risk: MEDIUM)
   - Pre-trained models may need retraining
   - Serialized model compatibility
   - Mitigation: Test all model loading/inference

2. **Native Extensions** (Risk: MEDIUM)
   - TA-Lib requires recompilation
   - NumPy C extensions
   - Mitigation: Use pre-built wheels

3. **Database Migrations** (Risk: LOW)
   - SQLAlchemy compatible with Python 3.12
   - Mitigation: Test migrations in staging

4. **WebSocket Connections** (Risk: LOW)
   - asyncio improvements in Python 3.12
   - Mitigation: Integration tests

### Rollback Plan

If issues arise:

1. **Immediate Rollback**
   ```bash
   # Switch back to Python 3.9 environment
   source .venv-py39/bin/activate

   # Restore previous requirements
   pip install -r requirements.txt.backup

   # Restart services
   docker-compose down
   docker-compose up -d
   ```

2. **Gradual Rollback**
   - Keep Python 3.9 containers running
   - Route traffic back to old version
   - Debug issues in Python 3.12 environment
   - Fix and redeploy

## Timeline

### Week 1: Preparation
- [ ] Code audit for deprecated features
- [ ] Set up Python 3.12 test environment
- [ ] Run static analysis tools
- [ ] Document all compatibility issues

### Week 2: Updates
- [ ] Update all dependencies
- [ ] Fix deprecated code
- [ ] Update type hints (optional)
- [ ] Run unit tests

### Week 3: Testing
- [ ] Integration testing
- [ ] End-to-end testing
- [ ] Performance benchmarks
- [ ] Security validation

### Week 4: Deployment
- [ ] Update Docker images
- [ ] Update CI/CD pipelines
- [ ] Deploy to staging
- [ ] Production deployment (gradual)

## Success Criteria

- ✅ All tests pass on Python 3.12
- ✅ No performance regression (< 5% acceptable)
- ✅ All integrations working (API, DB, exchanges)
- ✅ Docker containers build and run
- ✅ CI/CD pipeline passes
- ✅ Security audit clean
- ✅ Documentation updated

## Monitoring Post-Update

### Metrics to Track

1. **Performance**
   - Request latency (should improve)
   - Memory usage (should decrease)
   - CPU usage (should decrease)

2. **Stability**
   - Error rate
   - Exception frequency
   - Connection failures

3. **Business Metrics**
   - Trade execution time
   - Strategy performance
   - System uptime

### Alerts

Set up alerts for:
- Increased error rate (> 1%)
- Performance degradation (> 10%)
- Memory leaks
- Unexpected exceptions

## Documentation Updates

- [ ] Update README.md with Python 3.12 requirement
- [ ] Update CONTRIBUTING.md with new setup steps
- [ ] Update API documentation
- [ ] Update deployment guide
- [ ] Create Python 3.12 migration guide

## References

- [Python 3.12 Release Notes](https://docs.python.org/3.12/whatsnew/3.12.html)
- [Python 3.12 Porting Guide](https://docs.python.org/3.12/whatsnew/3.12.html#porting-to-python-3-12)
- [TensorFlow Python 3.12 Support](https://www.tensorflow.org/install)
- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-16
**Owner**: Engineering Team
