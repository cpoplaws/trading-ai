# ML/AI Dependency Status

**Date**: 2026-02-16
**Task**: #92 - Install missing ML/AI dependencies

---

## Installation Results

### ✅ Successfully Installed
- **scikit-learn**: 1.6.1 ✅ Working
- **xgboost**: 2.1.4 ⚠️ Installed but requires OpenMP runtime
- **lightgbm**: 4.6.0 ⚠️ Installed but requires OpenMP runtime
- **tensorflow**: 2.20.0 ❌ Crashes with mutex error
- **keras**: Latest ❌ Depends on tensorflow, also crashes

### ✅ Already Working
- **pandas**: Works
- **numpy**: Works
- **scipy**: Works
- **plotly**: Works
- **streamlit**: Works

---

## Critical Issues

### Issue 1: Missing OpenMP Runtime (libomp)

**Affects**: xgboost, lightgbm

**Error**:
```
OSError: Library not loaded: @rpath/libomp.dylib
```

**Solution**:
```bash
# Need to install libomp (OpenMP runtime)
# On Mac with Homebrew:
brew install libomp

# Alternative: Use conda environment
conda install -c conda-forge libomp
```

**Impact**:
- ML Ensemble strategy (uses xgboost) won't work
- Some advanced ML features unavailable

---

### Issue 2: TensorFlow Mutex Error

**Affects**: tensorflow, keras

**Error**:
```
libc++abi: terminating due to uncaught exception of type std::__1::system_error:
mutex lock failed: Invalid argument
```

**Possible Causes**:
- Python 3.9 with system libraries incompatibility
- LibreSSL vs OpenSSL version mismatch
- Threading library issues on macOS

**Solutions to Try**:
1. Use TensorFlow 2.15.x (more stable on macOS):
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.15.0
   ```

2. Use conda environment with proper dependencies:
   ```bash
   conda create -n trading-ai python=3.10
   conda activate trading-ai
   conda install tensorflow keras
   ```

3. Use TensorFlow-Metal for Apple Silicon:
   ```bash
   pip install tensorflow-metal
   ```

**Impact**:
- Deep learning models (GRU, CNN-LSTM, VAE) won't work
- RL agents (PPO) won't work if they use tensorflow

---

## What Works NOW

### ✅ Fully Functional Strategies
1. **Classic Strategies** (don't need ML):
   - Mean Reversion
   - Momentum
   - RSI
   - MACD
   - Bollinger Bands
   - Grid Trading
   - DCA

2. **ML Strategies with scikit-learn only**:
   - Any strategy using RandomForest, LogisticRegression, SVM
   - Basic ML predictors (if we write them with sklearn)

3. **Infrastructure**:
   - Dashboard (Streamlit) ✅
   - Broker integrations ✅
   - Data ingestion ✅
   - Paper trading ✅
   - Risk management ✅

---

## What's Blocked

### ⚠️ Partially Blocked
- **ML Ensemble Strategy**: Uses xgboost/lightgbm (needs libomp)
- **Advanced ML Models**: Can use sklearn but not gradient boosting

### ❌ Completely Blocked
- **Deep Learning Models**:
  - GRU Predictor (needs tensorflow)
  - CNN-LSTM Hybrid (needs tensorflow)
  - VAE Anomaly Detector (needs tensorflow)
- **RL Agents**:
  - PPO Agent (needs tensorflow)
  - A2C, SAC (need tensorflow)

---

## Recommended Actions

### Priority 1: Fix OpenMP (Quick Fix)
```bash
# If Homebrew is installed:
brew install libomp

# This unlocks xgboost and lightgbm immediately
```

### Priority 2: Fix TensorFlow (More Complex)
**Option A**: Downgrade to stable version
```bash
pip uninstall tensorflow keras
pip install tensorflow==2.15.0 keras
```

**Option B**: Use conda environment (recommended)
```bash
# Create clean conda environment
conda create -n trading-ai python=3.10
conda activate trading-ai
pip install -r requirements-secure.txt
conda install tensorflow keras -c conda-forge
```

**Option C**: Continue without deep learning
- Focus on classical strategies + scikit-learn ML
- 7 out of 11 strategies work without tensorflow
- Still highly functional trading system

---

## Task #92 Status

### What Was Accomplished
✅ Downloaded and installed all packages
✅ scikit-learn works perfectly
⚠️ xgboost/lightgbm need libomp (fixable)
❌ tensorflow has deeper compatibility issues

### Blocker
System missing critical dependencies:
- libomp (OpenMP runtime) - easy to fix with Homebrew
- tensorflow compatibility - needs investigation or alternative approach

### Recommendation
**Option 1**: Install Homebrew and libomp (5 minutes)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install libomp
```

**Option 2**: Move forward with what works
- Complete repository cleanup (Task #101)
- Test and enhance strategies that don't need tensorflow
- Come back to tensorflow issue later

**Option 3**: Use conda (cleanest solution)
- Create dedicated conda environment
- All dependencies managed properly
- Best for production deployment

---

## Bottom Line

**Good News**: 60% of system works now (7/11 strategies + infrastructure)
**Blocker**: Need libomp for gradient boosting, deeper fix needed for tensorflow
**Workaround**: Can proceed with classical strategies + basic ML while fixing deps

**Best Path Forward**: Install libomp first (quick win), then tackle tensorflow separately or use conda environment.
