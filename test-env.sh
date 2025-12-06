#!/bin/bash
# Quick test to verify Codespaces environment

echo "🔍 Testing Trading AI Environment..."
echo ""

# Check Python
echo "1️⃣ Checking Python..."
python --version || echo "❌ Python not found"

# Check pip
echo "2️⃣ Checking pip..."
pip --version || echo "❌ pip not found"

# Check if dependencies are installed
echo "3️⃣ Checking key dependencies..."
python -c "import pandas; print('✅ pandas:', pandas.__version__)" 2>/dev/null || echo "❌ pandas not installed"
python -c "import numpy; print('✅ numpy:', numpy.__version__)" 2>/dev/null || echo "❌ numpy not installed"
python -c "import sklearn; print('✅ scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "❌ scikit-learn not installed"
python -c "import yfinance; print('✅ yfinance:', yfinance.__version__)" 2>/dev/null || echo "❌ yfinance not installed"

# Check directories
echo "4️⃣ Checking directories..."
[ -d "src" ] && echo "✅ src/" || echo "❌ src/ not found"
[ -d "data" ] && echo "✅ data/" || echo "❌ data/ not found"
[ -d "models" ] && echo "✅ models/" || echo "❌ models/ not found"
[ -d "tests" ] && echo "✅ tests/" || echo "❌ tests/ not found"

# Check key files
echo "5️⃣ Checking key files..."
[ -f "requirements.txt" ] && echo "✅ requirements.txt" || echo "❌ requirements.txt not found"
[ -f "Makefile" ] && echo "✅ Makefile" || echo "❌ Makefile not found"
[ -f "src/execution/daily_retrain.py" ] && echo "✅ daily_retrain.py" || echo "❌ daily_retrain.py not found"

echo ""
echo "📊 Environment Status:"
if python -c "import pandas, numpy, sklearn, yfinance" 2>/dev/null; then
    echo "✅ READY - All core dependencies installed!"
    echo ""
    echo "Run these commands:"
    echo "  make test       # Run tests"
    echo "  make pipeline   # Run the trading pipeline"
else
    echo "⚠️  NEEDS SETUP - Run: pip install -r requirements.txt"
fi
