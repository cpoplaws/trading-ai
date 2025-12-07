#!/usr/bin/env bash
# Quick setup script for Trading-AI dashboard

set -e

echo "🚀 Trading-AI Dashboard Setup"
echo "=============================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{match($0, /[0-9]+\.[0-9]+/); print substr($0, RSTART, RLENGTH)}')
echo "   Found: Python $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "   ⚠️  Warning: Python 3.11+ recommended (you have $python_version)"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ Dependencies installed!"

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "   ✅ .env created from template"
    echo "   ℹ️  Edit .env to add your API keys (optional)"
else
    echo ""
    echo "✅ .env file already exists"
fi

# Check data availability
echo ""
echo "📊 Checking data availability..."
data_count=$(find data/processed -name "*.csv" 2>/dev/null | wc -l)
if [ "$data_count" -eq 0 ]; then
    echo "   ⚠️  No processed data found"
    echo "   Run: python src/execution/daily_retrain.py"
else
    echo "   ✅ Found $data_count data files"
fi

echo ""
echo "=============================="
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. (Optional) Edit .env with your API keys"
echo "   2. Launch dashboard: ./run_dashboard.sh"
echo "   3. Or run demo: python demo_live_trading.py"
echo ""
echo "📚 See GETTING_STARTED.md for detailed instructions"
echo ""
