#!/bin/bash

# Trading AI Setup Script
echo "🚀 Setting up Trading AI development environment..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p signals
mkdir -p logs
mkdir -p tests

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your API keys"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run basic tests to verify setup
echo "🧪 Running basic setup verification..."
python -c "
import sys
import os
sys.path.append('./src')

print('✅ Python environment: OK')

try:
    import pandas as pd
    import numpy as np
    import sklearn
    print('✅ Core dependencies: OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)

try:
    from utils.logger import setup_logger
    logger = setup_logger('setup_test')
    logger.info('Logger test successful')
    print('✅ Logger setup: OK')
except Exception as e:
    print(f'❌ Logger setup failed: {e}')

print('✅ Basic setup verification complete!')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: docker-compose up --build"
echo "3. Or run directly: python src/execution/daily_retrain.py"
echo ""
echo "For development:"
echo "- Run tests: python -m pytest tests/"
echo "- Check logs: tail -f logs/*.log"
echo "- View signals: ls -la signals/"
echo ""
