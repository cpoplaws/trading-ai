#!/bin/bash
# Codespaces post-create setup script

set -e

echo "🚀 Setting up Trading AI in Codespaces..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install dev dependencies
echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p signals
mkdir -p logs

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.template .env || echo "⚠️  No .env.template found - skipping"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with any API keys (optional for Phase 1)"
echo "  2. Run: make test"
echo "  3. Run: make pipeline"
echo ""
