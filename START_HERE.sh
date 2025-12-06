#!/bin/bash
# START HERE - First time setup for Codespaces

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🚀 Trading AI - Codespaces Setup                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Make scripts executable
chmod +x test-env.sh .devcontainer/postCreateCommand.sh

echo "📍 Current directory: $(pwd)"
echo ""

# Check if dependencies are installed
if python3 -c "import pandas" 2>/dev/null; then
    echo "✅ Dependencies already installed!"
    echo ""
    echo "Quick commands:"
    echo "  make test       - Run test suite"
    echo "  make pipeline   - Run trading pipeline"
    echo "  ./test-env.sh   - Check environment"
    echo ""
else
    echo "📦 Installing dependencies (this may take 2-3 minutes)..."
    echo ""
    
    # Upgrade pip
    python3 -m pip install --upgrade pip --quiet
    
    # Install core dependencies
    echo "Installing core dependencies..."
    pip3 install -r requirements.txt --quiet
    
    if [ $? -eq 0 ]; then
        echo "✅ Core dependencies installed!"
        echo ""
        
        # Ask about dev dependencies
        read -p "Install dev dependencies (pytest, ruff, black)? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing dev dependencies..."
            pip3 install -r requirements-dev.txt --quiet
            pre-commit install
            echo "✅ Dev dependencies installed!"
        fi
        
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "✨ Setup complete! Here's what to do next:"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "1️⃣  Test your setup:"
        echo "   ./test-env.sh"
        echo ""
        echo "2️⃣  Run the test suite:"
        echo "   make test"
        echo ""
        echo "3️⃣  Run your first pipeline:"
        echo "   make pipeline"
        echo ""
        echo "4️⃣  View results:"
        echo "   ls -lh signals/"
        echo "   cat signals/AAPL_signals.csv"
        echo ""
        echo "📚 Documentation:"
        echo "   - CODESPACES.md  - Codespaces-specific guide"
        echo "   - QUICKSTART.md  - General quick start"
        echo "   - README.md      - Full documentation"
        echo ""
        echo "💡 Tip: Run 'make help' to see all available commands"
        echo ""
    else
        echo "❌ Installation failed. Try manually:"
        echo "   pip3 install -r requirements.txt"
    fi
fi
