#!/usr/bin/env bash
# Fix pip installation on Alpine Linux

set -e

echo "🔧 Fixing pip installation..."

# Check if we're on Alpine
if [ -f /etc/alpine-release ]; then
    echo "📦 Detected Alpine Linux - installing pip via apk..."
    sudo apk add --no-cache \
        py3-pip \
        python3-dev \
        gcc \
        musl-dev \
        linux-headers \
        g++ \
        make \
        wget
    
    # Create symlinks if needed
    sudo ln -sf /usr/bin/pip3 /usr/local/bin/pip || true
    sudo ln -sf /usr/bin/python3 /usr/local/bin/python || true
    
    echo "✅ pip installed successfully!"
    
elif command -v apt-get &> /dev/null; then
    echo "📦 Detected Debian/Ubuntu - ensuring pip is available..."
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev build-essential
    
    echo "✅ pip installed successfully!"
    
else
    echo "⚠️  Unknown system - trying to bootstrap pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 --version
pip --version || pip3 --version

echo ""
echo "✅ All set! You can now run:"
echo "   pip install -r requirements.txt"
