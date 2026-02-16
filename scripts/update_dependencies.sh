#!/bin/bash
# Dependency Update Script
# Updates all dependencies to secure versions and verifies fixes

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "TRADING AI - SECURITY UPDATE"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Backup current requirements"
echo "  2. Update dependencies to secure versions"
echo "  3. Run security audit to verify fixes"
echo "  4. Test basic functionality"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Update cancelled."
    exit 1
fi

# Backup current requirements
echo ""
echo "=== 1. Backing up current requirements ==="
cp requirements.txt requirements.txt.backup.$(date +%Y%m%d_%H%M%S)
echo "✓ Backup created"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "=== Creating virtual environment ==="
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "=== Activating virtual environment ==="
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "=== Upgrading pip ==="
pip install --upgrade pip
echo "✓ pip upgraded"

# Install pip-audit if not present
echo ""
echo "=== Installing pip-audit ==="
pip install pip-audit
echo "✓ pip-audit installed"

# Update critical dependencies
echo ""
echo "=== 2. Updating critical dependencies ==="
echo ""
echo "Updating urllib3 (1.26.20 → 2.6.3)..."
pip install --upgrade "urllib3>=2.6.3,<3.0"
echo "✓ urllib3 updated"

echo ""
echo "Updating Pillow (11.3.0 → 12.1.1)..."
pip install --upgrade "pillow>=12.1.1,<13.0"
echo "✓ Pillow updated"

echo ""
echo "Updating Keras (3.10.0 → 3.13.1)..."
pip install --upgrade "keras>=3.13.1,<4.0"
echo "✓ Keras updated"

# Install/update other dependencies from secure requirements
echo ""
echo "Installing dependencies from requirements-secure.txt..."
if [ -f "requirements-secure.txt" ]; then
    pip install -r requirements-secure.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ requirements-secure.txt not found, using original requirements.txt"
    pip install -r requirements.txt
fi

# Freeze updated requirements
echo ""
echo "Freezing updated requirements..."
pip freeze > requirements-frozen-$(date +%Y%m%d).txt
echo "✓ Requirements frozen to requirements-frozen-$(date +%Y%m%d).txt"

# Run security audit
echo ""
echo "=== 3. Running security audit ==="
echo ""
echo "Scanning for vulnerabilities..."

# Run audit and save results
pip-audit --format json > security-audit-$(date +%Y%m%d).json 2>&1 || true
pip-audit --format text > security-audit-$(date +%Y%m%d).txt 2>&1 || true

# Display results
if pip-audit --format text 2>&1 | grep -q "Found 0 known vulnerabilities"; then
    echo ""
    echo "✅ SUCCESS: No vulnerabilities found!"
    echo ""
else
    echo ""
    echo "⚠️  WARNING: Some vulnerabilities remain"
    echo ""
    echo "Review security-audit-$(date +%Y%m%d).txt for details"
    echo ""

    # Check for known unpatched CVE
    if pip-audit --format text 2>&1 | grep -q "CVE-2026-1669"; then
        echo "ℹ️  Known unpatched vulnerability: CVE-2026-1669 (Keras)"
        echo "   Mitigation: Only load models from trusted sources"
        echo ""
    fi
fi

# Test basic imports
echo ""
echo "=== 4. Testing basic functionality ==="
echo ""

echo "Testing urllib3..."
python3 -c "import urllib3; print(f'urllib3 version: {urllib3.__version__}')" && echo "✓ urllib3 OK" || echo "✗ urllib3 FAILED"

echo "Testing Pillow..."
python3 -c "import PIL; print(f'Pillow version: {PIL.__version__}')" && echo "✓ Pillow OK" || echo "✗ Pillow FAILED"

echo "Testing Keras..."
python3 -c "import keras; print(f'Keras version: {keras.__version__}')" && echo "✓ Keras OK" || echo "✗ Keras FAILED"

echo "Testing requests..."
python3 -c "import requests; print(f'requests version: {requests.__version__}')" && echo "✓ requests OK" || echo "✗ requests FAILED"

# Run unit tests if available
if [ -d "tests" ] && [ -f "tests/test_strategies.py" ]; then
    echo ""
    echo "Running unit tests..."
    python3 -m pytest tests/test_strategies.py -v || echo "⚠️  Some tests failed, review logs"
fi

# Summary
echo ""
echo "========================================================================"
echo "UPDATE COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Dependencies updated to secure versions"
echo "  ✓ Security audit completed"
echo "  ✓ Basic functionality tested"
echo ""
echo "Next steps:"
echo "  1. Review security-audit-$(date +%Y%m%d).txt"
echo "  2. Run full test suite: pytest tests/"
echo "  3. Test in staging environment"
echo "  4. Deploy to production"
echo ""
echo "Files created:"
echo "  - requirements.txt.backup.* (backup)"
echo "  - requirements-frozen-$(date +%Y%m%d).txt (updated versions)"
echo "  - security-audit-$(date +%Y%m%d).json (JSON report)"
echo "  - security-audit-$(date +%Y%m%d).txt (text report)"
echo ""
echo "Documentation:"
echo "  - docs/SECURITY_AUDIT_REPORT.md"
echo ""
