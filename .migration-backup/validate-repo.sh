#!/bin/bash
# Comprehensive validation script for trading-ai repository

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         Trading AI - Repository Validation                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

echo "1️⃣  Checking Python environment..."
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version | cut -d' ' -f2)
    pass "Python found (version $VERSION)"
else
    fail "Python not found"
fi

echo ""
echo "2️⃣  Checking directory structure..."
for dir in "src" "tests" "config" "docs" "data" "models" "logs"; do
    if [ -d "$dir" ]; then
        pass "Directory exists: $dir"
    else
        fail "Missing directory: $dir"
    fi
done

echo ""
echo "3️⃣  Checking critical files..."
CRITICAL_FILES=(
    "README.md"
    "requirements.txt"
    "requirements-dev.txt"
    "Makefile"
    ".gitignore"
    "docker-compose.yml"
    "Dockerfile"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        pass "File exists: $file"
    else
        fail "Missing file: $file"
    fi
done

echo ""
echo "4️⃣  Checking source files..."
if [ -f "src/data_ingestion/fetch_data.py" ]; then
    pass "Data ingestion module exists"
else
    fail "Data ingestion module missing"
fi

if [ -f "src/feature_engineering/feature_generator.py" ]; then
    pass "Feature engineering module exists"
else
    fail "Feature engineering module missing"
fi

if [ -f "src/modeling/train_model.py" ]; then
    pass "Modeling module exists"
else
    fail "Modeling module missing"
fi

if [ -f "src/execution/daily_retrain.py" ]; then
    pass "Pipeline module exists"
else
    fail "Pipeline module missing"
fi

echo ""
echo "5️⃣  Checking test files..."
if [ -f "tests/test_trading_ai.py" ]; then
    pass "Main test suite exists"
else
    fail "Main test suite missing"
fi

if [ -f "tests/test_advanced_strategies.py" ]; then
    pass "Advanced strategies tests exist"
else
    warn "Advanced strategies tests missing"
fi

echo ""
echo "6️⃣  Checking configuration files..."
if [ -f "config/settings.yaml" ]; then
    pass "Settings file exists"
else
    warn "Settings file missing (will use defaults)"
fi

if [ -f ".env" ]; then
    pass ".env file exists"
else
    warn ".env file missing (will use defaults)"
fi

if [ -f ".env.template" ]; then
    pass ".env.template exists"
else
    fail ".env.template missing"
fi

echo ""
echo "7️⃣  Checking documentation..."
DOCS=(
    "QUICKSTART.md"
    "CODESPACES.md"
    "AUDIT_REPORT.md"
    "SECURITY_REPORT.md"
    "FIXES.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        pass "Documentation exists: $doc"
    else
        warn "Documentation missing: $doc"
    fi
done

echo ""
echo "8️⃣  Checking .gitignore coverage..."
GITIGNORE_PATTERNS=(
    ".env"
    "__pycache__"
    "*.pyc"
    "data/raw/"
    "data/processed/"
    "models/"
    "logs/"
)

if [ -f ".gitignore" ]; then
    for pattern in "${GITIGNORE_PATTERNS[@]}"; do
        if grep -q "$pattern" .gitignore; then
            pass "Gitignore includes: $pattern"
        else
            warn "Gitignore missing: $pattern"
        fi
    done
else
    fail ".gitignore not found"
fi

echo ""
echo "9️⃣  Checking dependencies (if installed)..."
if command -v python3 &> /dev/null; then
    if python3 -c "import pandas" 2>/dev/null; then
        pass "pandas installed"
    else
        warn "pandas not installed"
    fi
    
    if python3 -c "import numpy" 2>/dev/null; then
        pass "numpy installed"
    else
        warn "numpy not installed"
    fi
    
    if python3 -c "import sklearn" 2>/dev/null; then
        pass "scikit-learn installed"
    else
        warn "scikit-learn not installed"
    fi
    
    if python3 -c "import yfinance" 2>/dev/null; then
        pass "yfinance installed"
    else
        warn "yfinance not installed"
    fi
fi

echo ""
echo "🔟  Running configuration validator..."
if [ -f "src/utils/config_validator.py" ]; then
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml" 2>/dev/null; then
            python3 src/utils/config_validator.py 2>&1 | head -20
        else
            warn "PyYAML not installed, skipping config validation"
        fi
    fi
else
    warn "Config validator not found"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Repository validation PASSED!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies: make install"
    echo "  2. Run tests: make test"
    echo "  3. Run pipeline: make pipeline"
    exit 0
else
    echo -e "${RED}❌ Repository validation FAILED!${NC}"
    echo ""
    echo "Please fix the failed checks above."
    exit 1
fi
