#!/bin/bash

# ==========================================
# Install Core Dependencies Only
# ==========================================

set -e

echo "Installing core Python dependencies..."

# Install essential packages only
pip3 install --upgrade pip

# Core packages
pip3 install numpy pandas scikit-learn matplotlib seaborn scipy

# FastAPI & web
pip3 install fastapi uvicorn[standard] pydantic websockets

# Database
pip3 install sqlalchemy psycopg2-binary alembic asyncpg

# Redis
pip3 install redis hiredis

# Testing
pip3 install pytest pytest-cov pytest-asyncio pytest-mock

# Utilities
pip3 install python-dotenv loguru requests aiohttp

# Trading basics
pip3 install yfinance ccxt

# Web3
pip3 install web3 eth-account

echo "✓ Core dependencies installed!"
echo ""
echo "Optional (install if needed):"
echo "  Machine Learning: pip3 install tensorflow torch transformers"
echo "  Technical Analysis: pip3 install TA-Lib"
echo "  More exchanges: pip3 install python-binance coinbase alpaca-trade-api"
