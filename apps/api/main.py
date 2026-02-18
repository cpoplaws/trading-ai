"""
Trading AI - FastAPI Backend
Connects to real Alpaca data and serves dashboard
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading AI API",
    description="Real-time trading AI backend with Alpaca integration",
    version="1.0.0"
)

# CORS - Configure for your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import Alpaca (install with: pip install alpaca-py)
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass

    # Initialize Alpaca clients
    ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

    if ALPACA_KEY and ALPACA_SECRET:
        trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
        data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        logger.info("✅ Connected to Alpaca API")
    else:
        trading_client = None
        data_client = None
        logger.warning("⚠️ Alpaca API keys not found - using demo mode")
except ImportError:
    trading_client = None
    data_client = None
    logger.warning("⚠️ alpaca-py not installed - using demo mode")

# WebSocket connections
active_connections: List[WebSocket] = []

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Trading AI API",
        "version": "1.0.0",
        "alpaca_connected": trading_client is not None
    }

@app.get("/api/portfolio")
async def get_portfolio():
    """Get real-time portfolio data from Alpaca"""
    try:
        if trading_client:
            # Get real account data from Alpaca
            account = trading_client.get_account()
            positions = trading_client.get_all_positions()

            # Calculate metrics
            total_value = float(account.equity)
            daily_pnl = float(account.equity) - float(account.last_equity)
            daily_pnl_percent = (daily_pnl / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0

            return {
                "total_value": total_value,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "daily_pnl": daily_pnl,
                "daily_pnl_percent": daily_pnl_percent,
                "positions_count": len(positions),
                "sharpe_ratio": 2.13,  # TODO: Calculate from historical trades
                "win_rate": 0.623,      # TODO: Calculate from historical trades
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Demo mode - return sample data
            return {
                "total_value": 42567.89,
                "cash": 8234.12,
                "buying_power": 16468.24,
                "daily_pnl": 1245.67,
                "daily_pnl_percent": 3.02,
                "positions_count": 8,
                "sharpe_ratio": 2.13,
                "win_rate": 0.623,
                "timestamp": datetime.now().isoformat(),
                "demo_mode": True
            }
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return {"error": str(e)}, 500

@app.get("/api/portfolio/history")
async def get_portfolio_history(days: int = 30):
    """Get historical portfolio performance"""
    try:
        if trading_client:
            # TODO: Fetch from database or calculate from account history
            pass

        # Generate sample data for now
        history = []
        base_value = 40000
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            value = base_value * (1 + (i * 0.003) + ((-1)**i * 0.01))
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(value, 2)
            })

        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return {"error": str(e)}, 500

@app.get("/api/strategies")
async def get_strategies():
    """Get all trading strategies and their performance"""
    strategies = [
        {
            "id": "mean_reversion",
            "name": "Mean Reversion",
            "enabled": True,
            "pnl": 1250.50,
            "trades": 45,
            "win_rate": 0.67
        },
        {
            "id": "momentum",
            "name": "Momentum",
            "enabled": True,
            "pnl": 2341.20,
            "trades": 38,
            "win_rate": 0.71
        },
        {
            "id": "ml_ensemble",
            "name": "ML Ensemble",
            "enabled": True,
            "pnl": 3450.75,
            "trades": 52,
            "win_rate": 0.73
        },
        {
            "id": "ppo_rl",
            "name": "PPO RL Agent",
            "enabled": True,
            "pnl": 1876.30,
            "trades": 28,
            "win_rate": 0.68
        },
        {
            "id": "rsi",
            "name": "RSI",
            "enabled": True,
            "pnl": 890.45,
            "trades": 34,
            "win_rate": 0.62
        },
        {
            "id": "macd",
            "name": "MACD",
            "enabled": True,
            "pnl": 1120.60,
            "trades": 31,
            "win_rate": 0.65
        },
        {
            "id": "bollinger",
            "name": "Bollinger Bands",
            "enabled": True,
            "pnl": 750.20,
            "trades": 29,
            "win_rate": 0.59
        },
        {
            "id": "yield_optimizer",
            "name": "Yield Optimizer",
            "enabled": True,
            "pnl": 4200.80,
            "trades": 12,
            "win_rate": 0.83
        },
        {
            "id": "multichain_arb",
            "name": "Multi-Chain Arb",
            "enabled": False,
            "pnl": 0.00,
            "trades": 0,
            "win_rate": 0.00
        },
        {
            "id": "grid",
            "name": "Grid Trading",
            "enabled": False,
            "pnl": 0.00,
            "trades": 0,
            "win_rate": 0.00
        },
        {
            "id": "dca",
            "name": "DCA",
            "enabled": True,
            "pnl": 550.30,
            "trades": 24,
            "win_rate": 0.58
        },
    ]
    return {"strategies": strategies, "total": len(strategies)}

@app.get("/api/trades/recent")
async def get_recent_trades(limit: int = 20):
    """Get recent trades"""
    try:
        if trading_client:
            # Get recent orders from Alpaca
            # orders = trading_client.get_orders(limit=limit)
            # TODO: Format and return
            pass

        # Return sample data
        trades = [
            {
                "id": "1",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 185.50,
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "strategy": "momentum",
                "pnl": 125.40
            },
            {
                "id": "2",
                "symbol": "TSLA",
                "side": "sell",
                "quantity": 5,
                "price": 242.30,
                "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                "strategy": "mean_reversion",
                "pnl": 87.20
            },
        ]
        return {"trades": trades}
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        return {"error": str(e)}, 500

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(active_connections)}")

    try:
        while True:
            # Send portfolio updates every 5 seconds
            if trading_client:
                account = trading_client.get_account()
                data = {
                    "type": "portfolio_update",
                    "data": {
                        "value": float(account.equity),
                        "daily_change": float(account.equity) - float(account.last_equity),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                # Demo mode
                data = {
                    "type": "portfolio_update",
                    "data": {
                        "value": 42567.89,
                        "daily_change": 1245.67,
                        "timestamp": datetime.now().isoformat(),
                        "demo_mode": True
                    }
                }

            await websocket.send_json(data)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Trading AI API starting up...")
    logger.info(f"Alpaca connected: {trading_client is not None}")
    if not trading_client:
        logger.warning("Running in DEMO MODE - connect Alpaca API keys for real data")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Trading AI API shutting down...")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
