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

# Import strategy components
from strategy_runner import StrategyRunner
from strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    RSIStrategy,
    MLEnsembleStrategy,
    RLAgentStrategy
)
from ml import get_model_server
from swarm import get_swarm_controller
from intelligence import get_intelligence_service
from crypto_config import get_strategy_symbols, Chain, CryptoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading AI API",
    description="Real-time trading AI backend with Alpaca integration",
    version="1.0.1"  # Force Railway redeploy with all crypto features
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
trading_client = None
data_client = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass

    # Initialize Alpaca clients
    ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

    if ALPACA_KEY and ALPACA_SECRET:
        logger.info(f"🔑 Found Alpaca keys: {ALPACA_KEY[:8]}... and secret key (length: {len(ALPACA_SECRET)})")
        try:
            trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
            data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            # Test the connection
            account = trading_client.get_account()
            logger.info(f"✅ Connected to Alpaca API - Account Status: {account.status}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alpaca: {type(e).__name__}: {e}")
            trading_client = None
            data_client = None
    else:
        logger.warning(f"⚠️ Alpaca API keys not found - ALPACA_KEY: {'SET' if ALPACA_KEY else 'NOT SET'}, ALPACA_SECRET: {'SET' if ALPACA_SECRET else 'NOT SET'}")
except ImportError as e:
    logger.warning(f"⚠️ alpaca-py not installed - using demo mode: {e}")
except Exception as e:
    logger.error(f"❌ Unexpected error during Alpaca setup: {type(e).__name__}: {e}")

# WebSocket connections
active_connections: List[WebSocket] = []

# Strategy state management (in-memory for now, will be DB later)
strategy_states = {
    "mean_reversion": False,  # Start disabled for safety
    "momentum": False,
    "rsi": False,
    "ml_ensemble": False,
    "ppo_rl": False,
    "macd": False,
    "bollinger": False,
    "yield_optimizer": False,
    "multichain_arb": False,
    "grid": False,
    "dca": False,
}

# Strategy runner instance
strategy_runner: StrategyRunner = None

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
    """Get all crypto trading strategies and their performance"""
    # Strategy names with chain info
    strategy_names = {
        "mean_reversion": "Mean Reversion (Base)",
        "momentum": "Momentum (Solana)",
        "rsi": "RSI (Base)",
        "ml_ensemble": "ML Ensemble (Base)",
        "ppo_rl": "RL Agent (Solana)",
        "macd": "MACD (Optimism)",
        "bollinger": "Bollinger Bands (Base)",
        "yield_optimizer": "Yield Optimizer (Arbitrum)",
        "multichain_arb": "Cross-Chain Arb",
        "grid": "Grid Trading (BSC)",
        "dca": "DCA (Base)",
    }

    # Build strategy list with current enabled state and real performance
    strategies = []
    for strategy_id, enabled in strategy_states.items():
        # Get real performance from strategy runner if available
        if strategy_runner:
            perf = strategy_runner.get_performance(strategy_id)
        else:
            perf = {"pnl": 0.0, "trades": 0, "win_rate": 0.0}

        strategies.append({
            "id": strategy_id,
            "name": strategy_names.get(strategy_id, strategy_id.replace("_", " ").title()),
            "enabled": enabled,
            "pnl": perf.get("pnl", 0.0),
            "trades": perf.get("trades", 0),
            "win_rate": perf.get("win_rate", 0.0)
        })

    return {"strategies": strategies, "total": len(strategies)}

@app.post("/api/strategies/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str):
    """Toggle a strategy on/off"""
    if strategy_id not in strategy_states:
        return {"error": f"Strategy '{strategy_id}' not found"}, 404

    # Toggle the strategy state
    strategy_states[strategy_id] = not strategy_states[strategy_id]
    new_state = strategy_states[strategy_id]

    logger.info(f"Strategy '{strategy_id}' toggled to: {'ENABLED' if new_state else 'DISABLED'}")

    # Broadcast update to all WebSocket clients
    update_message = {
        "type": "strategy_update",
        "data": {
            "id": strategy_id,
            "enabled": new_state,
            "timestamp": datetime.now().isoformat()
        }
    }

    for connection in active_connections:
        try:
            await connection.send_json(update_message)
        except Exception as e:
            logger.error(f"Error sending WebSocket update: {e}")

    return {
        "success": True,
        "strategy_id": strategy_id,
        "enabled": new_state
    }

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

@app.get("/api/agents/status")
async def get_agents_status():
    """Get agent swarm status"""
    try:
        swarm = get_swarm_controller()
        status = swarm.get_status()
        return status
    except Exception as e:
        logger.error(f"Error fetching agent status: {e}")
        return {"error": str(e)}, 500

@app.post("/api/agents/enable")
async def enable_swarm():
    """Enable the agent swarm"""
    try:
        swarm = get_swarm_controller()
        swarm.enabled = True
        logger.info("Agent swarm enabled")
        return {"success": True, "enabled": True}
    except Exception as e:
        logger.error(f"Error enabling swarm: {e}")
        return {"error": str(e)}, 500

@app.post("/api/agents/disable")
async def disable_swarm():
    """Disable the agent swarm"""
    try:
        swarm = get_swarm_controller()
        swarm.enabled = False
        logger.info("Agent swarm disabled")
        return {"success": True, "enabled": False}
    except Exception as e:
        logger.error(f"Error disabling swarm: {e}")
        return {"error": str(e)}, 500

@app.post("/api/agents/{agent_name}/toggle")
async def toggle_agent(agent_name: str):
    """Toggle individual agent on/off"""
    try:
        swarm = get_swarm_controller()

        if agent_name not in swarm.agents:
            return {"error": f"Agent '{agent_name}' not found"}, 404

        agent = swarm.agents[agent_name]
        agent.enabled = not agent.enabled

        logger.info(f"Agent '{agent_name}' toggled to: {'ENABLED' if agent.enabled else 'DISABLED'}")

        return {
            "success": True,
            "agent": agent_name,
            "enabled": agent.enabled
        }
    except Exception as e:
        logger.error(f"Error toggling agent: {e}")
        return {"error": str(e)}, 500

@app.get("/api/agents/decisions")
async def get_agent_decisions(limit: int = 20):
    """Get recent agent decisions"""
    try:
        swarm = get_swarm_controller()
        decisions = swarm.get_recent_decisions(limit=limit)
        return {"decisions": decisions, "total": len(decisions)}
    except Exception as e:
        logger.error(f"Error fetching agent decisions: {e}")
        return {"error": str(e)}, 500

@app.get("/api/intelligence")
async def get_market_intelligence():
    """Get current market intelligence"""
    try:
        intel_service = get_intelligence_service()

        # Get current intelligence or analyze if needed
        intelligence = intel_service.get_current_intelligence()

        if intelligence is None:
            # Generate intelligence using recent data (placeholder for now)
            # In production, this would use real market data
            import numpy as np
            sample_prices = np.random.randn(100).cumsum() + 100
            market_data = {
                "prices": sample_prices.tolist(),
                "volumes": [1000000] * 100
            }
            intelligence = intel_service.analyze(market_data)

        return intelligence
    except Exception as e:
        logger.error(f"Error fetching intelligence: {e}")
        return {"error": str(e)}, 500

@app.post("/api/intelligence/analyze")
async def analyze_market(symbol: str = "SPY"):
    """Analyze market for a specific symbol"""
    try:
        intel_service = get_intelligence_service()

        # TODO: Fetch real market data from Alpaca for the symbol
        # For now, use sample data
        import numpy as np
        sample_prices = np.random.randn(100).cumsum() + 100
        market_data = {
            "prices": sample_prices.tolist(),
            "volumes": [1000000] * 100,
            "symbol": symbol
        }

        intelligence = intel_service.analyze(market_data)
        intelligence["symbol"] = symbol

        logger.info(f"Market intelligence analyzed for {symbol}")
        return intelligence
    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
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

async def intelligence_update_loop():
    """Background task to update market intelligence every 5 minutes"""
    logger.info("Intelligence update loop started")

    while True:
        try:
            await asyncio.sleep(300)  # Update every 5 minutes

            intel_service = get_intelligence_service()

            # TODO: Fetch real market data from Alpaca
            # For now, use sample data
            import numpy as np
            sample_prices = np.random.randn(100).cumsum() + 100
            market_data = {
                "prices": sample_prices.tolist(),
                "volumes": [1000000] * 100
            }

            intelligence = intel_service.analyze(market_data)
            logger.info(f"Intelligence updated: {intelligence['signal']} (confidence: {intelligence['confidence']:.2f})")

            # Broadcast to WebSocket clients
            update_message = {
                "type": "intelligence_update",
                "data": {
                    "signal": intelligence["signal"],
                    "regime": intelligence["regime"]["regime"],
                    "confidence": intelligence["confidence"],
                    "timestamp": intelligence["timestamp"]
                }
            }

            for connection in active_connections:
                try:
                    await connection.send_json(update_message)
                except Exception as e:
                    logger.error(f"Error sending intelligence update: {e}")

        except Exception as e:
            logger.error(f"Error in intelligence update loop: {e}")

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    global strategy_runner

    logger.info("🚀 Trading AI API starting up...")
    logger.info(f"Alpaca connected: {trading_client is not None}")

    # Initialize ML model server
    logger.info("Initializing ML Model Server...")
    model_server = get_model_server()
    # Try to load models (will gracefully handle if models don't exist)
    try:
        model_server.load_all_models()
    except Exception as e:
        logger.warning(f"Could not load ML models: {e}")
        logger.info("ML strategies will use fallback methods")

    # Initialize Agent Swarm
    logger.info("Initializing Agent Swarm...")
    swarm = get_swarm_controller()
    logger.info(f"✅ Agent Swarm initialized with {len(swarm.agents)} agents")

    # Initialize Intelligence Service
    logger.info("Initializing Intelligence Service...")
    intel_service = get_intelligence_service()
    logger.info("✅ Intelligence Service initialized")

    # Start intelligence update loop
    asyncio.create_task(intelligence_update_loop())

    # Initialize strategy runner
    if trading_client and data_client:
        logger.info("Initializing Strategy Runner...")
        strategy_runner = StrategyRunner(trading_client, data_client, strategy_states)

        # Register crypto strategies
        logger.info("Registering crypto strategies...")

        # Base Network strategies (PRIMARY)
        mean_rev = MeanReversionStrategy(symbols=get_strategy_symbols("mean_reversion"))
        strategy_runner.register_strategy("mean_reversion", mean_rev)
        logger.info(f"  ✅ Mean Reversion on Base: {get_strategy_symbols('mean_reversion')}")

        rsi = RSIStrategy(symbols=get_strategy_symbols("rsi"))
        strategy_runner.register_strategy("rsi", rsi)
        logger.info(f"  ✅ RSI on Base: {get_strategy_symbols('rsi')}")

        # Solana strategies (fast execution)
        momentum = MomentumStrategy(symbols=get_strategy_symbols("momentum"))
        strategy_runner.register_strategy("momentum", momentum)
        logger.info(f"  ✅ Momentum on Solana: {get_strategy_symbols('momentum')}")

        rl_agent = RLAgentStrategy(symbols=get_strategy_symbols("ppo_rl"))
        strategy_runner.register_strategy("ppo_rl", rl_agent)
        logger.info(f"  ✅ RL Agent on Solana: {get_strategy_symbols('ppo_rl')}")

        # ML Ensemble on Base (sophisticated)
        ml_ensemble = MLEnsembleStrategy(symbols=get_strategy_symbols("ml_ensemble"))
        strategy_runner.register_strategy("ml_ensemble", ml_ensemble)
        logger.info(f"  ✅ ML Ensemble on Base: {get_strategy_symbols('ml_ensemble')}")

        logger.info("✅ Registered 5 CRYPTO strategies")
        logger.info("   🔗 Primary: Base Network (Coinbase L2)")
        logger.info("   🔗 Secondary: Solana")
        logger.info("   📊 Supported chains: Base, Solana, Optimism, Linea, ZKsync, Arbitrum, BSC, Polygon")

        # Start strategy runner in background
        logger.info("Starting strategy execution loop...")
        asyncio.create_task(strategy_runner.start())
        logger.info("✅ Strategy Runner started")
    else:
        logger.warning("⚠️ Running in DEMO MODE - connect Alpaca API keys for real data and strategy execution")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Trading AI API shutting down...")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
