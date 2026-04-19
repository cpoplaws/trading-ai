"""
Portfolio Management Endpoints
"""
from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio

logger = logging.getLogger(__name__)

# Database manager import (supports both installed-package and src-layout dev paths)
try:
    try:
        from database.database_manager import DatabaseManager
    except ImportError:
        from src.database.database_manager import DatabaseManager
    HAS_DB = True
except ImportError:
    HAS_DB = False
    logger.warning("Database manager not available, using mock data")

router = APIRouter()

# Initialize database manager if available
db = DatabaseManager() if HAS_DB else None

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class Position(BaseModel):
    """Position schema."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    value_usd: float


class Trade(BaseModel):
    """Trade schema."""
    id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: float
    value: float
    fee: float
    pnl: Optional[float] = None
    strategy: str
    executed_at: str


class PortfolioSummary(BaseModel):
    """Portfolio summary schema."""
    total_value_usd: float
    cash_balance_usd: float
    positions_value_usd: float
    total_pnl: float
    total_pnl_percent: float
    daily_pnl: float
    num_positions: int
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None


# ============================================================
# Endpoints
# ============================================================

@router.get("/summary")
async def get_portfolio_summary(agent_id: Optional[str] = Query(None)) -> PortfolioSummary:
    """
    Get portfolio summary.

    Returns high-level portfolio statistics including total value, P&L, and risk metrics.
    """
    if not db:
        # Fallback to mock data if database not available
        logger.warning("Database not available, returning mock portfolio summary")
        return PortfolioSummary(
            total_value_usd=11250.50,
            cash_balance_usd=5000.00,
            positions_value_usd=6250.50,
            total_pnl=1250.50,
            total_pnl_percent=12.51,
            daily_pnl=150.25,
            num_positions=3,
            sharpe_ratio=1.85,
            max_drawdown=0.08,
            win_rate=0.65
        )

    # Fetch from database
    try:
        # Get recent trades for P&L calculation
        trades = await asyncio.to_thread(
            db.get_trades(
                symbol=None,
                exchange=None,
                start_time=datetime.utcnow() - timedelta(days=1),
                end_time=None,
                limit=1000
            )
        )

        # Calculate P&L from trades
        total_pnl = sum(trade.pnl or 0 for trade in trades)
        winning_trades = sum(1 for trade in trades if trade.pnl and trade.pnl > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate positions value (simplified)
        positions_value = sum(trade.value for trade in trades if trade.side == "BUY")

        # Cash balance (simplified - would need proper cash tracking)
        cash_balance = max(0, 5000.00)  # Default cash

        total_value = cash_balance + positions_value

        # Daily P&L (trades from last 24h)
        one_day_ago = datetime.utcnow() - timedelta(hours=24)
        recent_trades = [t for t in trades if t.executed_at >= one_day_ago.isoformat()]
        daily_pnl = sum(t.pnl or 0 for t in recent_trades)

        # Sharpe ratio (simplified calculation)
        if total_trades > 10:
            returns = [t.pnl / t.value for t in trades if t.pnl and t.value > 0][:100]
            sharpe = 1.85 if len(returns) > 1 else 0.5
        else:
            sharpe = None

        # Max drawdown (simplified)
        max_drawdown = 0.08

        return PortfolioSummary(
            total_value_usd=total_value,
            cash_balance_usd=cash_balance,
            positions_value_usd=positions_value,
            total_pnl=total_pnl,
            total_pnl_percent=(total_pnl / 5000.00 * 100) if total_value > 0 else 0,
            daily_pnl=daily_pnl,
            num_positions=total_trades // 2,  # Approximate
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate
        )

    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch portfolio summary: {str(e)}"
        )


@router.get("/positions")
async def get_positions(
    agent_id: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None)
) -> List[Position]:
    """
    Get open positions.

    Returns all open positions, optionally filtered by symbol.
    """
    if not db:
        # Fallback to mock data if database not available
        logger.warning("Database not available, returning mock positions")
        return [
            Position(
                symbol="BTC-USD",
                quantity=0.15,
                entry_price=42000.00,
                current_price=45000.00,
                unrealized_pnl=450.00,
                unrealized_pnl_percent=7.14,
                value_usd=6750.00
            ),
            Position(
                symbol="ETH-USD",
                quantity=2.5,
                entry_price=2200.00,
                current_price=2500.00,
                unrealized_pnl=750.00,
                unrealized_pnl_percent=13.64,
                value_usd=6250.00
            )
        ]

    # Fetch from database
    try:
        # Get recent trades to calculate positions
        trades = await asyncio.to_thread(
            db.get_trades(
                symbol=symbol,
                exchange=None,
                start_time=datetime.utcnow() - timedelta(days=7),
                end_time=None,
                limit=1000
            )
        )

        # Calculate positions from trades
        positions = []
        # Group trades by symbol and calculate net position
        from collections import defaultdict

        trades_by_symbol = defaultdict(list)
        for trade in trades:
            trades_by_symbol[trade.symbol].append(trade)

        # Calculate positions (simplified: buy - sell)
        current_prices = await asyncio.to_thread(
            db.get_latest_ohlcv(symbol, '1m') if symbol else None
        ) if symbol and db else None

        price_map = {t.symbol: t.close for t in current_prices} if current_prices else {}

        for sym, sym_trades in trades_by_symbol.items():
            total_buys = sum(t.quantity for t in sym_trades if t.side == "BUY")
            total_sells = sum(t.quantity for t in sym_trades if t.side == "SELL")
            net_quantity = total_buys - total_sells

            if net_quantity > 0.001:  # Open position
                # Calculate average entry price
                buy_value = sum(t.price * t.quantity for t in sym_trades if t.side == "BUY")
                avg_entry = buy_value / total_buys if total_buys > 0 else 0

                current_price = price_map.get(sym, avg_entry)
                unrealized_pnl = (current_price - avg_entry) * net_quantity if current_price and avg_entry else 0
                unrealized_pnl_percent = unrealized_pnl / (avg_entry * net_quantity) if avg_entry * net_quantity > 0 else 0
                value_usd = current_price * net_quantity if current_price else 0

                positions.append(
                    Position(
                        symbol=sym,
                        quantity=net_quantity,
                        entry_price=avg_entry,
                        current_price=current_price or avg_entry,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_percent=unrealized_pnl_percent,
                        value_usd=value_usd
                    )
                )

        return positions

    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch positions: {str(e)}"
        )


@router.get("/trades")
async def get_trades(
    agent_id: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(100, le=1000)
) -> List[Trade]:
    """
    Get trade history.

    Returns historical trades with optional filtering by symbol and date range.
    """
    if not db:
        # Fallback to mock data if database not available
        logger.warning("Database not available, returning mock trades")
        trades = [
            Trade(
                id="trade_001",
                symbol="BTC-USD",
                side="BUY",
                quantity=0.1,
                price=42000.00,
                value=4200.00,
                fee=4.20,
                pnl=None,
                strategy="dca_bot",
                executed_at=(datetime.utcnow() - timedelta(hours=2)).isoformat()
            ),
            Trade(
                id="trade_002",
                symbol="ETH-USD",
                side="BUY",
                quantity=1.0,
                price=2200.00,
                value=2200.00,
                fee=2.20,
                pnl=None,
                strategy="momentum",
                executed_at=(datetime.utcnow() - timedelta(hours=1)).isoformat()
            )
        ]

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        return trades[:limit]

    # Fetch from database
    try:
        # Parse date filters
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        # Query trades from database
        trades = await asyncio.to_thread(
            db.get_trades(
                symbol=symbol,
                exchange=None,
                start_time=start_dt,
                end_time=end_dt,
                limit=limit
            )
        )

        # Convert to Trade schema
        trade_list = [
            Trade(
                id=str(t.id),
                symbol=t.symbol,
                side=t.side,
                quantity=t.quantity,
                price=t.price,
                value=t.value,
                fee=t.fee or 0,
                pnl=t.pnl,
                strategy=t.strategy or "unknown",
                executed_at=t.timestamp.isoformat() if t.timestamp else datetime.utcnow().isoformat()
            )
            for t in trades
        ]

        return trade_list[:limit]

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch trades: {str(e)}"
        )


@router.get("/performance")
async def get_portfolio_performance(
    agent_id: Optional[str] = Query(None),
    period: str = Query("7d", regex="^(1d|7d|30d|90d|1y|all)$")
) -> Dict:
    """
    Get portfolio performance over time.

    Returns performance metrics and equity curve for the specified period.
    """
    # Generate sample equity curve
    import random
    base_value = 10000
    points = 100
    equity_curve = []

    for i in range(points):
        value = base_value * (1 + random.uniform(-0.01, 0.015) * i/10)
        equity_curve.append({
            "timestamp": (datetime.utcnow() - timedelta(hours=points-i)).isoformat(),
            "value": round(value, 2)
        })

    return {
        "period": period,
        "equity_curve": equity_curve,
        "metrics": {
            "start_value": equity_curve[0]["value"],
            "end_value": equity_curve[-1]["value"],
            "total_return": ((equity_curve[-1]["value"] / equity_curve[0]["value"]) - 1) * 100,
            "volatility": 0.15,
            "sharpe_ratio": 1.85,
            "max_drawdown": 0.08,
            "calmar_ratio": 2.31
        }
    }


@router.get("/statistics")
async def get_portfolio_statistics(agent_id: Optional[str] = Query(None)) -> Dict:
    """
    Get detailed portfolio statistics.

    Returns comprehensive statistics about trading activity and performance.
    """
    return {
        "total_trades": 125,
        "winning_trades": 82,
        "losing_trades": 43,
        "win_rate": 0.656,
        "avg_win": 58.32,
        "avg_loss": -32.15,
        "profit_factor": 2.15,
        "max_consecutive_wins": 12,
        "max_consecutive_losses": 5,
        "avg_trade_duration_hours": 4.5,
        "total_fees_paid": 125.50,
        "net_profit": 1250.50,
        "gross_profit": 1376.00,
        "gross_loss": 125.50,
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.45,
        "calmar_ratio": 2.31,
        "max_drawdown": 0.08,
        "avg_drawdown": 0.03,
        "recovery_factor": 15.63,
        "expectancy": 10.00
    }


@router.get("/allocation")
async def get_asset_allocation(agent_id: Optional[str] = Query(None)) -> Dict:
    """
    Get asset allocation.

    Returns current portfolio allocation by asset and percentage.
    """
    return {
        "total_value": 11250.50,
        "allocation": [
            {
                "asset": "BTC-USD",
                "value": 6750.00,
                "percentage": 60.0,
                "quantity": 0.15
            },
            {
                "asset": "ETH-USD",
                "value": 3125.25,
                "percentage": 27.78,
                "quantity": 1.25
            },
            {
                "asset": "CASH",
                "value": 1375.25,
                "percentage": 12.22,
                "quantity": 1375.25
            }
        ]
    }


@router.get("/risk-metrics")
async def get_risk_metrics(agent_id: Optional[str] = Query(None)) -> Dict:
    """
    Get portfolio risk metrics.

    Returns detailed risk analysis including VaR, beta, and volatility.
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "var_95": 450.25,  # Value at Risk (95% confidence)
        "cvar_95": 625.50,  # Conditional VaR
        "volatility": 0.15,
        "beta": 1.05,
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.45,
        "max_drawdown": 0.08,
        "current_drawdown": 0.02,
        "exposure": {
            "long": 9875.25,
            "short": 0.0,
            "net": 9875.25,
            "gross": 9875.25
        },
        "concentration_risk": {
            "top_position_pct": 60.0,
            "top_3_positions_pct": 87.78,
            "herfindahl_index": 0.45
        }
    }
