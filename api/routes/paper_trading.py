"""
Paper Trading API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from src.paper_trading.engine import (
    PaperTradingEngine, Exchange, OrderSide, OrderType, OrderStatus
)
from src.paper_trading.portfolio import PaperPortfolio
from src.paper_trading.strategy import (
    SimpleMovingAverageStrategy, MomentumStrategy, Backtester, Candle
)
from src.paper_trading.analytics import TradeHistory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper-trading", tags=["Paper Trading"])

# Global instances (in production, use dependency injection or session management)
_engine = PaperTradingEngine()
_portfolio = PaperPortfolio(initial_usd=10000.0)
_history = TradeHistory()


# Request/Response Models

class OrderRequest(BaseModel):
    """Order execution request."""
    exchange: str = Field(..., description="Exchange: coinbase, uniswap, sushiswap")
    symbol: str = Field(..., description="Trading pair, e.g., ETH-USD")
    side: str = Field(..., description="Order side: buy or sell")
    quantity: float = Field(..., gt=0, description="Quantity to trade")
    current_price: float = Field(..., gt=0, description="Current market price")
    order_type: str = Field(default="market", description="Order type: market or limit")
    limit_price: Optional[float] = Field(None, description="Limit price (for limit orders)")


class OrderResponse(BaseModel):
    """Order execution response."""
    order_id: str
    status: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_cost: float
    fees: float
    gas_cost: float
    slippage: float
    created_at: datetime
    filled_at: Optional[datetime]


class BalanceResponse(BaseModel):
    """Balance response."""
    symbol: str
    total: float
    available: float
    locked: float


class PortfolioResponse(BaseModel):
    """Portfolio summary response."""
    initial_value: float
    current_value: float
    total_pnl: float
    total_pnl_percent: float
    total_fees: float
    total_gas: float
    net_pnl: float
    balances: List[BalanceResponse]


class BacktestRequest(BaseModel):
    """Backtest request."""
    strategy: str = Field(..., description="Strategy type: sma or momentum")
    candles: List[Dict] = Field(..., description="Historical OHLCV data")
    initial_capital: float = Field(default=10000.0, gt=0)
    exchange: str = Field(default="coinbase")
    symbol: str = Field(default="ETH-USD")

    # Strategy parameters
    short_window: Optional[int] = Field(10, description="SMA short window")
    long_window: Optional[int] = Field(30, description="SMA long window")
    lookback_period: Optional[int] = Field(14, description="Momentum lookback")
    momentum_threshold: Optional[float] = Field(0.02, description="Momentum threshold")
    profit_target: Optional[float] = Field(0.05, description="Profit target")


class BacktestResponse(BaseModel):
    """Backtest results."""
    strategy: str
    exchange: str
    symbol: str
    candles: int
    period_start: datetime
    period_end: datetime
    trades_executed: int
    signals_generated: int
    initial_capital: float
    final_value: float
    total_pnl: float
    total_pnl_percent: float
    total_fees: float
    total_gas: float
    net_pnl: float
    win_rate: float
    max_drawdown: float


# Endpoints

@router.get("/", summary="Paper trading overview")
async def get_overview():
    """Get paper trading system overview."""
    return {
        "system": "Paper Trading Engine",
        "version": "1.0.0",
        "features": [
            "Portfolio management",
            "Order execution simulation",
            "Trade history tracking",
            "Strategy backtesting",
            "Performance analytics"
        ],
        "exchanges": ["coinbase", "uniswap", "sushiswap"],
        "order_types": ["market", "limit"],
        "endpoints": {
            "portfolio": "/paper-trading/portfolio",
            "orders": "/paper-trading/orders",
            "trades": "/paper-trading/trades",
            "analytics": "/paper-trading/analytics",
            "backtest": "/paper-trading/backtest"
        }
    }


@router.get("/portfolio", response_model=PortfolioResponse, summary="Get portfolio")
async def get_portfolio(
    eth_price: float = Query(2000.0, description="Current ETH price for valuation")
):
    """Get current portfolio summary with balances and P&L."""
    try:
        token_prices = {'ETH': eth_price}
        pnl = _portfolio.get_pnl(token_prices)

        balances = []
        for symbol, balance in _portfolio.balances.items():
            if balance.amount > 0 or balance.locked > 0:
                balances.append(BalanceResponse(
                    symbol=symbol,
                    total=balance.amount,
                    available=balance.available,
                    locked=balance.locked
                ))

        return PortfolioResponse(
            initial_value=pnl['initial_value'],
            current_value=pnl['current_value'],
            total_pnl=pnl['total_pnl'],
            total_pnl_percent=pnl['total_pnl_percent'],
            total_fees=pnl['total_fees'],
            total_gas=pnl['total_gas'],
            net_pnl=pnl['net_pnl'],
            balances=balances
        )
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders", response_model=OrderResponse, summary="Execute order")
async def execute_order(request: OrderRequest):
    """Execute a paper trading order."""
    try:
        # Validate exchange
        try:
            exchange = Exchange[request.exchange.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid exchange: {request.exchange}. Valid: coinbase, uniswap, sushiswap"
            )

        # Validate side
        try:
            side = OrderSide[request.side.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid side: {request.side}. Valid: buy, sell"
            )

        # Check balance
        if side == OrderSide.BUY:
            usd_needed = request.quantity * request.current_price * 1.01  # +1% buffer
            available = _portfolio.get_balance('USD')
            if available < usd_needed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient USD: need ${usd_needed:.2f}, have ${available:.2f}"
                )
        else:
            token = request.symbol.split('-')[0]
            available = _portfolio.get_balance(token)
            if available < request.quantity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient {token}: need {request.quantity}, have {available}"
                )

        # Execute order
        if request.order_type == "market":
            order = _engine.execute_market_order(
                exchange=exchange,
                symbol=request.symbol,
                side=side,
                quantity=request.quantity,
                current_price=request.current_price
            )
        elif request.order_type == "limit":
            if not request.limit_price:
                raise HTTPException(status_code=400, detail="Limit price required for limit orders")

            order = _engine.execute_limit_order(
                exchange=exchange,
                symbol=request.symbol,
                side=side,
                quantity=request.quantity,
                limit_price=request.limit_price,
                current_price=request.current_price
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid order type: {request.order_type}")

        # Update portfolio and history
        if order.status == OrderStatus.FILLED:
            _portfolio.process_order(order, request.current_price)
            _history.add_order(order)

        return OrderResponse(
            order_id=order.order_id,
            status=order.status.value,
            exchange=order.exchange.value,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            avg_fill_price=order.avg_fill_price,
            total_cost=order.total_cost,
            fees=order.fees,
            gas_cost=order.gas_cost,
            slippage=order.slippage,
            created_at=order.created_at,
            filled_at=order.filled_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders", summary="Get order history")
async def get_orders(
    status: Optional[str] = Query(None, description="Filter by status: filled, pending, cancelled")
):
    """Get all paper trading orders."""
    try:
        status_filter = None
        if status:
            try:
                status_filter = OrderStatus[status.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid: filled, pending, cancelled"
                )

        orders = _engine.get_all_orders(status_filter)

        return {
            "total": len(orders),
            "orders": [
                {
                    "order_id": o.order_id,
                    "status": o.status.value,
                    "exchange": o.exchange.value,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "order_type": o.order_type.value,
                    "quantity": o.quantity,
                    "filled_quantity": o.filled_quantity,
                    "avg_fill_price": o.avg_fill_price,
                    "total_cost": o.total_cost,
                    "fees": o.fees,
                    "gas_cost": o.gas_cost,
                    "slippage": o.slippage,
                    "created_at": o.created_at,
                    "filled_at": o.filled_at
                }
                for o in orders
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades", summary="Get trade history")
async def get_trades(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """Get all completed trades."""
    try:
        trades = _history.get_all_trades(symbol)

        return {
            "total": len(trades),
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "total_cost": t.total_cost,
                    "fees": t.fees,
                    "gas_cost": t.gas_cost,
                    "exchange": t.exchange,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "holding_period": str(t.holding_period) if t.holding_period else None
                }
                for t in trades
            ]
        }
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics", summary="Get performance analytics")
async def get_analytics():
    """Get comprehensive performance metrics."""
    try:
        metrics = _history.get_performance_metrics()

        # Convert timedelta to string
        if metrics['avg_holding_period']:
            metrics['avg_holding_period'] = str(metrics['avg_holding_period'])

        # Convert infinity to null for JSON compliance
        import math
        if math.isinf(metrics.get('profit_factor', 0)):
            metrics['profit_factor'] = None

        return metrics
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse, summary="Run strategy backtest")
async def run_backtest(request: BacktestRequest):
    """Run a strategy backtest on historical data."""
    try:
        # Parse candles
        candles = []
        for c in request.candles:
            candle = Candle(
                timestamp=datetime.fromisoformat(c['timestamp']) if isinstance(c['timestamp'], str) else c['timestamp'],
                open=c['open'],
                high=c['high'],
                low=c['low'],
                close=c['close'],
                volume=c['volume']
            )
            candles.append(candle)

        if len(candles) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 candles for backtesting")

        # Create strategy
        if request.strategy.lower() == "sma":
            strategy = SimpleMovingAverageStrategy(
                short_window=request.short_window,
                long_window=request.long_window,
                position_size=1.0
            )
        elif request.strategy.lower() == "momentum":
            strategy = MomentumStrategy(
                lookback_period=request.lookback_period,
                momentum_threshold=request.momentum_threshold,
                position_size=1.0,
                profit_target=request.profit_target
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy: {request.strategy}. Valid: sma, momentum"
            )

        # Validate exchange
        try:
            exchange = Exchange[request.exchange.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid exchange: {request.exchange}"
            )

        # Run backtest
        backtester = Backtester(
            strategy=strategy,
            initial_capital=request.initial_capital,
            exchange=exchange,
            symbol=request.symbol
        )

        results = backtester.run(candles)

        # Flatten period structure for response model
        period = results.pop('period')
        results['period_start'] = period['start']
        results['period_end'] = period['end']

        return BacktestResponse(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", summary="Reset paper trading system")
async def reset_system(initial_usd: float = Query(10000.0, gt=0)):
    """Reset portfolio, orders, and trade history."""
    try:
        global _engine, _portfolio, _history

        _engine = PaperTradingEngine()
        _portfolio = PaperPortfolio(initial_usd=initial_usd)
        _history = TradeHistory()

        logger.info(f"Paper trading system reset with ${initial_usd:,.2f}")

        return {
            "status": "reset",
            "initial_capital": initial_usd,
            "message": "Paper trading system has been reset"
        }
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))
