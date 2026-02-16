"""
Market Data Endpoints
"""
from fastapi import APIRouter, Query
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

router = APIRouter()


class Ticker(BaseModel):
    """Ticker data schema."""
    symbol: str
    price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percent_24h: float
    timestamp: str


class OrderBook(BaseModel):
    """Order book schema."""
    symbol: str
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]
    timestamp: str


@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str, exchange: str = Query("binance")) -> Ticker:
    """Get current ticker data for a symbol."""
    # TODO: Fetch from cache or exchange
    import random
    base_price = 45000 if 'BTC' in symbol else 2500
    price = base_price * (1 + random.uniform(-0.02, 0.02))

    return Ticker(
        symbol=symbol,
        price=round(price, 2),
        volume_24h=round(random.uniform(1000, 10000), 2),
        high_24h=round(price * 1.05, 2),
        low_24h=round(price * 0.95, 2),
        price_change_24h=round(price * 0.02, 2),
        price_change_percent_24h=2.0,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/tickers")
async def get_tickers(
    symbols: str = Query(..., description="Comma-separated symbols"),
    exchange: str = Query("binance")
) -> List[Ticker]:
    """Get ticker data for multiple symbols."""
    symbol_list = symbols.split(',')
    return [await get_ticker(symbol.strip(), exchange) for symbol in symbol_list]


@router.get("/orderbook/{symbol}")
async def get_orderbook(
    symbol: str,
    exchange: str = Query("binance"),
    depth: int = Query(10, le=100)
) -> OrderBook:
    """Get order book for a symbol."""
    # TODO: Fetch from cache or exchange
    import random
    base_price = 45000 if 'BTC' in symbol else 2500

    # Generate bids
    bids = []
    for i in range(depth):
        price = base_price * (1 - (i * 0.0001))
        quantity = random.uniform(0.1, 2.0)
        bids.append([round(price, 2), round(quantity, 4)])

    # Generate asks
    asks = []
    for i in range(depth):
        price = base_price * (1 + (i * 0.0001))
        quantity = random.uniform(0.1, 2.0)
        asks.append([round(price, 2), round(quantity, 4)])

    return OrderBook(
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    interval: str = Query("1h", regex="^(1m|5m|15m|1h|4h|1d)$"),
    exchange: str = Query("binance"),
    limit: int = Query(100, le=1000)
) -> List[Dict]:
    """Get candlestick/OHLCV data."""
    # TODO: Fetch from cache or exchange
    import random
    base_price = 45000 if 'BTC' in symbol else 2500

    candles = []
    for i in range(limit):
        open_price = base_price * (1 + random.uniform(-0.05, 0.05))
        close_price = open_price * (1 + random.uniform(-0.02, 0.02))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        volume = random.uniform(10, 1000)

        candles.append({
            "timestamp": (datetime.utcnow() - timedelta(hours=limit-i)).isoformat(),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2)
        })

    return candles


@router.get("/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    exchange: str = Query("binance"),
    limit: int = Query(50, le=500)
) -> List[Dict]:
    """Get recent trades for a symbol."""
    # TODO: Fetch from cache or exchange
    import random
    base_price = 45000 if 'BTC' in symbol else 2500

    trades = []
    for i in range(limit):
        price = base_price * (1 + random.uniform(-0.01, 0.01))
        quantity = random.uniform(0.01, 1.0)

        trades.append({
            "id": f"trade_{i}",
            "price": round(price, 2),
            "quantity": round(quantity, 4),
            "side": random.choice(["buy", "sell"]),
            "timestamp": (datetime.utcnow() - timedelta(seconds=limit-i)).isoformat()
        })

    return trades


@router.get("/exchanges")
async def list_exchanges() -> List[Dict]:
    """List supported exchanges."""
    return [
        {
            "id": "binance",
            "name": "Binance",
            "status": "operational",
            "features": ["spot", "futures", "websocket"]
        },
        {
            "id": "coinbase",
            "name": "Coinbase Pro",
            "status": "operational",
            "features": ["spot", "websocket"]
        }
    ]


@router.get("/markets")
async def list_markets(exchange: str = Query("binance")) -> List[Dict]:
    """List available trading pairs."""
    markets = [
        {"symbol": "BTC-USD", "base": "BTC", "quote": "USD", "active": True},
        {"symbol": "ETH-USD", "base": "ETH", "quote": "USD", "active": True},
        {"symbol": "SOL-USD", "base": "SOL", "quote": "USD", "active": True},
        {"symbol": "BTC-USDT", "base": "BTC", "quote": "USDT", "active": True},
        {"symbol": "ETH-USDT", "base": "ETH", "quote": "USDT", "active": True}
    ]
    return markets
