"""
Market Data Routes
Prices, tickers, OHLCV data from Coinbase and DEXs.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta

from api.dependencies import get_coinbase_collector, get_redis
from api.config import settings
import json

router = APIRouter()


class TickerResponse(BaseModel):
    """Ticker price response."""
    symbol: str
    exchange: str
    price: float
    volume_24h: Optional[float]
    timestamp: datetime


class CandleResponse(BaseModel):
    """OHLCV candle response."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@router.get("/ticker/{symbol}", response_model=TickerResponse)
async def get_ticker(
    symbol: str,
    exchange: str = Query("coinbase", description="Exchange name"),
    coinbase=Depends(get_coinbase_collector),
    redis=Depends(get_redis)
):
    """
    Get current ticker price for a symbol.

    - **symbol**: Trading pair (e.g., BTC-USD, ETH-USD)
    - **exchange**: Exchange name (coinbase, uniswap)
    """
    cache_key = f"ticker:{exchange}:{symbol}"

    # Try cache first
    try:
        cached = redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return TickerResponse(**data)
    except:
        pass

    # Fetch from Coinbase
    if exchange == "coinbase":
        ticker = coinbase.get_ticker(symbol)

        if not ticker:
            raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

        response = TickerResponse(
            symbol=symbol,
            exchange=exchange,
            price=float(ticker.get('price', 0)),
            volume_24h=float(ticker.get('volume', 0)) if 'volume' in ticker else None,
            timestamp=datetime.now()
        )

        # Cache result
        try:
            redis.setex(
                cache_key,
                settings.cache_ttl,
                json.dumps(response.dict(), default=str)
            )
        except:
            pass

        return response

    raise HTTPException(status_code=400, detail=f"Exchange {exchange} not supported")


@router.get("/candles/{symbol}", response_model=List[CandleResponse])
async def get_candles(
    symbol: str,
    interval: str = Query("3600", description="Candle interval in seconds"),
    limit: int = Query(100, ge=1, le=300, description="Number of candles"),
    coinbase=Depends(get_coinbase_collector)
):
    """
    Get historical OHLCV candles.

    - **symbol**: Trading pair (e.g., BTC-USD)
    - **interval**: Candle size (60=1m, 300=5m, 3600=1h, 86400=1d)
    - **limit**: Number of candles (max 300)
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=int(interval) * limit)

    candles = coinbase.get_candles(
        symbol=symbol,
        granularity=interval,
        start=start_time,
        end=end_time
    )

    if not candles:
        raise HTTPException(status_code=404, detail=f"No candles found for {symbol}")

    return [
        CandleResponse(
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume
        )
        for candle in candles[:limit]
    ]


@router.get("/products")
async def get_products(coinbase=Depends(get_coinbase_collector)):
    """
    Get all available trading pairs.

    Returns list of all products on Coinbase.
    """
    products = coinbase.get_products()
    return {"count": len(products), "products": products}
