"""
DEX Routes
Uniswap pools, liquidity, price impact, swap simulation.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from api.dependencies import get_uniswap_collector

router = APIRouter()


class PoolResponse(BaseModel):
    """DEX pool information."""
    address: str
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    price: float
    liquidity_usd: float


class PriceImpactResponse(BaseModel):
    """Price impact calculation."""
    amount_in: float
    amount_out: float
    spot_price: float
    execution_price: float
    price_impact_percent: float
    slippage_percent: float


@router.get("/pools/{token0}/{token1}", response_model=PoolResponse)
async def get_pool(
    token0: str,
    token1: str,
    uniswap=Depends(get_uniswap_collector)
):
    """
    Get Uniswap pool information.

    - **token0**: First token address
    - **token1**: Second token address

    Returns pool reserves, liquidity, and current price.
    """
    pool = uniswap.get_pool_info(token0, token1)

    if not pool:
        raise HTTPException(
            status_code=404,
            detail=f"Pool not found for {token0}/{token1}"
        )

    return PoolResponse(
        address=pool.address,
        token0=pool.token0_symbol,
        token1=pool.token1_symbol,
        reserve0=pool.reserve0,
        reserve1=pool.reserve1,
        price=pool.price,
        liquidity_usd=pool.liquidity_usd
    )


@router.post("/price-impact", response_model=PriceImpactResponse)
async def calculate_price_impact(
    token_in: str,
    token_out: str,
    amount_in: float,
    uniswap=Depends(get_uniswap_collector)
):
    """
    Calculate price impact for a swap.

    - **token_in**: Input token address
    - **token_out**: Output token address
    - **amount_in**: Amount to swap

    Returns execution price and slippage estimate.
    """
    result = uniswap.calculate_price_impact(amount_in, token_in, token_out)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    slippage = abs((result['execution_price'] - result['spot_price']) / result['spot_price'] * 100)

    return PriceImpactResponse(
        amount_in=result['amount_in'],
        amount_out=result['amount_out'],
        spot_price=result['spot_price'],
        execution_price=result['execution_price'],
        price_impact_percent=result['price_impact_percent'],
        slippage_percent=slippage
    )


@router.get("/route/{token_in}/{token_out}")
async def get_best_route(
    token_in: str,
    token_out: str,
    amount_in: float = Query(1.0, description="Amount to route"),
    uniswap=Depends(get_uniswap_collector)
):
    """
    Find best swap route between tokens.

    - **token_in**: Input token
    - **token_out**: Output token
    - **amount_in**: Amount to swap

    Returns optimal routing path (may include WETH).
    """
    path = uniswap.get_best_path(token_in, token_out, amount_in)

    return {
        "token_in": token_in,
        "token_out": token_out,
        "path": path,
        "hops": len(path) - 1
    }
