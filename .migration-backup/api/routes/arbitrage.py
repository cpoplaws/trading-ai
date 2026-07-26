"""
Arbitrage Routes
Find and analyze arbitrage opportunities.
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime

from api.dependencies import get_dex_analyzer, get_coinbase_collector, get_uniswap_collector, get_gas_tracker
from src.data_collection.uniswap_collector import WETH_ADDRESS, USDC_ADDRESS

router = APIRouter()


class ArbitrageOpportunityResponse(BaseModel):
    """Arbitrage opportunity."""
    type: str
    token: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_percent: float
    gross_profit: float
    gas_cost: float
    net_profit: float
    roi_percent: float
    confidence: float
    timestamp: datetime


@router.get("/opportunities", response_model=List[ArbitrageOpportunityResponse])
async def find_opportunities(
    token: str = Query("ETH", description="Token to find arbitrage for"),
    min_profit: float = Query(10.0, description="Minimum profit USD"),
    trade_size: float = Query(5000.0, description="Trade size USD"),
    analyzer=Depends(get_dex_analyzer),
    coinbase=Depends(get_coinbase_collector),
    uniswap=Depends(get_uniswap_collector),
    gas_tracker=Depends(get_gas_tracker)
):
    """
    Find current arbitrage opportunities.

    - **token**: Token symbol (ETH, BTC, etc.)
    - **min_profit**: Minimum profit threshold
    - **trade_size**: Size of trade in USD

    Returns list of profitable opportunities sorted by profit.
    """
    opportunities = []

    try:
        # Get current prices
        if token == "ETH":
            symbol = "ETH-USD"
            token_in = WETH_ADDRESS
            token_out = USDC_ADDRESS
        else:
            return opportunities  # Only ETH supported for now

        # Get Coinbase price
        cb_ticker = coinbase.get_ticker(symbol)
        if not cb_ticker:
            return opportunities

        cb_price = float(cb_ticker['price'])

        # Get Uniswap price
        uni_pool = uniswap.get_pool_info(token_in, token_out)
        if not uni_pool:
            return opportunities

        uni_price = uni_pool.price

        # Get gas price
        gas = gas_tracker.get_current_gas_price()
        if not gas:
            return opportunities

        gas_cost = gas_tracker.estimate_transaction_cost(
            'uniswap_v2_swap',
            gas_price_gwei=gas.standard,
            eth_price_usd=cb_price
        )

        # Find CEX-DEX arbitrage
        opp = analyzer.find_cex_dex_arbitrage(
            cex_price=cb_price,
            dex_price=uni_price,
            token=token,
            trade_size_usd=trade_size,
            gas_cost_usd=gas_cost.cost_usd
        )

        if opp and opp.net_profit >= min_profit:
            spread = abs(opp.sell_price - opp.buy_price) / opp.buy_price * 100

            opportunities.append(
                ArbitrageOpportunityResponse(
                    type=opp.type.value,
                    token=opp.token,
                    buy_exchange=opp.buy_exchange,
                    sell_exchange=opp.sell_exchange,
                    buy_price=opp.buy_price,
                    sell_price=opp.sell_price,
                    spread_percent=spread,
                    gross_profit=opp.gross_profit,
                    gas_cost=opp.gas_cost,
                    net_profit=opp.net_profit,
                    roi_percent=opp.roi_percent,
                    confidence=opp.confidence,
                    timestamp=opp.timestamp
                )
            )

    except Exception as e:
        # Log error but return empty list
        pass

    return opportunities


@router.get("/spread/{token}")
async def get_spread(
    token: str,
    coinbase=Depends(get_coinbase_collector),
    uniswap=Depends(get_uniswap_collector)
):
    """
    Get price spread between Coinbase and Uniswap.

    - **token**: Token symbol (ETH)

    Returns current prices and spread percentage.
    """
    if token == "ETH":
        symbol = "ETH-USD"
        token_in = WETH_ADDRESS
        token_out = USDC_ADDRESS
    else:
        return {"error": "Token not supported"}

    # Get prices
    cb_ticker = coinbase.get_ticker(symbol)
    uni_pool = uniswap.get_pool_info(token_in, token_out)

    if not cb_ticker or not uni_pool:
        return {"error": "Failed to fetch prices"}

    cb_price = float(cb_ticker['price'])
    uni_price = uni_pool.price

    spread = abs(uni_price - cb_price)
    spread_percent = (spread / min(cb_price, uni_price)) * 100

    return {
        "token": token,
        "coinbase_price": cb_price,
        "uniswap_price": uni_price,
        "spread": spread,
        "spread_percent": spread_percent,
        "higher_on": "uniswap" if uni_price > cb_price else "coinbase"
    }
