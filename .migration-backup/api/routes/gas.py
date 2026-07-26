"""
Gas Routes
Gas prices, cost estimation, optimal timing.
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from datetime import datetime

from api.dependencies import get_gas_tracker

router = APIRouter()


class GasPriceResponse(BaseModel):
    """Gas price response."""
    slow: float
    standard: float
    fast: float
    instant: float
    base_fee: float
    priority_fee: float
    timestamp: datetime
    conditions: str


class GasCostResponse(BaseModel):
    """Gas cost estimation."""
    operation: str
    gas_limit: int
    gas_price_gwei: float
    cost_eth: float
    cost_usd: float


@router.get("/prices", response_model=GasPriceResponse)
async def get_gas_prices(gas_tracker=Depends(get_gas_tracker)):
    """
    Get current gas prices.

    Returns gas prices for different speed levels.
    """
    gas = gas_tracker.get_current_gas_price()

    if not gas:
        return {
            "error": "Failed to fetch gas prices"
        }

    # Determine conditions
    if gas.standard < 20:
        conditions = "GOOD"
    elif gas.standard < 50:
        conditions = "OK"
    else:
        conditions = "HIGH"

    return GasPriceResponse(
        slow=gas.slow,
        standard=gas.standard,
        fast=gas.fast,
        instant=gas.instant,
        base_fee=gas.base_fee or 0,
        priority_fee=gas.priority_fee or 0,
        timestamp=gas.timestamp,
        conditions=conditions
    )


@router.get("/estimate", response_model=GasCostResponse)
async def estimate_gas_cost(
    operation: str = Query("uniswap_v2_swap", description="Operation type"),
    eth_price: float = Query(2000.0, description="ETH price in USD"),
    gas_tracker=Depends(get_gas_tracker)
):
    """
    Estimate gas cost for an operation.

    - **operation**: Type (uniswap_v2_swap, erc20_transfer, etc.)
    - **eth_price**: Current ETH price

    Returns cost in ETH and USD.
    """
    cost = gas_tracker.estimate_transaction_cost(
        operation=operation,
        eth_price_usd=eth_price
    )

    return GasCostResponse(
        operation=operation,
        gas_limit=cost.gas_limit,
        gas_price_gwei=cost.gas_price_gwei,
        cost_eth=cost.cost_eth,
        cost_usd=cost.cost_usd
    )


@router.get("/max-profitable")
async def calculate_max_profitable_gas(
    profit_eth: float = Query(0.05, description="Expected profit in ETH"),
    gas_limit: int = Query(150000, description="Gas limit"),
    gas_tracker=Depends(get_gas_tracker)
):
    """
    Calculate maximum gas price for profitable trade.

    - **profit_eth**: Expected profit
    - **gas_limit**: Gas limit for transaction

    Returns max gas price to remain profitable.
    """
    max_gas = gas_tracker.calculate_max_profitable_gas(profit_eth, gas_limit)

    current_gas = gas_tracker.get_current_gas_price()
    is_profitable = current_gas.standard <= max_gas if current_gas else False

    return {
        "profit_eth": profit_eth,
        "gas_limit": gas_limit,
        "max_gas_gwei": max_gas,
        "current_gas_gwei": current_gas.standard if current_gas else None,
        "is_profitable": is_profitable
    }


@router.get("/conditions")
async def get_trading_conditions(gas_tracker=Depends(get_gas_tracker)):
    """
    Get current trading conditions based on gas.

    Returns recommendation for trading.
    """
    gas = gas_tracker.get_current_gas_price()

    if not gas:
        return {"status": "unknown", "message": "Unable to fetch gas prices"}

    if gas.standard < 20:
        status = "excellent"
        message = "Gas is very low. Great time to trade!"
        color = "green"
    elif gas.standard < 35:
        status = "good"
        message = "Gas is reasonable. Good for larger trades."
        color = "green"
    elif gas.standard < 50:
        status = "moderate"
        message = "Gas is moderate. Only profitable trades recommended."
        color = "yellow"
    else:
        status = "poor"
        message = "Gas is high. Consider waiting or using different strategy."
        color = "red"

    return {
        "status": status,
        "message": message,
        "color": color,
        "gas_price": gas.standard,
        "timestamp": gas.timestamp
    }
