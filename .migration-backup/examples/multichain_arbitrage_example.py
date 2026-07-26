"""
Multi-Chain Arbitrage Example
=============================

Complete example of multi-chain arbitrage strategy.

This example demonstrates:
1. Setting up arbitrage scanner
2. Simulating price feeds from multiple chains
3. Finding profitable opportunities
4. Executing arbitrage trades (dry run)
5. Monitoring performance
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List
import random

from src.defi.multichain_arbitrage import (
    MultichainArbitrage,
    MultichainArbitrageConfig,
    ArbitrageOpportunity,
    Chain,
    BridgeProtocol
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Token addresses by chain (real mainnet addresses)
TOKEN_ADDRESSES = {
    'USDC': {
        Chain.ETHEREUM: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
        Chain.BSC: '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',
        Chain.POLYGON: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        Chain.ARBITRUM: '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
        Chain.OPTIMISM: '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
        Chain.AVALANCHE: '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
    },
    'USDT': {
        Chain.ETHEREUM: '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        Chain.BSC: '0x55d398326f99059fF775485246999027B3197955',
        Chain.POLYGON: '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
        Chain.ARBITRUM: '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
        Chain.OPTIMISM: '0x94b008aA00579c1307B0EF2c499aD98a8ce58e58',
        Chain.AVALANCHE: '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7',
    },
    'WETH': {
        Chain.ETHEREUM: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        Chain.BSC: '0x2170Ed0880ac9A755fd29B2688956BD959F933F8',
        Chain.POLYGON: '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
        Chain.ARBITRUM: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
        Chain.OPTIMISM: '0x4200000000000000000000000000000000000006',
        Chain.AVALANCHE: '0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB',
    },
    'DAI': {
        Chain.ETHEREUM: '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        Chain.POLYGON: '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063',
        Chain.ARBITRUM: '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1',
        Chain.OPTIMISM: '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1',
        Chain.AVALANCHE: '0xd586E7F844cEa2F87f50152665BCbc2C279D8d70',
    }
}

# DEXs by chain
DEXS = {
    Chain.ETHEREUM: 'Uniswap',
    Chain.BSC: 'PancakeSwap',
    Chain.POLYGON: 'Quickswap',
    Chain.ARBITRUM: 'Sushiswap',
    Chain.OPTIMISM: 'Velodrome',
    Chain.AVALANCHE: 'TraderJoe',
}


def simulate_price_feed(
    base_price: float,
    volatility: float = 0.002
) -> float:
    """
    Simulate realistic price with small random variations.

    Args:
        base_price: Base price (e.g., 1.0 for USDC)
        volatility: Price volatility (default 0.2%)

    Returns:
        Simulated price
    """
    # Random variation around base price
    variation = random.gauss(0, volatility)
    price = base_price * (1 + variation)
    return price


def simulate_liquidity(
    chain: Chain,
    token: str
) -> float:
    """
    Simulate liquidity based on chain and token.

    Args:
        chain: Blockchain network
        token: Token symbol

    Returns:
        Liquidity in USD
    """
    # Base liquidity by chain
    chain_liquidity = {
        Chain.ETHEREUM: 1000000000,  # $1B
        Chain.BSC: 200000000,        # $200M
        Chain.POLYGON: 150000000,    # $150M
        Chain.ARBITRUM: 100000000,   # $100M
        Chain.OPTIMISM: 80000000,    # $80M
        Chain.AVALANCHE: 60000000,   # $60M
    }

    # Token multiplier
    token_multiplier = {
        'USDC': 1.0,
        'USDT': 0.9,
        'WETH': 0.7,
        'DAI': 0.5,
    }

    base_liq = chain_liquidity.get(chain, 50000000)
    multiplier = token_multiplier.get(token, 0.3)

    return base_liq * multiplier * random.uniform(0.8, 1.2)


async def fetch_prices_for_chain(
    arbitrage: MultichainArbitrage,
    chain: Chain
):
    """
    Fetch prices for all monitored tokens on a specific chain.

    Args:
        arbitrage: Arbitrage instance
        chain: Blockchain to fetch from
    """
    for token in arbitrage.config.monitored_tokens:
        if token not in TOKEN_ADDRESSES:
            continue
        if chain not in TOKEN_ADDRESSES[token]:
            continue

        # Get token address
        token_address = TOKEN_ADDRESSES[token][chain]

        # Base prices
        base_prices = {
            'USDC': 1.000,
            'USDT': 1.000,
            'DAI': 1.000,
            'WETH': 3000.0,
            'WBTC': 60000.0,
        }

        # Simulate price with small variation
        base_price = base_prices.get(token, 1.0)
        price = simulate_price_feed(base_price)

        # Simulate liquidity
        liquidity = simulate_liquidity(chain, token)

        # Update arbitrage scanner
        arbitrage.update_token_price(
            chain=chain,
            token_address=token_address,
            symbol=token,
            price_usd=price,
            liquidity_usd=liquidity,
            dex=DEXS[chain]
        )

        logger.debug(
            f"Updated {token} on {chain.value}: "
            f"${price:.4f} (liq: ${liquidity:,.0f})"
        )

    # Simulate network delay
    await asyncio.sleep(0.1)


async def update_all_prices(arbitrage: MultichainArbitrage):
    """
    Update prices from all enabled chains in parallel.

    Args:
        arbitrage: Arbitrage instance
    """
    tasks = [
        fetch_prices_for_chain(arbitrage, chain)
        for chain in arbitrage.config.enabled_chains
    ]

    await asyncio.gather(*tasks)
    logger.info("Updated prices from all chains")


def should_execute_opportunity(opp: ArbitrageOpportunity) -> bool:
    """
    Check if opportunity meets execution criteria.

    Args:
        opp: Arbitrage opportunity

    Returns:
        True if should execute
    """
    # Profitability checks
    if opp.net_profit_usd < 20.0:
        logger.debug(f"Skipping {opp.token_symbol}: profit too low (${opp.net_profit_usd:.2f})")
        return False

    if opp.roi_pct < 1.5:
        logger.debug(f"Skipping {opp.token_symbol}: ROI too low ({opp.roi_pct:.2f}%)")
        return False

    # Liquidity checks
    if opp.source_price.liquidity_usd < 50000:
        logger.debug(f"Skipping {opp.token_symbol}: source liquidity too low")
        return False

    if opp.dest_price.liquidity_usd < 50000:
        logger.debug(f"Skipping {opp.token_symbol}: dest liquidity too low")
        return False

    # Time checks
    if opp.bridge_time_minutes > 30:
        logger.debug(f"Skipping {opp.token_symbol}: bridge time too long")
        return False

    # Freshness check
    age_seconds = (datetime.now() - opp.timestamp).total_seconds()
    if age_seconds > 60:
        logger.debug(f"Skipping {opp.token_symbol}: price data too old")
        return False

    return True


def log_opportunity(opp: ArbitrageOpportunity):
    """
    Log detailed opportunity information.

    Args:
        opp: Arbitrage opportunity
    """
    logger.info("=" * 80)
    logger.info(f"ARBITRAGE OPPORTUNITY: {opp.token_symbol}")
    logger.info(f"Route: {opp.source_chain.value} → {opp.dest_chain.value}")
    logger.info(f"Bridge: {opp.bridge_protocol.value} (~{opp.bridge_time_minutes:.0f} min)")
    logger.info("-" * 80)
    logger.info(f"Source Price: ${opp.source_price.price_usd:.6f} ({opp.source_price.dex})")
    logger.info(f"Dest Price:   ${opp.dest_price.price_usd:.6f} ({opp.dest_price.dex})")
    logger.info(f"Price Diff:   {opp.price_difference_pct:.3f}%")
    logger.info("-" * 80)
    logger.info(f"Trade Amount: ${opp.trade_amount:.2f}")
    logger.info(f"Gross Profit: ${opp.gross_profit_usd:.2f}")
    logger.info(f"Bridge Fee:   ${opp.bridge_fee:.2f}")
    logger.info(f"Gas Cost:     ${opp.total_gas_cost:.2f}")
    logger.info(f"Net Profit:   ${opp.net_profit_usd:.2f}")
    logger.info(f"ROI:          {opp.roi_pct:.2f}%")
    logger.info("=" * 80)


async def arbitrage_monitoring_loop(
    arbitrage: MultichainArbitrage,
    duration_seconds: int = 300,
    scan_interval: int = 30
):
    """
    Run arbitrage monitoring loop.

    Args:
        arbitrage: Arbitrage instance
        duration_seconds: How long to run (default 5 minutes)
        scan_interval: Seconds between scans (default 30s)
    """
    logger.info(f"Starting arbitrage monitoring for {duration_seconds}s...")

    start_time = datetime.now()
    scan_count = 0
    total_opportunities = 0
    executed_trades = 0

    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        scan_count += 1
        logger.info(f"\n--- Scan #{scan_count} ---")

        # Update prices
        await update_all_prices(arbitrage)

        # Scan for opportunities
        opportunities = arbitrage.scan_opportunities()
        total_opportunities += len(opportunities)

        logger.info(f"Found {len(opportunities)} profitable opportunities")

        # Filter by execution criteria
        executable = [
            opp for opp in opportunities
            if should_execute_opportunity(opp)
        ]

        logger.info(f"Executable opportunities: {len(executable)}")

        # Execute best opportunities
        for opp in executable[:3]:  # Top 3 only
            log_opportunity(opp)

            # Execute (dry run)
            result = arbitrage.execute_arbitrage(opp, dry_run=True)

            if result['success']:
                logger.info("Execution Steps:")
                for step in result['steps']:
                    logger.info(f"  {step}")
                executed_trades += 1
            else:
                logger.error(f"Execution failed: {result.get('error')}")

        # Show stats
        stats = arbitrage.get_statistics()
        logger.info("\n--- Statistics ---")
        logger.info(f"Total scans: {scan_count}")
        logger.info(f"Total opportunities: {total_opportunities}")
        logger.info(f"Executed trades: {executed_trades}")
        logger.info(f"Avg profit: ${stats['avg_profit_usd']:.2f}")
        logger.info(f"Avg ROI: {stats['avg_roi_pct']:.2f}%")
        logger.info(f"Total potential profit: ${stats['total_potential_profit']:.2f}")

        # Wait before next scan
        logger.info(f"\nWaiting {scan_interval}s before next scan...")
        await asyncio.sleep(scan_interval)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total scans: {scan_count}")
    logger.info(f"Total opportunities found: {total_opportunities}")
    logger.info(f"Average opportunities per scan: {total_opportunities / scan_count:.1f}")
    logger.info(f"Trades executed: {executed_trades}")

    final_stats = arbitrage.get_statistics()
    logger.info(f"\nAverage profit per opportunity: ${final_stats['avg_profit_usd']:.2f}")
    logger.info(f"Max profit: ${final_stats['max_profit_usd']:.2f}")
    logger.info(f"Average ROI: {final_stats['avg_roi_pct']:.2f}%")
    logger.info(f"Total potential profit: ${final_stats['total_potential_profit']:.2f}")

    logger.info("\nOpportunities by chain pair:")
    for pair, count in final_stats['opportunities_by_chain_pair'].items():
        logger.info(f"  {pair}: {count}")

    logger.info("\nOpportunities by token:")
    for token, count in final_stats['opportunities_by_token'].items():
        logger.info(f"  {token}: {count}")


async def main():
    """Main example."""
    print("Multi-Chain Arbitrage Example")
    print("=" * 80)

    # Configure arbitrage
    config = MultichainArbitrageConfig(
        enabled_chains=[
            Chain.ETHEREUM,
            Chain.POLYGON,
            Chain.ARBITRUM,
            Chain.OPTIMISM
        ],
        enabled_bridges=[
            BridgeProtocol.HOP,
            BridgeProtocol.ACROSS,
            BridgeProtocol.STARGATE
        ],
        monitored_tokens=['USDC', 'USDT', 'WETH', 'DAI'],
        min_profit_usd=20.0,
        min_roi_pct=1.5,
        default_trade_amount_usd=1000.0,
        max_trade_amount_usd=10000.0,
        max_bridge_time_minutes=30.0,
        min_liquidity_usd=50000.0,
        auto_execute=False  # Dry run only
    )

    # Create arbitrage scanner
    arbitrage = MultichainArbitrage(config)

    print(f"\nConfiguration:")
    print(f"  Chains: {[c.value for c in config.enabled_chains]}")
    print(f"  Bridges: {[b.value for b in config.enabled_bridges]}")
    print(f"  Tokens: {config.monitored_tokens}")
    print(f"  Min Profit: ${config.min_profit_usd}")
    print(f"  Min ROI: {config.min_roi_pct}%")
    print(f"  Trade Size: ${config.default_trade_amount_usd}")

    # Run monitoring loop
    print("\nStarting monitoring loop...")
    print("(This is a simulation - prices are randomly generated)")
    print("-" * 80)

    await arbitrage_monitoring_loop(
        arbitrage,
        duration_seconds=180,  # Run for 3 minutes
        scan_interval=30       # Scan every 30 seconds
    )

    print("\n✅ Multi-Chain Arbitrage Example Complete!")


if __name__ == '__main__':
    asyncio.run(main())
