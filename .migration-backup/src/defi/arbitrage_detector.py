"""
Cross-DEX Arbitrage Detector
Identifies profitable arbitrage opportunities across multiple DEXs.
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.defi.uniswap_v3 import UniswapV3Client, WETH, USDC, USDT, DAI
from src.defi.curve_finance import CurveFinanceClient
from src.defi.pancakeswap_trader import PancakeSwapTrader

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    token_path: List[str]
    dex_path: List[str]
    buy_dex: str
    sell_dex: str
    buy_price: float
    sell_price: float
    profit_percent: float
    profit_after_gas: float
    amount_in: float
    amount_out: float
    gas_cost_usd: float
    confidence: float  # 0-1, based on liquidity and slippage
    execution_route: List[Dict]


class ArbitrageDetector:
    """
    Detects and evaluates arbitrage opportunities across DEXs.

    Strategies:
    1. Simple arbitrage: Buy on DEX A, sell on DEX B
    2. Triangular arbitrage: A -> B -> C -> A
    3. Cross-chain arbitrage: Bridge tokens between chains
    4. Flash loan arbitrage: Borrow, arbitrage, repay in one transaction
    """

    def __init__(
        self,
        rpc_url: str,
        min_profit_percent: float = 0.5,
        max_gas_cost_usd: float = 50.0
    ):
        """
        Initialize arbitrage detector.

        Args:
            rpc_url: Ethereum RPC endpoint
            min_profit_percent: Minimum profit percentage to consider
            max_gas_cost_usd: Maximum gas cost in USD
        """
        self.rpc_url = rpc_url
        self.min_profit_percent = min_profit_percent
        self.max_gas_cost_usd = max_gas_cost_usd

        # Initialize DEX clients
        self.uniswap_v3 = UniswapV3Client(rpc_url)
        self.curve = CurveFinanceClient(rpc_url)
        # self.pancakeswap = PancakeSwapTrader(...)  # BSC only

        # Common token pairs to monitor
        self.token_pairs = [
            (WETH, USDC),
            (WETH, DAI),
            (WETH, USDT),
            (USDC, DAI),
            (USDC, USDT),
            (DAI, USDT),
        ]

        logger.info(f"Arbitrage detector initialized (min profit: {min_profit_percent}%)")

    def detect_simple_arbitrage(
        self,
        token_in: str,
        token_out: str,
        amount_in: float = 1.0
    ) -> List[ArbitrageOpportunity]:
        """
        Detect simple arbitrage: buy on one DEX, sell on another.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount to trade

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        try:
            # Get quotes from all DEXs
            quotes = self._get_all_quotes(token_in, token_out, amount_in)

            if len(quotes) < 2:
                return opportunities

            # Find best buy (highest output) and best sell (highest input for reverse)
            best_buy = max(quotes, key=lambda q: q['amount_out'])

            # Get reverse quotes (token_out -> token_in) for selling
            reverse_quotes = self._get_all_quotes(token_out, token_in, best_buy['amount_out'])

            if not reverse_quotes:
                return opportunities

            best_sell = max(reverse_quotes, key=lambda q: q['amount_out'])

            # Calculate profit
            final_amount = best_sell['amount_out']
            profit = final_amount - amount_in
            profit_percent = (profit / amount_in) * 100 if amount_in > 0 else 0

            # Calculate gas costs
            total_gas = best_buy['gas_estimate'] + best_sell['gas_estimate']
            gas_cost = self._estimate_gas_cost_usd(total_gas)

            # Net profit after gas
            profit_after_gas = (profit_percent / 100) * amount_in - gas_cost

            # Check if profitable
            if profit_percent >= self.min_profit_percent and gas_cost <= self.max_gas_cost_usd:
                opportunity = ArbitrageOpportunity(
                    token_path=[token_in, token_out, token_in],
                    dex_path=[best_buy['dex'], best_sell['dex']],
                    buy_dex=best_buy['dex'],
                    sell_dex=best_sell['dex'],
                    buy_price=best_buy['price'],
                    sell_price=1 / best_sell['price'] if best_sell['price'] > 0 else 0,
                    profit_percent=profit_percent,
                    profit_after_gas=profit_after_gas,
                    amount_in=amount_in,
                    amount_out=final_amount,
                    gas_cost_usd=gas_cost,
                    confidence=self._calculate_confidence(best_buy, best_sell),
                    execution_route=[
                        {
                            'step': 1,
                            'action': 'buy',
                            'dex': best_buy['dex'],
                            'token_in': token_in,
                            'token_out': token_out,
                            'amount_in': amount_in,
                            'amount_out': best_buy['amount_out']
                        },
                        {
                            'step': 2,
                            'action': 'sell',
                            'dex': best_sell['dex'],
                            'token_in': token_out,
                            'token_out': token_in,
                            'amount_in': best_buy['amount_out'],
                            'amount_out': best_sell['amount_out']
                        }
                    ]
                )

                opportunities.append(opportunity)
                logger.info(f"Found arbitrage: {profit_percent:.2f}% profit on {token_in}/{token_out}")

        except Exception as e:
            logger.error(f"Error detecting simple arbitrage: {e}")

        return opportunities

    def detect_triangular_arbitrage(
        self,
        token_a: str,
        token_b: str,
        token_c: str,
        amount_in: float = 1.0
    ) -> List[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage: A -> B -> C -> A

        Args:
            token_a: First token
            token_b: Second token
            token_c: Third token
            amount_in: Amount to start with

        Returns:
            List of triangular arbitrage opportunities
        """
        opportunities = []

        try:
            # Path: A -> B -> C -> A
            routes = [
                (token_a, token_b, token_c, token_a),
                (token_a, token_c, token_b, token_a),
            ]

            for route in routes:
                # Execute three swaps
                quotes = []
                current_amount = amount_in

                for i in range(3):
                    token_in = route[i]
                    token_out = route[i + 1]

                    # Get best quote for this leg
                    leg_quotes = self._get_all_quotes(token_in, token_out, current_amount)
                    if not leg_quotes:
                        break

                    best_quote = max(leg_quotes, key=lambda q: q['amount_out'])
                    quotes.append(best_quote)
                    current_amount = best_quote['amount_out']

                # Check if we completed all three legs
                if len(quotes) == 3:
                    final_amount = quotes[-1]['amount_out']
                    profit = final_amount - amount_in
                    profit_percent = (profit / amount_in) * 100 if amount_in > 0 else 0

                    # Calculate total gas
                    total_gas = sum(q['gas_estimate'] for q in quotes)
                    gas_cost = self._estimate_gas_cost_usd(total_gas)
                    profit_after_gas = (profit_percent / 100) * amount_in - gas_cost

                    # Check profitability
                    if profit_percent >= self.min_profit_percent and gas_cost <= self.max_gas_cost_usd:
                        opportunity = ArbitrageOpportunity(
                            token_path=list(route),
                            dex_path=[q['dex'] for q in quotes],
                            buy_dex=quotes[0]['dex'],
                            sell_dex=quotes[-1]['dex'],
                            buy_price=quotes[0]['price'],
                            sell_price=quotes[-1]['price'],
                            profit_percent=profit_percent,
                            profit_after_gas=profit_after_gas,
                            amount_in=amount_in,
                            amount_out=final_amount,
                            gas_cost_usd=gas_cost,
                            confidence=self._calculate_triangular_confidence(quotes),
                            execution_route=[
                                {
                                    'step': i + 1,
                                    'action': 'swap',
                                    'dex': quotes[i]['dex'],
                                    'token_in': route[i],
                                    'token_out': route[i + 1],
                                    'amount_in': quotes[i]['amount_in'],
                                    'amount_out': quotes[i]['amount_out']
                                }
                                for i in range(3)
                            ]
                        )

                        opportunities.append(opportunity)
                        logger.info(f"Found triangular arbitrage: {profit_percent:.2f}% on {route}")

        except Exception as e:
            logger.error(f"Error detecting triangular arbitrage: {e}")

        return opportunities

    def scan_all_pairs(self, amount_in: float = 1.0) -> List[ArbitrageOpportunity]:
        """
        Scan all configured token pairs for arbitrage opportunities.

        Args:
            amount_in: Amount to use for detection

        Returns:
            List of all detected opportunities
        """
        all_opportunities = []

        logger.info(f"Scanning {len(self.token_pairs)} pairs for arbitrage...")

        for token_in, token_out in self.token_pairs:
            # Simple arbitrage
            opportunities = self.detect_simple_arbitrage(token_in, token_out, amount_in)
            all_opportunities.extend(opportunities)

        # Sort by profit
        all_opportunities.sort(key=lambda o: o.profit_after_gas, reverse=True)

        logger.info(f"Found {len(all_opportunities)} arbitrage opportunities")

        return all_opportunities

    def _get_all_quotes(self, token_in: str, token_out: str, amount_in: float) -> List[Dict]:
        """Get quotes from all available DEXs."""
        quotes = []

        # Uniswap V3
        try:
            uni_quote = self.uniswap_v3.get_best_quote(token_in, token_out, amount_in)
            if uni_quote:
                quotes.append(uni_quote)
        except Exception as e:
            logger.debug(f"Uniswap V3 quote failed: {e}")

        # Curve (for stablecoins)
        try:
            curve_quote = self.curve.quote_price(token_in, token_out, amount_in)
            if curve_quote:
                quotes.append(curve_quote)
        except Exception as e:
            logger.debug(f"Curve quote failed: {e}")

        return quotes

    def _estimate_gas_cost_usd(self, gas_units: int, eth_price_usd: float = 2000.0) -> float:
        """
        Estimate gas cost in USD.

        Args:
            gas_units: Estimated gas units
            eth_price_usd: ETH price in USD

        Returns:
            Gas cost in USD
        """
        try:
            gas_price_gwei = 30  # Rough estimate
            gas_price_eth = (gas_units * gas_price_gwei) / 1e9
            return gas_price_eth * eth_price_usd
        except Exception as e:
            logger.error(f"Error estimating gas cost: {e}")
            return 0.0

    def _calculate_confidence(self, buy_quote: Dict, sell_quote: Dict) -> float:
        """
        Calculate confidence score for arbitrage opportunity.

        Factors:
        - Price impact
        - Liquidity
        - Historical reliability of DEXs
        """
        confidence = 1.0

        # Reduce confidence for high price impact
        buy_impact = buy_quote.get('price_impact', 0)
        sell_impact = sell_quote.get('price_impact', 0)
        total_impact = buy_impact + sell_impact

        if total_impact > 1.0:
            confidence *= 0.7
        elif total_impact > 0.5:
            confidence *= 0.85

        # Boost confidence for low-slippage DEXs (Curve)
        if buy_quote['dex'] == 'curve_finance' or sell_quote['dex'] == 'curve_finance':
            confidence *= 1.1

        # Cap at 1.0
        return min(confidence, 1.0)

    def _calculate_triangular_confidence(self, quotes: List[Dict]) -> float:
        """Calculate confidence for triangular arbitrage."""
        confidence = 1.0

        # Average price impact across all legs
        avg_impact = sum(q.get('price_impact', 0) for q in quotes) / len(quotes)

        if avg_impact > 1.0:
            confidence *= 0.6
        elif avg_impact > 0.5:
            confidence *= 0.8

        # Reduce confidence for multi-leg trades (more risk)
        confidence *= 0.9

        return min(confidence, 1.0)

    def format_opportunity(self, opp: ArbitrageOpportunity) -> str:
        """Format arbitrage opportunity for display."""
        route_str = " -> ".join([f"{step['dex']}({step['token_in'][:6]}→{step['token_out'][:6]})"
                                 for step in opp.execution_route])

        return f"""
Arbitrage Opportunity:
  Route: {route_str}
  Profit: {opp.profit_percent:.2f}% (${opp.profit_after_gas:.2f} after gas)
  Amount In: {opp.amount_in:.4f}
  Amount Out: {opp.amount_out:.4f}
  Gas Cost: ${opp.gas_cost_usd:.2f}
  Confidence: {opp.confidence:.0%}
  Buy: {opp.buy_dex} @ {opp.buy_price:.6f}
  Sell: {opp.sell_dex} @ {opp.sell_price:.6f}
        """


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🔍 Cross-DEX Arbitrage Detector Demo")
    print("=" * 60)

    # Initialize
    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')
    detector = ArbitrageDetector(
        rpc_url=rpc_url,
        min_profit_percent=0.5,  # 0.5% minimum profit
        max_gas_cost_usd=50.0
    )

    # Scan for opportunities
    print("\n🔍 Scanning for arbitrage opportunities...")
    opportunities = detector.scan_all_pairs(amount_in=1000.0)  # $1000 test amount

    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:5], 1):  # Show top 5
            print(f"\n{'='*60}")
            print(f"Opportunity #{i}:")
            print(detector.format_opportunity(opp))
    else:
        print("\n❌ No profitable arbitrage opportunities found")
        print("   (Market is likely efficient or gas costs too high)")

    print("\n✅ Arbitrage detector demo complete!")
