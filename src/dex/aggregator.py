"""
DEX Aggregator and Smart Router
Compares prices across multiple DEXs and finds optimal execution paths.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)


class DEX(Enum):
    """Supported DEXs."""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP = "sushiswap"
    CURVE = "curve"
    BALANCER = "balancer"


@dataclass
class Pool:
    """DEX liquidity pool."""
    dex: DEX
    token_a: str
    token_b: str
    reserve_a: float
    reserve_b: float
    fee: float  # Fee percentage (0.003 = 0.3%)
    liquidity_usd: float = 0.0

    @property
    def price_a_to_b(self) -> float:
        """Price of token A in terms of token B."""
        return self.reserve_b / self.reserve_a if self.reserve_a > 0 else 0

    @property
    def price_b_to_a(self) -> float:
        """Price of token B in terms of token A."""
        return self.reserve_a / self.reserve_b if self.reserve_b > 0 else 0


@dataclass
class Quote:
    """Price quote from a DEX."""
    dex: DEX
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    price: float
    price_impact: float
    fee: float
    gas_cost: float
    route: List[str] = field(default_factory=list)

    @property
    def net_amount_out(self) -> float:
        """Amount out after gas costs (in USD)."""
        return self.amount_out - self.gas_cost

    @property
    def effective_price(self) -> float:
        """Effective price including fees and slippage."""
        return self.amount_in / self.amount_out if self.amount_out > 0 else 0


@dataclass
class SplitRoute:
    """Split routing across multiple DEXs."""
    quotes: List[Tuple[Quote, float]]  # (quote, percentage)
    total_amount_in: float
    total_amount_out: float
    total_gas_cost: float
    avg_price: float
    total_price_impact: float

    @property
    def net_amount_out(self) -> float:
        """Total output after gas."""
        return self.total_amount_out - self.total_gas_cost


class DEXAggregator:
    """
    DEX Aggregator and Smart Router

    Finds best execution paths across multiple DEXs.
    """

    # Gas costs per DEX (USD)
    GAS_COSTS = {
        DEX.UNISWAP_V2: 5.0,
        DEX.UNISWAP_V3: 7.0,
        DEX.SUSHISWAP: 5.0,
        DEX.CURVE: 8.0,
        DEX.BALANCER: 10.0
    }

    def __init__(self):
        """Initialize DEX aggregator."""
        self.pools: Dict[Tuple[str, str, DEX], Pool] = {}
        logger.info("DEX aggregator initialized")

    def add_pool(self, pool: Pool):
        """
        Add liquidity pool.

        Args:
            pool: Pool to add
        """
        # Store both directions
        key1 = (pool.token_a, pool.token_b, pool.dex)
        key2 = (pool.token_b, pool.token_a, pool.dex)

        self.pools[key1] = pool
        self.pools[key2] = Pool(
            dex=pool.dex,
            token_a=pool.token_b,
            token_b=pool.token_a,
            reserve_a=pool.reserve_b,
            reserve_b=pool.reserve_a,
            fee=pool.fee,
            liquidity_usd=pool.liquidity_usd
        )

        logger.debug(
            f"Added pool: {pool.dex.value} {pool.token_a}/{pool.token_b} "
            f"(${pool.liquidity_usd:,.0f} liquidity)"
        )

    def get_quote(
        self,
        dex: DEX,
        token_in: str,
        token_out: str,
        amount_in: float
    ) -> Optional[Quote]:
        """
        Get price quote from specific DEX.

        Args:
            dex: DEX to query
            token_in: Input token
            token_out: Output token
            amount_in: Input amount

        Returns:
            Quote if pool exists
        """
        pool = self.pools.get((token_in, token_out, dex))
        if not pool:
            return None

        # Calculate output using constant product formula
        # x * y = k
        # amount_out = (reserve_out * amount_in) / (reserve_in + amount_in)
        amount_in_with_fee = amount_in * (1 - pool.fee)
        amount_out = (pool.reserve_b * amount_in_with_fee) / (pool.reserve_a + amount_in_with_fee)

        # Calculate price and price impact
        price = amount_out / amount_in if amount_in > 0 else 0
        ideal_price = pool.price_a_to_b
        price_impact = abs((price - ideal_price) / ideal_price) * 100 if ideal_price > 0 else 0

        # Get gas cost
        gas_cost = self.GAS_COSTS.get(dex, 5.0)

        return Quote(
            dex=dex,
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=amount_out,
            price=price,
            price_impact=price_impact,
            fee=amount_in * pool.fee,
            gas_cost=gas_cost,
            route=[token_in, token_out]
        )

    def get_best_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        exclude_dexs: Optional[List[DEX]] = None
    ) -> Optional[Quote]:
        """
        Find best quote across all DEXs.

        Args:
            token_in: Input token
            token_out: Output token
            amount_in: Input amount
            exclude_dexs: DEXs to exclude

        Returns:
            Best quote (highest net output)
        """
        exclude_dexs = exclude_dexs or []

        quotes = []
        for dex in DEX:
            if dex in exclude_dexs:
                continue

            quote = self.get_quote(dex, token_in, token_out, amount_in)
            if quote:
                quotes.append(quote)

        if not quotes:
            return None

        # Sort by net amount out (after gas)
        quotes.sort(key=lambda q: q.net_amount_out, reverse=True)

        best = quotes[0]
        logger.info(
            f"Best quote: {best.dex.value} - "
            f"{best.amount_out:.6f} {token_out} "
            f"(${best.net_amount_out:.2f} after gas, "
            f"{best.price_impact:.2f}% impact)"
        )

        return best

    def get_all_quotes(
        self,
        token_in: str,
        token_out: str,
        amount_in: float
    ) -> List[Quote]:
        """
        Get quotes from all DEXs for comparison.

        Args:
            token_in: Input token
            token_out: Output token
            amount_in: Input amount

        Returns:
            List of quotes sorted by best price
        """
        quotes = []
        for dex in DEX:
            quote = self.get_quote(dex, token_in, token_out, amount_in)
            if quote:
                quotes.append(quote)

        # Sort by net amount out
        quotes.sort(key=lambda q: q.net_amount_out, reverse=True)
        return quotes

    def get_split_route(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        max_splits: int = 3
    ) -> Optional[SplitRoute]:
        """
        Find optimal split routing across multiple DEXs.

        Splits large orders to reduce price impact.

        Args:
            token_in: Input token
            token_out: Output token
            amount_in: Input amount
            max_splits: Maximum number of DEXs to split across

        Returns:
            Optimal split route
        """
        # Get all available quotes for full amount
        all_quotes = self.get_all_quotes(token_in, token_out, amount_in)

        if not all_quotes:
            return None

        if len(all_quotes) == 1:
            # Only one option, no splitting possible
            quote = all_quotes[0]
            return SplitRoute(
                quotes=[(quote, 1.0)],
                total_amount_in=amount_in,
                total_amount_out=quote.amount_out,
                total_gas_cost=quote.gas_cost,
                avg_price=quote.price,
                total_price_impact=quote.price_impact
            )

        # Try different split combinations
        best_route = None
        best_net_output = 0

        # Consider top DEXs only
        top_dexs = all_quotes[:min(max_splits, len(all_quotes))]

        # Single route (no split)
        single_best = all_quotes[0]
        best_route = SplitRoute(
            quotes=[(single_best, 1.0)],
            total_amount_in=amount_in,
            total_amount_out=single_best.amount_out,
            total_gas_cost=single_best.gas_cost,
            avg_price=single_best.price,
            total_price_impact=single_best.price_impact
        )
        best_net_output = single_best.net_amount_out

        # Try 50/50 split between top 2
        if len(top_dexs) >= 2:
            split_amount = amount_in / 2
            q1 = self.get_quote(top_dexs[0].dex, token_in, token_out, split_amount)
            q2 = self.get_quote(top_dexs[1].dex, token_in, token_out, split_amount)

            if q1 and q2:
                total_out = q1.amount_out + q2.amount_out
                total_gas = q1.gas_cost + q2.gas_cost
                net_out = total_out - total_gas

                if net_out > best_net_output:
                    best_route = SplitRoute(
                        quotes=[(q1, 0.5), (q2, 0.5)],
                        total_amount_in=amount_in,
                        total_amount_out=total_out,
                        total_gas_cost=total_gas,
                        avg_price=total_out / amount_in,
                        total_price_impact=(q1.price_impact + q2.price_impact) / 2
                    )
                    best_net_output = net_out

        # Try 33/33/33 split between top 3
        if len(top_dexs) >= 3:
            split_amount = amount_in / 3
            quotes = []
            for dex_quote in top_dexs[:3]:
                q = self.get_quote(dex_quote.dex, token_in, token_out, split_amount)
                if q:
                    quotes.append(q)

            if len(quotes) == 3:
                total_out = sum(q.amount_out for q in quotes)
                total_gas = sum(q.gas_cost for q in quotes)
                net_out = total_out - total_gas

                if net_out > best_net_output:
                    best_route = SplitRoute(
                        quotes=[(q, 1/3) for q in quotes],
                        total_amount_in=amount_in,
                        total_amount_out=total_out,
                        total_gas_cost=total_gas,
                        avg_price=total_out / amount_in,
                        total_price_impact=sum(q.price_impact for q in quotes) / 3
                    )

        if best_route:
            logger.info(
                f"Optimal route: {len(best_route.quotes)} DEX(s), "
                f"output: {best_route.total_amount_out:.6f} {token_out}, "
                f"impact: {best_route.total_price_impact:.2f}%"
            )

        return best_route

    def compare_dexs(
        self,
        token_in: str,
        token_out: str,
        amount_in: float
    ) -> str:
        """
        Generate comparison table of all DEX quotes.

        Args:
            token_in: Input token
            token_out: Output token
            amount_in: Input amount

        Returns:
            Formatted comparison table
        """
        quotes = self.get_all_quotes(token_in, token_out, amount_in)

        if not quotes:
            return "No quotes available"

        table = f"""
╔══════════════════════════════════════════════════════════════╗
║            DEX PRICE COMPARISON: {token_in}/{token_out}                ║
║            Trade Size: {amount_in:,.2f} {token_in}                          ║
╠══════════════════════════════════════════════════════════════╣
"""

        for i, quote in enumerate(quotes, 1):
            best = "⭐ " if i == 1 else "   "
            savings = ""
            if i > 1:
                diff = quotes[0].net_amount_out - quote.net_amount_out
                savings = f" (-${diff:.2f})"

            table += f"""║ {best}{quote.dex.value.upper():<20}                              ║
║   Output:   {quote.amount_out:>12.6f} {token_out:<10}               ║
║   Gas Cost: ${quote.gas_cost:>6.2f}   Net: ${quote.net_amount_out:>10.2f}{savings:<10} ║
║   Impact:   {quote.price_impact:>6.2f}%   Fee: ${quote.fee:>8.2f}                 ║
╠══════════════════════════════════════════════════════════════╣
"""

        table += "╚══════════════════════════════════════════════════════════════╝\n"

        if len(quotes) > 1:
            best_vs_worst = quotes[0].net_amount_out - quotes[-1].net_amount_out
            table += f"\nBest vs Worst: ${best_vs_worst:.2f} difference ({(best_vs_worst/quotes[-1].net_amount_out*100):.1f}%)\n"

        return table


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🔄 DEX Aggregator Demo")
    print("=" * 60)

    # Initialize aggregator
    aggregator = DEXAggregator()

    # Add liquidity pools (simulated data)
    print("\n1. Adding liquidity pools...")
    print("-" * 60)

    # Uniswap V2: ETH/USDC
    aggregator.add_pool(Pool(
        dex=DEX.UNISWAP_V2,
        token_a="ETH",
        token_b="USDC",
        reserve_a=1000.0,  # 1000 ETH
        reserve_b=2100000.0,  # $2.1M USDC
        fee=0.003,  # 0.3%
        liquidity_usd=4200000.0
    ))

    # Sushiswap: ETH/USDC (slightly worse price)
    aggregator.add_pool(Pool(
        dex=DEX.SUSHISWAP,
        token_a="ETH",
        token_b="USDC",
        reserve_a=500.0,
        reserve_b=1045000.0,
        fee=0.003,
        liquidity_usd=2090000.0
    ))

    # Uniswap V3: ETH/USDC (best price, concentrated liquidity)
    aggregator.add_pool(Pool(
        dex=DEX.UNISWAP_V3,
        token_a="ETH",
        token_b="USDC",
        reserve_a=2000.0,
        reserve_b=4220000.0,
        fee=0.0005,  # 0.05% (concentrated)
        liquidity_usd=8440000.0
    ))

    print("Added 3 pools")

    # Get best quote
    print("\n2. Finding Best Quote")
    print("-" * 60)
    print(f"Trade: 5 ETH → USDC")

    best_quote = aggregator.get_best_quote("ETH", "USDC", 5.0)
    if best_quote:
        print(f"\nBest: {best_quote.dex.value}")
        print(f"  Output: {best_quote.amount_out:,.2f} USDC")
        print(f"  Price: ${best_quote.price:,.2f} per ETH")
        print(f"  Impact: {best_quote.price_impact:.3f}%")
        print(f"  Fee: ${best_quote.fee:.2f}")
        print(f"  Gas: ${best_quote.gas_cost:.2f}")
        print(f"  Net: ${best_quote.net_amount_out:,.2f}")

    # Compare all DEXs
    print("\n3. DEX Comparison")
    print("-" * 60)
    print(aggregator.compare_dexs("ETH", "USDC", 5.0))

    # Try split routing for large order
    print("4. Split Routing (Large Order)")
    print("-" * 60)
    print("Trade: 50 ETH → USDC (large order)")

    split_route = aggregator.get_split_route("ETH", "USDC", 50.0)
    if split_route:
        print(f"\nOptimal Route: Split across {len(split_route.quotes)} DEX(s)")
        for quote, pct in split_route.quotes:
            amount = split_route.total_amount_in * pct
            print(f"  • {quote.dex.value}: {pct*100:.0f}% ({amount:.2f} ETH)")
            print(f"    → {quote.amount_out:,.2f} USDC (impact: {quote.price_impact:.2f}%)")

        print(f"\nTotal Output: ${split_route.total_amount_out:,.2f}")
        print(f"Gas Cost: ${split_route.total_gas_cost:.2f}")
        print(f"Net Output: ${split_route.net_amount_out:,.2f}")
        print(f"Avg Impact: {split_route.total_price_impact:.2f}%")

        # Compare with single route
        single_quote = aggregator.get_best_quote("ETH", "USDC", 50.0)
        if single_quote:
            improvement = split_route.net_amount_out - single_quote.net_amount_out
            print(f"\nSplit vs Single: +${improvement:.2f} ({(improvement/single_quote.net_amount_out*100):.2f}% better)")

    print("\n✅ DEX aggregator demo complete!")
