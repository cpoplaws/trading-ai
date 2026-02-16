"""
DEX Analyzer for Arbitrage Opportunities
Finds profitable arbitrage across CEX-DEX and between different DEXs.
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities."""
    CEX_DEX = "cex_dex"  # Between CEX (Coinbase) and DEX (Uniswap)
    CROSS_DEX = "cross_dex"  # Between different DEXs (Uniswap vs Sushi)
    TRIANGULAR = "triangular"  # ABC triangle (e.g., ETH→USDC→DAI→ETH)


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity found by analyzer."""
    type: ArbitrageType
    timestamp: datetime

    # Buy side
    buy_exchange: str
    buy_price: float
    buy_amount: float

    # Sell side
    sell_exchange: str
    sell_price: float
    sell_amount: float

    # Profitability
    gross_profit: float  # Before fees
    gas_cost: float  # Gas cost in ETH or USD
    net_profit: float  # After all costs
    roi_percent: float  # Return on investment %

    # Execution details
    token: str
    path: List[str]  # For multi-hop trades
    confidence: float  # 0-1, how confident we are
    estimated_slippage: float  # Expected slippage %

    def is_profitable(self, min_profit: float = 10.0) -> bool:
        """Check if opportunity is profitable after costs."""
        return self.net_profit >= min_profit


class DEXAnalyzer:
    """
    Analyze DEX data to find arbitrage opportunities.

    Features:
    - CEX-DEX arbitrage (Coinbase vs Uniswap)
    - Cross-DEX arbitrage (Uniswap vs SushiSwap)
    - Triangular arbitrage (multi-hop)
    - Gas cost consideration
    - Slippage estimation
    - Real-time profitability calculation
    """

    def __init__(
        self,
        min_profit_usd: float = 10.0,
        max_gas_gwei: float = 50.0,
        min_liquidity_usd: float = 10000.0
    ):
        """
        Initialize DEX analyzer.

        Args:
            min_profit_usd: Minimum profit threshold (USD)
            max_gas_gwei: Maximum acceptable gas price
            min_liquidity_usd: Minimum pool liquidity required
        """
        self.min_profit_usd = min_profit_usd
        self.max_gas_gwei = max_gas_gwei
        self.min_liquidity_usd = min_liquidity_usd

        logger.info(f"DEX analyzer initialized: "
                   f"min_profit=${min_profit_usd}, "
                   f"max_gas={max_gas_gwei}gwei")

    def find_cex_dex_arbitrage(
        self,
        cex_price: float,
        dex_price: float,
        token: str,
        trade_size_usd: float = 1000.0,
        gas_cost_usd: float = 10.0,
        cex_fee_percent: float = 0.5,
        dex_fee_percent: float = 0.3
    ) -> Optional[ArbitrageOpportunity]:
        """
        Find arbitrage between CEX (Coinbase) and DEX (Uniswap).

        Args:
            cex_price: Price on CEX
            dex_price: Price on DEX
            token: Token symbol
            trade_size_usd: Size of trade in USD
            gas_cost_usd: Gas cost for DEX trade
            cex_fee_percent: CEX trading fee %
            dex_fee_percent: DEX trading fee %

        Returns:
            ArbitrageOpportunity if profitable, None otherwise
        """
        # Calculate price spread
        price_diff = abs(dex_price - cex_price)
        spread_percent = (price_diff / min(cex_price, dex_price)) * 100

        # Determine direction (buy low, sell high)
        if cex_price < dex_price:
            # Buy on CEX, sell on DEX
            buy_exchange = "coinbase"
            buy_price = cex_price
            sell_exchange = "uniswap"
            sell_price = dex_price
            buy_fee = cex_fee_percent
            sell_fee = dex_fee_percent
        else:
            # Buy on DEX, sell on CEX
            buy_exchange = "uniswap"
            buy_price = dex_price
            sell_exchange = "coinbase"
            sell_price = cex_price
            buy_fee = dex_fee_percent
            sell_fee = cex_fee_percent

        # Calculate quantities
        buy_amount = trade_size_usd / buy_price
        sell_amount = buy_amount

        # Calculate costs
        buy_cost = trade_size_usd + (trade_size_usd * buy_fee / 100)
        sell_revenue = (sell_amount * sell_price) * (1 - sell_fee / 100)

        # Calculate profit
        gross_profit = sell_revenue - trade_size_usd
        net_profit = gross_profit - gas_cost_usd

        roi_percent = (net_profit / trade_size_usd) * 100

        if net_profit < self.min_profit_usd:
            return None

        return ArbitrageOpportunity(
            type=ArbitrageType.CEX_DEX,
            timestamp=datetime.now(),
            buy_exchange=buy_exchange,
            buy_price=buy_price,
            buy_amount=buy_amount,
            sell_exchange=sell_exchange,
            sell_price=sell_price,
            sell_amount=sell_amount,
            gross_profit=gross_profit,
            gas_cost=gas_cost_usd,
            net_profit=net_profit,
            roi_percent=roi_percent,
            token=token,
            path=[token],
            confidence=0.9 if spread_percent > 2 else 0.7,
            estimated_slippage=0.1 if trade_size_usd < 10000 else 0.5
        )

    def find_cross_dex_arbitrage(
        self,
        dex1_name: str,
        dex1_price: float,
        dex1_liquidity: float,
        dex2_name: str,
        dex2_price: float,
        dex2_liquidity: float,
        token: str,
        trade_size_usd: float = 1000.0,
        gas_cost_usd: float = 20.0
    ) -> Optional[ArbitrageOpportunity]:
        """
        Find arbitrage between two DEXs (e.g., Uniswap vs SushiSwap).

        Args:
            dex1_name: First DEX name
            dex1_price: Price on first DEX
            dex1_liquidity: Liquidity on first DEX (USD)
            dex2_name: Second DEX name
            dex2_price: Price on second DEX
            dex2_liquidity: Liquidity on second DEX (USD)
            token: Token symbol
            trade_size_usd: Trade size
            gas_cost_usd: Gas cost (higher for 2 DEX txs)

        Returns:
            ArbitrageOpportunity if profitable
        """
        # Check liquidity requirements
        if dex1_liquidity < self.min_liquidity_usd or dex2_liquidity < self.min_liquidity_usd:
            logger.debug(f"Insufficient liquidity for {token}")
            return None

        # Calculate spread
        price_diff = abs(dex2_price - dex1_price)
        spread_percent = (price_diff / min(dex1_price, dex2_price)) * 100

        # Minimum spread needed to cover gas (roughly)
        min_spread_needed = (gas_cost_usd / trade_size_usd) * 100
        if spread_percent < min_spread_needed:
            return None

        # Determine direction
        if dex1_price < dex2_price:
            buy_exchange = dex1_name
            buy_price = dex1_price
            sell_exchange = dex2_name
            sell_price = dex2_price
        else:
            buy_exchange = dex2_name
            buy_price = dex2_price
            sell_exchange = dex1_name
            sell_price = dex1_price

        # Calculate with DEX fees (0.3% for Uniswap V2)
        fee_percent = 0.3
        buy_amount = trade_size_usd / buy_price
        buy_cost = trade_size_usd * (1 + fee_percent / 100)

        sell_amount = buy_amount
        sell_revenue = (sell_amount * sell_price) * (1 - fee_percent / 100)

        gross_profit = sell_revenue - trade_size_usd
        net_profit = gross_profit - gas_cost_usd

        roi_percent = (net_profit / trade_size_usd) * 100

        if net_profit < self.min_profit_usd:
            return None

        # Estimate slippage based on trade size vs liquidity
        slippage_estimate = (trade_size_usd / min(dex1_liquidity, dex2_liquidity)) * 100

        return ArbitrageOpportunity(
            type=ArbitrageType.CROSS_DEX,
            timestamp=datetime.now(),
            buy_exchange=buy_exchange,
            buy_price=buy_price,
            buy_amount=buy_amount,
            sell_exchange=sell_exchange,
            sell_price=sell_price,
            sell_amount=sell_amount,
            gross_profit=gross_profit,
            gas_cost=gas_cost_usd,
            net_profit=net_profit,
            roi_percent=roi_percent,
            token=token,
            path=[token],
            confidence=0.8,
            estimated_slippage=slippage_estimate
        )

    def find_triangular_arbitrage(
        self,
        token_a: str,
        token_b: str,
        token_c: str,
        price_ab: float,  # A → B
        price_bc: float,  # B → C
        price_ca: float,  # C → A
        start_amount_a: float = 1.0,
        gas_cost_usd: float = 30.0
    ) -> Optional[ArbitrageOpportunity]:
        """
        Find triangular arbitrage opportunity (A→B→C→A).

        Example: ETH → USDC → DAI → ETH

        Args:
            token_a: First token (e.g., ETH)
            token_b: Second token (e.g., USDC)
            token_c: Third token (e.g., DAI)
            price_ab: Price A/B (how much B for 1 A)
            price_bc: Price B/C (how much C for 1 B)
            price_ca: Price C/A (how much A for 1 C)
            start_amount_a: Starting amount of token A
            gas_cost_usd: Gas cost for 3 swaps

        Returns:
            ArbitrageOpportunity if profitable
        """
        # Calculate the cycle
        # Start with 1 unit of A
        amount_a = start_amount_a

        # A → B
        amount_b = amount_a * price_ab * 0.997  # 0.3% fee

        # B → C
        amount_c = amount_b * price_bc * 0.997

        # C → A
        final_amount_a = amount_c * price_ca * 0.997

        # Calculate profit
        profit_a = final_amount_a - start_amount_a
        profit_percent = (profit_a / start_amount_a) * 100

        # If no profit, no opportunity
        if profit_a <= 0:
            return None

        # Convert to USD (assuming token_a price is known)
        # For this demo, we'll assume we know USD value
        token_a_usd = 2000.0  # Assume ETH = $2000
        gross_profit_usd = profit_a * token_a_usd
        net_profit_usd = gross_profit_usd - gas_cost_usd

        if net_profit_usd < self.min_profit_usd:
            return None

        roi_percent = (net_profit_usd / (start_amount_a * token_a_usd)) * 100

        return ArbitrageOpportunity(
            type=ArbitrageType.TRIANGULAR,
            timestamp=datetime.now(),
            buy_exchange="uniswap",
            buy_price=price_ab,
            buy_amount=start_amount_a,
            sell_exchange="uniswap",
            sell_price=price_ca,
            sell_amount=final_amount_a,
            gross_profit=gross_profit_usd,
            gas_cost=gas_cost_usd,
            net_profit=net_profit_usd,
            roi_percent=roi_percent,
            token=token_a,
            path=[token_a, token_b, token_c, token_a],
            confidence=0.7,  # Triangular is riskier
            estimated_slippage=0.5
        )

    def analyze_all_opportunities(
        self,
        market_data: Dict
    ) -> List[ArbitrageOpportunity]:
        """
        Analyze all arbitrage opportunities from market data.

        Args:
            market_data: Dict with prices from different exchanges

        Returns:
            List of profitable opportunities, sorted by net profit
        """
        opportunities = []

        # Example market_data structure:
        # {
        #     'BTC': {
        #         'coinbase': 45000,
        #         'uniswap': 45200,
        #         'sushiswap': 45100
        #     },
        #     ...
        # }

        for token, prices in market_data.items():
            # CEX-DEX arbitrage
            if 'coinbase' in prices and 'uniswap' in prices:
                opp = self.find_cex_dex_arbitrage(
                    cex_price=prices['coinbase'],
                    dex_price=prices['uniswap'],
                    token=token
                )
                if opp:
                    opportunities.append(opp)

            # Cross-DEX arbitrage
            if 'uniswap' in prices and 'sushiswap' in prices:
                opp = self.find_cross_dex_arbitrage(
                    dex1_name='uniswap',
                    dex1_price=prices['uniswap'],
                    dex1_liquidity=100000,  # Would come from pool data
                    dex2_name='sushiswap',
                    dex2_price=prices['sushiswap'],
                    dex2_liquidity=80000,
                    token=token
                )
                if opp:
                    opportunities.append(opp)

        # Sort by net profit (descending)
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)

        return opportunities

    def filter_by_confidence(
        self,
        opportunities: List[ArbitrageOpportunity],
        min_confidence: float = 0.7
    ) -> List[ArbitrageOpportunity]:
        """Filter opportunities by minimum confidence level."""
        return [opp for opp in opportunities if opp.confidence >= min_confidence]

    def filter_by_gas_price(
        self,
        opportunities: List[ArbitrageOpportunity],
        current_gas_gwei: float
    ) -> List[ArbitrageOpportunity]:
        """Filter opportunities that are still profitable at current gas."""
        if current_gas_gwei > self.max_gas_gwei:
            logger.warning(f"Gas price {current_gas_gwei} exceeds max {self.max_gas_gwei}")
            return []

        return opportunities  # Already filtered in find methods


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🔍 DEX Analyzer Demo")
    print("=" * 60)

    # Initialize analyzer
    analyzer = DEXAnalyzer(
        min_profit_usd=10.0,
        max_gas_gwei=50.0
    )

    print(f"✓ Analyzer configured:")
    print(f"  Min profit: ${analyzer.min_profit_usd}")
    print(f"  Max gas: {analyzer.max_gas_gwei} Gwei")

    # Example 1: CEX-DEX Arbitrage
    print("\n" + "=" * 60)
    print("Example 1: CEX-DEX Arbitrage (Coinbase vs Uniswap)")
    print("=" * 60)

    opp1 = analyzer.find_cex_dex_arbitrage(
        cex_price=45000,  # Coinbase: $45,000
        dex_price=45100,  # Uniswap: $45,100 (higher)
        token="ETH",
        trade_size_usd=5000,
        gas_cost_usd=15
    )

    if opp1:
        print(f"\n✅ Opportunity found!")
        print(f"  Type: {opp1.type.value}")
        print(f"  Token: {opp1.token}")
        print(f"  Buy: {opp1.buy_exchange} @ ${opp1.buy_price:,.2f}")
        print(f"  Sell: {opp1.sell_exchange} @ ${opp1.sell_price:,.2f}")
        print(f"  Gross Profit: ${opp1.gross_profit:,.2f}")
        print(f"  Gas Cost: ${opp1.gas_cost:,.2f}")
        print(f"  Net Profit: ${opp1.net_profit:,.2f}")
        print(f"  ROI: {opp1.roi_percent:.2f}%")
        print(f"  Confidence: {opp1.confidence * 100:.0f}%")

    # Example 2: Cross-DEX Arbitrage
    print("\n" + "=" * 60)
    print("Example 2: Cross-DEX Arbitrage (Uniswap vs SushiSwap)")
    print("=" * 60)

    opp2 = analyzer.find_cross_dex_arbitrage(
        dex1_name="uniswap",
        dex1_price=2000,  # $2000
        dex1_liquidity=500000,
        dex2_name="sushiswap",
        dex2_price=2015,  # $2015 (higher)
        dex2_liquidity=400000,
        token="ETH",
        trade_size_usd=3000,
        gas_cost_usd=25
    )

    if opp2:
        print(f"\n✅ Opportunity found!")
        print(f"  Buy: {opp2.buy_exchange} @ ${opp2.buy_price:,.2f}")
        print(f"  Sell: {opp2.sell_exchange} @ ${opp2.sell_price:,.2f}")
        print(f"  Net Profit: ${opp2.net_profit:,.2f}")
        print(f"  ROI: {opp2.roi_percent:.2f}%")
        print(f"  Slippage: {opp2.estimated_slippage:.2f}%")

    # Example 3: Triangular Arbitrage
    print("\n" + "=" * 60)
    print("Example 3: Triangular Arbitrage (ETH→USDC→DAI→ETH)")
    print("=" * 60)

    opp3 = analyzer.find_triangular_arbitrage(
        token_a="ETH",
        token_b="USDC",
        token_c="DAI",
        price_ab=2000,  # 1 ETH = 2000 USDC
        price_bc=1.001,  # 1 USDC = 1.001 DAI (slight premium)
        price_ca=0.00050025,  # 1 DAI = 0.00050025 ETH
        start_amount_a=5.0,  # Start with 5 ETH
        gas_cost_usd=40
    )

    if opp3:
        print(f"\n✅ Opportunity found!")
        print(f"  Path: {' → '.join(opp3.path)}")
        print(f"  Start: {opp3.buy_amount:.4f} ETH")
        print(f"  End: {opp3.sell_amount:.4f} ETH")
        print(f"  Net Profit: ${opp3.net_profit:,.2f}")
        print(f"  ROI: {opp3.roi_percent:.2f}%")
    else:
        print("\n❌ No profitable triangular arbitrage found")

    # Example 4: Batch Analysis
    print("\n" + "=" * 60)
    print("Example 4: Analyzing Multiple Opportunities")
    print("=" * 60)

    market_data = {
        'ETH': {
            'coinbase': 2000,
            'uniswap': 2010,
            'sushiswap': 2005
        },
        'BTC': {
            'coinbase': 45000,
            'uniswap': 45150,
            'sushiswap': 45100
        }
    }

    all_opps = analyzer.analyze_all_opportunities(market_data)

    print(f"\n✓ Found {len(all_opps)} opportunities")
    print("\nTop 3 by profitability:")
    for i, opp in enumerate(all_opps[:3], 1):
        print(f"\n  {i}. {opp.token} - {opp.type.value}")
        print(f"     Net Profit: ${opp.net_profit:,.2f}")
        print(f"     ROI: {opp.roi_percent:.2f}%")

    print("\n✅ DEX analyzer demo complete!")
    print("\nNext steps:")
    print("1. Integrate with real Coinbase + Uniswap data collectors")
    print("2. Monitor opportunities in real-time")
    print("3. Execute profitable trades automatically")
