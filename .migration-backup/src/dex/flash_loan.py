"""
Flash Loan Arbitrage System
Detects and simulates profitable arbitrage opportunities using flash loans.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .aggregator import DEXAggregator, DEX, Quote, Pool

logger = logging.getLogger(__name__)


class FlashLoanProvider(Enum):
    """Flash loan providers."""
    AAVE = "aave"
    DYDX = "dydx"
    UNISWAP = "uniswap"
    BALANCER = "balancer"


@dataclass
class FlashLoanTerms:
    """Flash loan terms and costs."""
    provider: FlashLoanProvider
    max_loan_amount: float  # Maximum borrowable amount
    fee_percent: float  # Fee percentage (0.0009 = 0.09%)
    gas_cost: float  # Estimated gas cost in USD

    def calculate_fee(self, amount: float) -> float:
        """Calculate flash loan fee."""
        return amount * self.fee_percent


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    opportunity_id: str
    timestamp: datetime

    # Trade details
    token: str
    amount: float

    # DEX routing
    buy_dex: DEX
    sell_dex: DEX
    buy_price: float
    sell_price: float
    price_difference: float
    price_difference_percent: float

    # Profitability
    buy_cost: float
    sell_proceeds: float
    gross_profit: float

    # Costs
    flash_loan_fee: float
    gas_cost: float
    dex_fees: float
    total_costs: float

    # Net result
    net_profit: float
    roi: float

    # Flash loan details
    flash_loan_provider: FlashLoanProvider
    loan_amount: float

    # Execution details
    is_profitable: bool
    min_profit_threshold: float

    metadata: Dict = field(default_factory=dict)


class FlashLoanArbitrage:
    """
    Flash Loan Arbitrage System

    Detects and executes profitable arbitrage across DEXs using flash loans.

    Flash loans allow borrowing large amounts without collateral, enabling
    arbitrage that wouldn't be possible with limited capital.
    """

    # Flash loan provider terms
    FLASH_LOAN_TERMS = {
        FlashLoanProvider.AAVE: FlashLoanTerms(
            provider=FlashLoanProvider.AAVE,
            max_loan_amount=10_000_000,  # $10M
            fee_percent=0.0009,  # 0.09%
            gas_cost=15.0  # Higher gas for flash loan
        ),
        FlashLoanProvider.DYDX: FlashLoanTerms(
            provider=FlashLoanProvider.DYDX,
            max_loan_amount=50_000_000,  # $50M
            fee_percent=0.0,  # No fee!
            gas_cost=20.0
        ),
        FlashLoanProvider.UNISWAP: FlashLoanTerms(
            provider=FlashLoanProvider.UNISWAP,
            max_loan_amount=5_000_000,  # $5M
            fee_percent=0.0005,  # 0.05%
            gas_cost=18.0
        )
    }

    def __init__(
        self,
        aggregator: DEXAggregator,
        min_profit_usd: float = 50.0,
        min_roi_percent: float = 0.5
    ):
        """
        Initialize flash loan arbitrage system.

        Args:
            aggregator: DEX aggregator for price comparison
            min_profit_usd: Minimum net profit to execute (USD)
            min_roi_percent: Minimum ROI percentage
        """
        self.aggregator = aggregator
        self.min_profit_usd = min_profit_usd
        self.min_roi_percent = min_roi_percent

        # Tracking
        self.opportunities: List[ArbitrageOpportunity] = []
        self.executed_arbitrages: List[ArbitrageOpportunity] = []
        self.opportunity_counter = 0

        # Statistics
        self.total_profit = 0.0
        self.total_volume = 0.0
        self.success_rate = 0.0

        logger.info(
            f"Flash loan arbitrage initialized "
            f"(min_profit=${min_profit_usd}, min_roi={min_roi_percent}%)"
        )

    def _generate_opportunity_id(self) -> str:
        """Generate unique opportunity ID."""
        self.opportunity_counter += 1
        return f"ARB-{self.opportunity_counter:06d}"

    def detect_opportunity(
        self,
        token: str,
        base_token: str = "USDC",
        loan_amounts: Optional[List[float]] = None
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunity for a token.

        Strategy:
        1. Check prices across all DEXs
        2. Find best buy and sell prices
        3. Calculate if price difference covers all costs
        4. Use flash loan to maximize profit

        Args:
            token: Token to arbitrage
            base_token: Quote token (usually stablecoin)
            loan_amounts: Amounts to test (defaults to [10k, 50k, 100k])

        Returns:
            ArbitrageOpportunity if profitable
        """
        loan_amounts = loan_amounts or [10000, 50000, 100000]

        best_opportunity = None
        best_net_profit = 0

        for loan_amount in loan_amounts:
            # Get quotes from all DEXs
            buy_quotes = self.aggregator.get_all_quotes(base_token, token, loan_amount)

            if len(buy_quotes) < 2:
                continue

            # Best buy (highest output) and sell (we'd sell on buy DEX)
            for buy_quote in buy_quotes:
                # Calculate how much token we'd get
                token_amount = buy_quote.amount_out

                # Get sell quotes (reverse direction)
                sell_quotes = self.aggregator.get_all_quotes(token, base_token, token_amount)

                for sell_quote in sell_quotes:
                    # Skip same DEX
                    if buy_quote.dex == sell_quote.dex:
                        continue

                    # Calculate profitability
                    buy_cost = loan_amount
                    sell_proceeds = sell_quote.amount_out
                    gross_profit = sell_proceeds - buy_cost

                    # Skip if no gross profit
                    if gross_profit <= 0:
                        continue

                    # Calculate all costs
                    dex_fees = buy_quote.fee + sell_quote.fee
                    gas_cost = buy_quote.gas_cost + sell_quote.gas_cost

                    # Choose best flash loan provider
                    best_provider = self._choose_flash_loan_provider(loan_amount)
                    if not best_provider:
                        continue

                    terms = self.FLASH_LOAN_TERMS[best_provider]
                    flash_loan_fee = terms.calculate_fee(loan_amount)
                    flash_loan_gas = terms.gas_cost

                    total_costs = dex_fees + gas_cost + flash_loan_fee + flash_loan_gas
                    net_profit = gross_profit - total_costs
                    roi = (net_profit / loan_amount) * 100

                    # Check if profitable
                    is_profitable = (
                        net_profit >= self.min_profit_usd and
                        roi >= self.min_roi_percent
                    )

                    if is_profitable and net_profit > best_net_profit:
                        price_diff = sell_quote.price - buy_quote.price
                        price_diff_pct = (price_diff / buy_quote.price) * 100

                        opportunity = ArbitrageOpportunity(
                            opportunity_id=self._generate_opportunity_id(),
                            timestamp=datetime.now(),
                            token=token,
                            amount=token_amount,
                            buy_dex=buy_quote.dex,
                            sell_dex=sell_quote.dex,
                            buy_price=buy_quote.price,
                            sell_price=sell_quote.price,
                            price_difference=price_diff,
                            price_difference_percent=price_diff_pct,
                            buy_cost=buy_cost,
                            sell_proceeds=sell_proceeds,
                            gross_profit=gross_profit,
                            flash_loan_fee=flash_loan_fee,
                            gas_cost=gas_cost + flash_loan_gas,
                            dex_fees=dex_fees,
                            total_costs=total_costs,
                            net_profit=net_profit,
                            roi=roi,
                            flash_loan_provider=best_provider,
                            loan_amount=loan_amount,
                            is_profitable=is_profitable,
                            min_profit_threshold=self.min_profit_usd,
                            metadata={
                                'buy_price_impact': buy_quote.price_impact,
                                'sell_price_impact': sell_quote.price_impact
                            }
                        )

                        best_opportunity = opportunity
                        best_net_profit = net_profit

        if best_opportunity:
            self.opportunities.append(best_opportunity)
            logger.info(
                f"💰 Arbitrage opportunity detected: {best_opportunity.opportunity_id} "
                f"| {best_opportunity.buy_dex.value} → {best_opportunity.sell_dex.value} "
                f"| Net profit: ${best_opportunity.net_profit:.2f} ({best_opportunity.roi:.2f}% ROI)"
            )

        return best_opportunity

    def _choose_flash_loan_provider(self, amount: float) -> Optional[FlashLoanProvider]:
        """
        Choose best flash loan provider for amount.

        Prioritizes: 1) Can provide amount, 2) Lowest fees
        """
        eligible = []

        for provider, terms in self.FLASH_LOAN_TERMS.items():
            if amount <= terms.max_loan_amount:
                eligible.append((provider, terms))

        if not eligible:
            return None

        # Sort by fee (lowest first)
        eligible.sort(key=lambda x: x[1].fee_percent)
        return eligible[0][0]

    def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Simulate flash loan arbitrage execution.

        In production, this would:
        1. Borrow flash loan
        2. Buy on DEX A
        3. Sell on DEX B
        4. Repay flash loan + fee
        5. Keep profit

        Args:
            opportunity: Arbitrage opportunity

        Returns:
            True if successful
        """
        logger.info(f"🚀 Executing arbitrage: {opportunity.opportunity_id}")

        # Simulate execution (in production, would interact with smart contracts)
        steps = [
            f"1. Borrow ${opportunity.loan_amount:,.2f} from {opportunity.flash_loan_provider.value}",
            f"2. Buy {opportunity.amount:.6f} {opportunity.token} on {opportunity.buy_dex.value}",
            f"3. Sell {opportunity.amount:.6f} {opportunity.token} on {opportunity.sell_dex.value}",
            f"4. Repay loan: ${opportunity.loan_amount:,.2f} + fee: ${opportunity.flash_loan_fee:.2f}",
            f"5. Profit: ${opportunity.net_profit:.2f} ✅"
        ]

        for step in steps:
            logger.info(f"   {step}")

        # Track execution
        self.executed_arbitrages.append(opportunity)
        self.total_profit += opportunity.net_profit
        self.total_volume += opportunity.loan_amount
        self.success_rate = len(self.executed_arbitrages) / len(self.opportunities) * 100

        return True

    def scan_all_tokens(self, tokens: List[str], base_token: str = "USDC") -> List[ArbitrageOpportunity]:
        """
        Scan multiple tokens for arbitrage opportunities.

        Args:
            tokens: List of tokens to check
            base_token: Quote token

        Returns:
            List of profitable opportunities
        """
        opportunities = []

        logger.info(f"🔍 Scanning {len(tokens)} tokens for arbitrage...")

        for token in tokens:
            opp = self.detect_opportunity(token, base_token)
            if opp:
                opportunities.append(opp)

        logger.info(f"Found {len(opportunities)} profitable opportunities")

        return opportunities

    def generate_report(self, opportunity: ArbitrageOpportunity) -> str:
        """Generate detailed opportunity report."""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║        FLASH LOAN ARBITRAGE: {opportunity.opportunity_id}            ║
╠══════════════════════════════════════════════════════════════╣
║ Token: {opportunity.token:<20}  Time: {opportunity.timestamp.strftime('%H:%M:%S')}        ║
╠══════════════════════════════════════════════════════════════╣
║ PRICE ARBITRAGE                                               ║
╠══════════════════════════════════════════════════════════════╣
║ Buy DEX:  {opportunity.buy_dex.value.upper():<20}  Price: ${opportunity.buy_price:>10,.2f} ║
║ Sell DEX: {opportunity.sell_dex.value.upper():<20}  Price: ${opportunity.sell_price:>10,.2f} ║
║ Difference: ${opportunity.price_difference:>10,.2f} ({opportunity.price_difference_percent:>6.2f}%)            ║
╠══════════════════════════════════════════════════════════════╣
║ FLASH LOAN                                                    ║
╠══════════════════════════════════════════════════════════════╣
║ Provider: {opportunity.flash_loan_provider.value.upper():<20}                      ║
║ Loan Amount: ${opportunity.loan_amount:>12,.2f}                 ║
║ Loan Fee: ${opportunity.flash_loan_fee:>15,.2f}                 ║
╠══════════════════════════════════════════════════════════════╣
║ PROFITABILITY                                                 ║
╠══════════════════════════════════════════════════════════════╣
║ Buy Cost:         ${opportunity.buy_cost:>14,.2f}                 ║
║ Sell Proceeds:    ${opportunity.sell_proceeds:>14,.2f}                 ║
║ Gross Profit:     ${opportunity.gross_profit:>14,.2f}                 ║
║                                                               ║
║ Flash Loan Fee:   ${opportunity.flash_loan_fee:>14,.2f}                 ║
║ DEX Fees:         ${opportunity.dex_fees:>14,.2f}                 ║
║ Gas Cost:         ${opportunity.gas_cost:>14,.2f}                 ║
║ Total Costs:      ${opportunity.total_costs:>14,.2f}                 ║
║                                                               ║
║ NET PROFIT:       ${opportunity.net_profit:>14,.2f}                 ║
║ ROI:              {opportunity.roi:>15.2f}%                ║
╠══════════════════════════════════════════════════════════════╣
║ STATUS: {'✅ PROFITABLE' if opportunity.is_profitable else '❌ NOT PROFITABLE':<20}                            ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report

    def get_statistics(self) -> Dict:
        """Get arbitrage statistics."""
        return {
            'total_opportunities': len(self.opportunities),
            'profitable_opportunities': sum(1 for o in self.opportunities if o.is_profitable),
            'executed_arbitrages': len(self.executed_arbitrages),
            'total_profit': self.total_profit,
            'total_volume': self.total_volume,
            'success_rate': self.success_rate,
            'avg_profit': self.total_profit / len(self.executed_arbitrages) if self.executed_arbitrages else 0,
            'best_opportunity': max(
                self.opportunities,
                key=lambda x: x.net_profit
            ) if self.opportunities else None
        }


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("⚡ Flash Loan Arbitrage Demo")
    print("=" * 60)

    # Setup aggregator with pools
    print("\n1. Setting up DEX pools...")
    print("-" * 60)

    aggregator = DEXAggregator()

    # ETH pools with price discrepancy
    aggregator.add_pool(Pool(
        dex=DEX.UNISWAP_V2,
        token_a="USDC",
        token_b="ETH",
        reserve_a=2000000,
        reserve_b=1000,
        fee=0.003,
        liquidity_usd=4000000
    ))

    aggregator.add_pool(Pool(
        dex=DEX.SUSHISWAP,
        token_a="USDC",
        token_b="ETH",
        reserve_a=1050000,  # Different ratio = price discrepancy
        reserve_b=500,
        fee=0.003,
        liquidity_usd=2100000
    ))

    print("Added 2 pools with price discrepancy")

    # Initialize arbitrage system
    print("\n2. Detecting Arbitrage Opportunity...")
    print("-" * 60)

    arb = FlashLoanArbitrage(
        aggregator=aggregator,
        min_profit_usd=50.0,
        min_roi_percent=0.5
    )

    # Detect opportunity
    opportunity = arb.detect_opportunity("ETH", "USDC", loan_amounts=[10000, 50000])

    if opportunity:
        # Show report
        print(arb.generate_report(opportunity))

        # Execute
        print("\n3. Executing Flash Loan Arbitrage...")
        print("-" * 60)
        success = arb.execute_arbitrage(opportunity)

        if success:
            print(f"\n✅ Arbitrage executed successfully!")
            print(f"Net Profit: ${opportunity.net_profit:.2f}")

        # Statistics
        print("\n4. Statistics")
        print("-" * 60)
        stats = arb.get_statistics()
        print(f"Total Opportunities: {stats['total_opportunities']}")
        print(f"Profitable: {stats['profitable_opportunities']}")
        print(f"Executed: {stats['executed_arbitrages']}")
        print(f"Total Profit: ${stats['total_profit']:,.2f}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
    else:
        print("No profitable arbitrage opportunities found")

    print("\n✅ Flash loan arbitrage demo complete!")
