"""
Sandwich Attack Detector and Analyzer
Specialized tool for detecting and analyzing sandwich attacks with detailed metrics.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from .detector import Transaction, MEVAttack, MEVType, AttackSeverity

logger = logging.getLogger(__name__)


@dataclass
class SandwichAnalysis:
    """Detailed sandwich attack analysis."""
    attack_id: str
    timestamp: datetime

    # Parties
    attacker_address: str
    victim_address: str

    # Transaction details
    frontrun_tx: Transaction
    victim_tx: Transaction
    backrun_tx: Transaction

    # Price impact
    price_before: float
    price_after_frontrun: float
    price_after_victim: float
    price_after_backrun: float

    # Financial metrics
    victim_expected_output: float
    victim_actual_output: float
    victim_loss_usd: float
    victim_loss_percent: float

    attacker_buy_cost: float
    attacker_sell_proceeds: float
    attacker_gross_profit: float
    attacker_gas_cost: float
    attacker_net_profit: float
    attacker_roi: float

    # Gas competition
    frontrun_gas_premium: float  # % above victim
    victim_gas_price: float
    backrun_gas_price: float

    # Block/timing info
    block_number: Optional[int] = None
    position_in_block: Optional[Tuple[int, int, int]] = None  # (frontrun, victim, backrun)

    # DEX info
    dex: str = "unknown"
    token_pair: str = ""
    pool_liquidity: Optional[float] = None


class SandwichDetector:
    """
    Advanced Sandwich Attack Detector

    Provides detailed analysis of sandwich attacks including:
    - Price impact visualization
    - Profitability metrics
    - Gas competition analysis
    - Protection strategies
    """

    def __init__(self, min_victim_loss: float = 10.0):
        """
        Initialize sandwich detector.

        Args:
            min_victim_loss: Minimum victim loss to flag (USD)
        """
        self.min_victim_loss = min_victim_loss

        # Detection tracking
        self.detected_sandwiches: List[SandwichAnalysis] = []
        self.attack_counter = 0

        # Statistics
        self.total_attacks = 0
        self.total_victim_loss = 0.0
        self.total_attacker_profit = 0.0
        self.avg_gas_premium = 0.0

        logger.info(f"Sandwich detector initialized (min_loss=${min_victim_loss})")

    def analyze_sandwich(
        self,
        frontrun: Transaction,
        victim: Transaction,
        backrun: Transaction,
        pool_price_before: float = 2000.0  # ETH price as example
    ) -> Optional[SandwichAnalysis]:
        """
        Perform detailed analysis of potential sandwich attack.

        Args:
            frontrun: Frontrun transaction (attacker buys)
            victim: Victim transaction
            backrun: Backrun transaction (attacker sells)
            pool_price_before: Pool price before attack

        Returns:
            SandwichAnalysis if attack detected
        """
        # Verify sandwich pattern
        if not self._is_sandwich_pattern(frontrun, victim, backrun):
            return None

        # Calculate price impacts
        prices = self._calculate_price_impacts(
            frontrun, victim, backrun, pool_price_before
        )

        # Calculate victim loss
        victim_expected = victim.amount_in / pool_price_before
        victim_actual = victim.amount_out or victim_expected * 0.98  # Assume 2% loss if not specified
        victim_loss_amount = victim_expected - victim_actual
        victim_loss_usd = victim_loss_amount * pool_price_before
        victim_loss_pct = (victim_loss_amount / victim_expected) * 100

        if victim_loss_usd < self.min_victim_loss:
            return None

        # Calculate attacker profit
        attacker_buy_cost = frontrun.amount_in
        attacker_sell_proceeds = backrun.amount_out or (frontrun.amount_out * prices['after_backrun'])
        attacker_gross = attacker_sell_proceeds - attacker_buy_cost
        attacker_gas = frontrun.gas_price + backrun.gas_price
        attacker_net = attacker_gross - attacker_gas
        attacker_roi = (attacker_net / attacker_buy_cost) * 100 if attacker_buy_cost > 0 else 0

        # Gas competition metrics
        gas_premium = ((frontrun.gas_price - victim.gas_price) / victim.gas_price) * 100

        # Create analysis
        self.attack_counter += 1
        analysis = SandwichAnalysis(
            attack_id=f"SAND-{self.attack_counter:06d}",
            timestamp=victim.timestamp,
            attacker_address=frontrun.from_address,
            victim_address=victim.from_address,
            frontrun_tx=frontrun,
            victim_tx=victim,
            backrun_tx=backrun,
            price_before=prices['before'],
            price_after_frontrun=prices['after_frontrun'],
            price_after_victim=prices['after_victim'],
            price_after_backrun=prices['after_backrun'],
            victim_expected_output=victim_expected,
            victim_actual_output=victim_actual,
            victim_loss_usd=victim_loss_usd,
            victim_loss_percent=victim_loss_pct,
            attacker_buy_cost=attacker_buy_cost,
            attacker_sell_proceeds=attacker_sell_proceeds,
            attacker_gross_profit=attacker_gross,
            attacker_gas_cost=attacker_gas,
            attacker_net_profit=attacker_net,
            attacker_roi=attacker_roi,
            frontrun_gas_premium=gas_premium,
            victim_gas_price=victim.gas_price,
            backrun_gas_price=backrun.gas_price,
            block_number=victim.block_number,
            dex=victim.dex or "unknown",
            token_pair=f"{victim.token_in}/{victim.token_out}"
        )

        self.detected_sandwiches.append(analysis)
        self.total_attacks += 1
        self.total_victim_loss += victim_loss_usd
        self.total_attacker_profit += attacker_net
        self.avg_gas_premium = (self.avg_gas_premium * (self.total_attacks - 1) + gas_premium) / self.total_attacks

        logger.warning(
            f"🥪 Sandwich attack {analysis.attack_id}: "
            f"Victim lost ${victim_loss_usd:.2f} ({victim_loss_pct:.2f}%), "
            f"Attacker profited ${attacker_net:.2f} (ROI: {attacker_roi:.2f}%)"
        )

        return analysis

    def _is_sandwich_pattern(
        self,
        frontrun: Transaction,
        victim: Transaction,
        backrun: Transaction
    ) -> bool:
        """Verify transactions match sandwich pattern."""
        # Same attacker for frontrun and backrun
        if frontrun.from_address != backrun.from_address:
            return False

        # Frontrun and victim in same direction (both buy or both sell)
        if (frontrun.token_in != victim.token_in or
            frontrun.token_out != victim.token_out):
            return False

        # Backrun in opposite direction
        if (backrun.token_in != frontrun.token_out or
            backrun.token_out != frontrun.token_in):
            return False

        # Frontrun has higher gas (executes first)
        if frontrun.gas_price <= victim.gas_price:
            return False

        # Backrun has lower gas (executes last)
        if backrun.gas_price >= victim.gas_price:
            return False

        return True

    def _calculate_price_impacts(
        self,
        frontrun: Transaction,
        victim: Transaction,
        backrun: Transaction,
        initial_price: float
    ) -> Dict[str, float]:
        """
        Calculate price at each stage of sandwich.

        Simplified constant product AMM model: x * y = k
        Price impact = amount / (liquidity + amount)
        """
        # Assume pool liquidity (would query on-chain in production)
        pool_liquidity_usd = 1000000.0  # $1M liquidity

        # Price before
        price_before = initial_price

        # After frontrun (price goes up)
        frontrun_impact = frontrun.amount_in / pool_liquidity_usd
        price_after_frontrun = price_before * (1 + frontrun_impact)

        # After victim (price goes up more)
        victim_impact = victim.amount_in / pool_liquidity_usd
        price_after_victim = price_after_frontrun * (1 + victim_impact)

        # After backrun (price goes back down)
        backrun_impact = backrun.amount_in / pool_liquidity_usd
        price_after_backrun = price_after_victim * (1 - backrun_impact)

        return {
            'before': price_before,
            'after_frontrun': price_after_frontrun,
            'after_victim': price_after_victim,
            'after_backrun': price_after_backrun
        }

    def get_protection_strategy(self, analysis: SandwichAnalysis) -> Dict:
        """
        Generate protection strategy based on attack analysis.

        Args:
            analysis: Sandwich attack analysis

        Returns:
            Protection recommendations
        """
        strategies = []

        # Gas premium analysis
        if analysis.frontrun_gas_premium > 20:
            strategies.append({
                'method': 'Private Mempool (Flashbots Protect)',
                'effectiveness': 'high',
                'description': 'Submit transaction directly to block builders, bypassing public mempool',
                'cost': 'free'
            })

        # Transaction size analysis
        if analysis.victim_tx.amount_in > 10000:
            strategies.append({
                'method': 'TWAP (Time-Weighted Average Price)',
                'effectiveness': 'high',
                'description': f'Split ${analysis.victim_tx.amount_in:.0f} order into 10-20 smaller orders over time',
                'cost': 'gas for multiple transactions'
            })

        # Slippage analysis
        if analysis.victim_loss_percent > 1.0:
            strategies.append({
                'method': 'Tight Slippage Tolerance',
                'effectiveness': 'medium',
                'description': f'Set slippage to 0.5% (attack caused {analysis.victim_loss_percent:.2f}% loss)',
                'cost': 'free, may cause reverts'
            })

        # DEX selection
        strategies.append({
            'method': 'Use MEV-Protected DEX',
            'effectiveness': 'medium',
            'description': 'Use CoW Swap, 1inch Fusion, or other MEV-protected aggregators',
            'cost': 'similar or better pricing'
        })

        # Limit orders
        strategies.append({
            'method': 'Limit Orders',
            'effectiveness': 'high',
            'description': 'Use limit orders instead of market orders to guarantee price',
            'cost': 'may not fill immediately'
        })

        return {
            'attack_severity': 'high' if analysis.victim_loss_percent > 2 else 'medium',
            'victim_loss': analysis.victim_loss_usd,
            'strategies': strategies,
            'estimated_savings': analysis.victim_loss_usd * 0.8  # Could save 80% with protection
        }

    def generate_report(self, analysis: SandwichAnalysis) -> str:
        """Generate detailed attack report."""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           SANDWICH ATTACK ANALYSIS: {analysis.attack_id}         ║
╠══════════════════════════════════════════════════════════════╣
║ Timestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}                        ║
║ DEX: {analysis.dex:<20}   Pair: {analysis.token_pair:<15} ║
╠══════════════════════════════════════════════════════════════╣
║ VICTIM ANALYSIS                                               ║
╠══════════════════════════════════════════════════════════════╣
║ Address:          {analysis.victim_address[:30]}...          ║
║ Transaction:      {analysis.victim_tx.tx_hash[:30]}...          ║
║ Expected Output:  {analysis.victim_expected_output:>15.6f} {analysis.token_pair.split('/')[1]:<10} ║
║ Actual Output:    {analysis.victim_actual_output:>15.6f} {analysis.token_pair.split('/')[1]:<10} ║
║ Loss (USD):       ${analysis.victim_loss_usd:>14,.2f}                 ║
║ Loss (%):         {analysis.victim_loss_percent:>15.2f}%                ║
╠══════════════════════════════════════════════════════════════╣
║ ATTACKER ANALYSIS                                             ║
╠══════════════════════════════════════════════════════════════╣
║ Address:          {analysis.attacker_address[:30]}...          ║
║ Buy Cost:         ${analysis.attacker_buy_cost:>14,.2f}                 ║
║ Sell Proceeds:    ${analysis.attacker_sell_proceeds:>14,.2f}                 ║
║ Gross Profit:     ${analysis.attacker_gross_profit:>14,.2f}                 ║
║ Gas Cost:         ${analysis.attacker_gas_cost:>14,.2f}                 ║
║ Net Profit:       ${analysis.attacker_net_profit:>14,.2f}                 ║
║ ROI:              {analysis.attacker_roi:>15.2f}%                ║
╠══════════════════════════════════════════════════════════════╣
║ PRICE IMPACT                                                  ║
╠══════════════════════════════════════════════════════════════╣
║ Before Attack:    ${analysis.price_before:>14,.2f}                 ║
║ After Frontrun:   ${analysis.price_after_frontrun:>14,.2f} ({((analysis.price_after_frontrun/analysis.price_before-1)*100):>+6.2f}%)    ║
║ After Victim:     ${analysis.price_after_victim:>14,.2f} ({((analysis.price_after_victim/analysis.price_before-1)*100):>+6.2f}%)    ║
║ After Backrun:    ${analysis.price_after_backrun:>14,.2f} ({((analysis.price_after_backrun/analysis.price_before-1)*100):>+6.2f}%)    ║
╠══════════════════════════════════════════════════════════════╣
║ GAS COMPETITION                                               ║
╠══════════════════════════════════════════════════════════════╣
║ Frontrun Gas:     {analysis.frontrun_tx.gas_price:>15.2f} Gwei             ║
║ Victim Gas:       {analysis.victim_gas_price:>15.2f} Gwei             ║
║ Backrun Gas:      {analysis.backrun_gas_price:>15.2f} Gwei             ║
║ Gas Premium:      {analysis.frontrun_gas_premium:>15.2f}% above victim   ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report

    def get_statistics(self) -> Dict:
        """Get sandwich attack statistics."""
        return {
            'total_sandwiches': self.total_attacks,
            'total_victim_loss': self.total_victim_loss,
            'total_attacker_profit': self.total_attacker_profit,
            'avg_victim_loss': self.total_victim_loss / self.total_attacks if self.total_attacks > 0 else 0,
            'avg_attacker_profit': self.total_attacker_profit / self.total_attacks if self.total_attacks > 0 else 0,
            'avg_gas_premium': self.avg_gas_premium,
            'most_targeted_dex': self._get_most_targeted_dex()
        }

    def _get_most_targeted_dex(self) -> str:
        """Find most targeted DEX."""
        if not self.detected_sandwiches:
            return "none"

        dex_counts = defaultdict(int)
        for analysis in self.detected_sandwiches:
            dex_counts[analysis.dex] += 1

        return max(dex_counts.items(), key=lambda x: x[1])[0] if dex_counts else "unknown"


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🥪 Sandwich Attack Detector Demo")
    print("=" * 60)

    # Initialize detector
    detector = SandwichDetector(min_victim_loss=10.0)

    # Simulate sandwich attack
    print("\n1. Simulating Sandwich Attack...")
    print("-" * 60)

    # Frontrun (attacker buys, high gas)
    frontrun = Transaction(
        tx_hash="0xfront123abc",
        from_address="0xattacker_mev_bot",
        to_address="0xuniswap_router",
        value=10000.0,
        gas_price=150.0,  # High gas
        gas_limit=200000,
        data="swap",
        timestamp=datetime.now(),
        token_in="USDC",
        token_out="ETH",
        amount_in=10000.0,
        amount_out=4.76,
        dex="Uniswap V2"
    )

    # Victim (buys, normal gas)
    victim = Transaction(
        tx_hash="0xvictim456def",
        from_address="0xregular_user",
        to_address="0xuniswap_router",
        value=50000.0,
        gas_price=100.0,  # Normal gas
        gas_limit=200000,
        data="swap",
        timestamp=datetime.now(),
        token_in="USDC",
        token_out="ETH",
        amount_in=50000.0,
        amount_out=23.5,  # Gets less than expected
        dex="Uniswap V2"
    )

    # Backrun (attacker sells, low gas)
    backrun = Transaction(
        tx_hash="0xback789ghi",
        from_address="0xattacker_mev_bot",
        to_address="0xuniswap_router",
        value=4.76,
        gas_price=80.0,  # Low gas
        gas_limit=200000,
        data="swap",
        timestamp=datetime.now() + timedelta(seconds=2),
        token_in="ETH",
        token_out="USDC",
        amount_in=4.76,
        amount_out=10500.0,  # Profit!
        dex="Uniswap V2"
    )

    # Analyze sandwich
    analysis = detector.analyze_sandwich(frontrun, victim, backrun, pool_price_before=2100.0)

    if analysis:
        # Show report
        print(detector.generate_report(analysis))

        # Get protection strategies
        print("\n2. Protection Strategies")
        print("-" * 60)

        protection = detector.get_protection_strategy(analysis)
        print(f"Severity: {protection['attack_severity'].upper()}")
        print(f"Potential Savings: ${protection['estimated_savings']:.2f}\n")

        for i, strategy in enumerate(protection['strategies'], 1):
            print(f"{i}. {strategy['method']}")
            print(f"   Effectiveness: {strategy['effectiveness']}")
            print(f"   {strategy['description']}")
            print(f"   Cost: {strategy['cost']}\n")

        # Statistics
        print("3. Detection Statistics")
        print("-" * 60)
        stats = detector.get_statistics()
        print(f"Total Sandwiches: {stats['total_sandwiches']}")
        print(f"Total Victim Loss: ${stats['total_victim_loss']:,.2f}")
        print(f"Avg Loss per Attack: ${stats['avg_victim_loss']:,.2f}")
        print(f"Avg Gas Premium: {stats['avg_gas_premium']:.1f}%")

    print("\n✅ Sandwich detector demo complete!")
