"""
MEV Detection System
Identifies MEV attacks including sandwich attacks, frontrunning, and backrunning.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class MEVType(Enum):
    """Types of MEV attacks."""
    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    LIQUIDATION = "liquidation"
    ARBITRAGE = "arbitrage"
    JIT_LIQUIDITY = "jit_liquidity"


class AttackSeverity(Enum):
    """Attack severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Transaction:
    """Blockchain transaction."""
    tx_hash: str
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_limit: int
    data: str
    timestamp: datetime
    block_number: Optional[int] = None

    # DEX specific
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    amount_in: Optional[float] = None
    amount_out: Optional[float] = None
    dex: Optional[str] = None


@dataclass
class MEVAttack:
    """Detected MEV attack."""
    attack_id: str
    attack_type: MEVType
    severity: AttackSeverity
    timestamp: datetime

    # Transactions involved
    attacker_address: str
    victim_tx: Transaction
    attack_txs: List[Transaction]

    # Financial impact
    victim_loss: float  # USD
    attacker_profit: float  # USD
    gas_cost: float  # USD
    net_profit: float  # USD

    # Details
    token_pair: str
    dex: str
    block_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class MEVDetector:
    """
    MEV Attack Detector

    Analyzes transaction patterns to identify MEV attacks.
    """

    def __init__(
        self,
        min_profit_threshold: float = 10.0,  # Min $10 profit to consider
        max_block_distance: int = 3  # Max blocks between related txs
    ):
        """
        Initialize MEV detector.

        Args:
            min_profit_threshold: Minimum profit to flag as attack (USD)
            max_block_distance: Maximum blocks between related transactions
        """
        self.min_profit_threshold = min_profit_threshold
        self.max_block_distance = max_block_distance

        # Attack tracking
        self.detected_attacks: List[MEVAttack] = []
        self.pending_txs: List[Transaction] = []
        self.attack_counter = 0

        # Statistics
        self.total_victim_loss = 0.0
        self.total_attacker_profit = 0.0
        self.attacks_by_type = defaultdict(int)

        logger.info(f"MEV detector initialized (min_profit=${min_profit_threshold})")

    def _generate_attack_id(self) -> str:
        """Generate unique attack ID."""
        self.attack_counter += 1
        return f"MEV-{self.attack_counter:06d}"

    def add_pending_transaction(self, tx: Transaction):
        """
        Add transaction to pending pool (mempool simulation).

        Args:
            tx: Transaction to analyze
        """
        self.pending_txs.append(tx)
        logger.debug(f"Added pending tx: {tx.tx_hash[:10]}... ({tx.from_address[:8]}...)")

    def detect_sandwich_attack(
        self,
        victim_tx: Transaction,
        potential_attackers: List[Transaction]
    ) -> Optional[MEVAttack]:
        """
        Detect sandwich attack pattern.

        Sandwich attack:
        1. Attacker sees victim's large buy in mempool
        2. Attacker frontruns with buy (pushes price up)
        3. Victim's tx executes at worse price
        4. Attacker backruns with sell (profits from price difference)

        Args:
            victim_tx: Potential victim transaction
            potential_attackers: Transactions from same address

        Returns:
            MEVAttack if detected, None otherwise
        """
        if not victim_tx.token_in or not victim_tx.token_out:
            return None

        # Group by address
        by_address = defaultdict(list)
        for tx in potential_attackers:
            by_address[tx.from_address].append(tx)

        # Look for sandwich pattern from each address
        for attacker_addr, txs in by_address.items():
            if len(txs) < 2:
                continue

            # Sort by gas price (higher gas = earlier execution)
            txs_sorted = sorted(txs, key=lambda x: x.gas_price, reverse=True)

            # Find frontrun + backrun pair
            for i, frontrun in enumerate(txs_sorted):
                # Frontrun must have higher gas than victim
                if frontrun.gas_price <= victim_tx.gas_price:
                    continue

                # Same token pair
                if (frontrun.token_in != victim_tx.token_in or
                    frontrun.token_out != victim_tx.token_out):
                    continue

                # Look for backrun (opposite direction)
                for backrun in txs_sorted[i+1:]:
                    # Backrun sells what frontrun bought
                    if (backrun.token_in == frontrun.token_out and
                        backrun.token_out == frontrun.token_in):

                        # Calculate impact
                        victim_loss = self._calculate_sandwich_victim_loss(
                            victim_tx, frontrun, backrun
                        )
                        attacker_profit = self._calculate_sandwich_profit(
                            frontrun, backrun
                        )

                        if attacker_profit < self.min_profit_threshold:
                            continue

                        # Determine severity
                        severity = self._calculate_severity(victim_loss)

                        # Create attack record
                        attack = MEVAttack(
                            attack_id=self._generate_attack_id(),
                            attack_type=MEVType.SANDWICH,
                            severity=severity,
                            timestamp=victim_tx.timestamp,
                            attacker_address=attacker_addr,
                            victim_tx=victim_tx,
                            attack_txs=[frontrun, backrun],
                            victim_loss=victim_loss,
                            attacker_profit=attacker_profit,
                            gas_cost=frontrun.gas_price + backrun.gas_price,
                            net_profit=attacker_profit - (frontrun.gas_price + backrun.gas_price),
                            token_pair=f"{victim_tx.token_in}/{victim_tx.token_out}",
                            dex=victim_tx.dex or "unknown",
                            metadata={
                                'frontrun_gas': frontrun.gas_price,
                                'victim_gas': victim_tx.gas_price,
                                'backrun_gas': backrun.gas_price
                            }
                        )

                        self.detected_attacks.append(attack)
                        self.total_victim_loss += victim_loss
                        self.total_attacker_profit += attacker_profit
                        self.attacks_by_type[MEVType.SANDWICH] += 1

                        logger.warning(
                            f"🚨 Sandwich attack detected! "
                            f"Victim loss: ${victim_loss:.2f}, "
                            f"Attacker profit: ${attacker_profit:.2f}"
                        )

                        return attack

        return None

    def detect_frontrunning(
        self,
        victim_tx: Transaction,
        frontrunner_txs: List[Transaction]
    ) -> Optional[MEVAttack]:
        """
        Detect frontrunning attack.

        Frontrunning:
        - Attacker sees profitable tx in mempool
        - Attacker submits same tx with higher gas
        - Attacker's tx executes first, taking the profit

        Args:
            victim_tx: Potential victim transaction
            frontrunner_txs: Transactions with higher gas price

        Returns:
            MEVAttack if detected
        """
        for frontrunner in frontrunner_txs:
            # Must have higher gas price
            if frontrunner.gas_price <= victim_tx.gas_price:
                continue

            # Similar transaction (same token pair, similar amount)
            if (frontrunner.token_in == victim_tx.token_in and
                frontrunner.token_out == victim_tx.token_out and
                abs(frontrunner.amount_in - victim_tx.amount_in) / victim_tx.amount_in < 0.1):

                # Estimate victim loss (missed opportunity)
                victim_loss = victim_tx.amount_out * 0.05  # Assume 5% opportunity loss
                attacker_profit = victim_loss  # Attacker gains what victim loses

                if attacker_profit < self.min_profit_threshold:
                    continue

                severity = self._calculate_severity(victim_loss)

                attack = MEVAttack(
                    attack_id=self._generate_attack_id(),
                    attack_type=MEVType.FRONTRUN,
                    severity=severity,
                    timestamp=victim_tx.timestamp,
                    attacker_address=frontrunner.from_address,
                    victim_tx=victim_tx,
                    attack_txs=[frontrunner],
                    victim_loss=victim_loss,
                    attacker_profit=attacker_profit,
                    gas_cost=frontrunner.gas_price,
                    net_profit=attacker_profit - frontrunner.gas_price,
                    token_pair=f"{victim_tx.token_in}/{victim_tx.token_out}",
                    dex=victim_tx.dex or "unknown"
                )

                self.detected_attacks.append(attack)
                self.total_victim_loss += victim_loss
                self.total_attacker_profit += attacker_profit
                self.attacks_by_type[MEVType.FRONTRUN] += 1

                logger.warning(f"🚨 Frontrunning detected! Profit: ${attacker_profit:.2f}")

                return attack

        return None

    def _calculate_sandwich_victim_loss(
        self,
        victim_tx: Transaction,
        frontrun: Transaction,
        backrun: Transaction
    ) -> float:
        """
        Calculate victim's loss from sandwich attack.

        Loss = (expected price - actual price) * amount
        """
        # Simplified: assume 1-3% slippage increase from frontrun
        # In reality, would calculate from AMM curve
        additional_slippage = 0.02  # 2% extra slippage
        victim_loss = victim_tx.amount_in * additional_slippage
        return victim_loss

    def _calculate_sandwich_profit(
        self,
        frontrun: Transaction,
        backrun: Transaction
    ) -> float:
        """Calculate attacker's profit from sandwich."""
        # Profit = sell proceeds - buy cost
        if frontrun.amount_out and backrun.amount_out:
            # Assuming amounts are in USD equivalent
            profit = backrun.amount_out - frontrun.amount_in
            return max(0, profit)
        return 0.0

    def _calculate_severity(self, victim_loss: float) -> AttackSeverity:
        """Calculate attack severity based on victim loss."""
        if victim_loss >= 1000:
            return AttackSeverity.CRITICAL
        elif victim_loss >= 500:
            return AttackSeverity.HIGH
        elif victim_loss >= 100:
            return AttackSeverity.MEDIUM
        else:
            return AttackSeverity.LOW

    def analyze_block(self, transactions: List[Transaction]) -> List[MEVAttack]:
        """
        Analyze a block of transactions for MEV attacks.

        Args:
            transactions: All transactions in the block

        Returns:
            List of detected attacks
        """
        attacks_found = []

        # Sort by gas price (execution order)
        txs_sorted = sorted(transactions, key=lambda x: x.gas_price, reverse=True)

        # Check each transaction as potential victim
        for i, victim_tx in enumerate(txs_sorted):
            if not victim_tx.token_in or not victim_tx.token_out:
                continue

            # Look for attacks involving this transaction
            potential_attackers = [
                tx for j, tx in enumerate(txs_sorted)
                if j != i and tx.from_address != victim_tx.from_address
            ]

            # Check for sandwich
            attack = self.detect_sandwich_attack(victim_tx, potential_attackers)
            if attack:
                attacks_found.append(attack)
                continue

            # Check for frontrunning
            frontrunners = [
                tx for tx in potential_attackers
                if tx.gas_price > victim_tx.gas_price
            ]
            attack = self.detect_frontrunning(victim_tx, frontrunners)
            if attack:
                attacks_found.append(attack)

        return attacks_found

    def get_protection_recommendation(self, tx: Transaction) -> Dict:
        """
        Get protection recommendations for a transaction.

        Args:
            tx: Transaction to protect

        Returns:
            Protection recommendations
        """
        recommendations = {
            'risk_level': 'low',
            'recommendations': [],
            'estimated_max_loss': 0.0
        }

        # Check transaction size (larger = more attractive to MEV)
        if tx.amount_in and tx.amount_in > 10000:  # $10k+
            recommendations['risk_level'] = 'high'
            recommendations['recommendations'].extend([
                "Use private mempool (Flashbots Protect)",
                "Split order into smaller chunks (TWAP)",
                "Set tight slippage tolerance (<0.5%)",
                "Use MEV-protected RPC endpoint"
            ])
            recommendations['estimated_max_loss'] = tx.amount_in * 0.02  # 2% max

        elif tx.amount_in and tx.amount_in > 1000:  # $1k+
            recommendations['risk_level'] = 'medium'
            recommendations['recommendations'].extend([
                "Set slippage tolerance to 0.5-1%",
                "Consider using private mempool",
                "Monitor transaction closely"
            ])
            recommendations['estimated_max_loss'] = tx.amount_in * 0.01  # 1% max

        else:
            recommendations['recommendations'].append(
                "Transaction size is small, MEV risk is minimal"
            )
            recommendations['estimated_max_loss'] = tx.amount_in * 0.005 if tx.amount_in else 0

        return recommendations

    def get_statistics(self) -> Dict:
        """Get MEV detection statistics."""
        return {
            'total_attacks': len(self.detected_attacks),
            'total_victim_loss': self.total_victim_loss,
            'total_attacker_profit': self.total_attacker_profit,
            'attacks_by_type': dict(self.attacks_by_type),
            'avg_profit_per_attack': (
                self.total_attacker_profit / len(self.detected_attacks)
                if self.detected_attacks else 0
            ),
            'most_common_attack': max(
                self.attacks_by_type.items(),
                key=lambda x: x[1]
            )[0].value if self.attacks_by_type else None
        }

    def get_summary(self) -> str:
        """Get formatted statistics summary."""
        stats = self.get_statistics()

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                  MEV DETECTION STATISTICS                     ║
╠══════════════════════════════════════════════════════════════╣
║ Total Attacks:        {stats['total_attacks']:>15}                    ║
║ Total Victim Loss:    ${stats['total_victim_loss']:>12,.2f}                    ║
║ Total Attacker Profit:${stats['total_attacker_profit']:>12,.2f}                    ║
║ Avg Profit/Attack:    ${stats['avg_profit_per_attack']:>12,.2f}                    ║
╚══════════════════════════════════════════════════════════════╝

Attack Types:
"""
        for attack_type, count in stats['attacks_by_type'].items():
            summary += f"  {attack_type.value}: {count}\n"

        return summary


if __name__ == '__main__':
    import logging
    from datetime import datetime, timedelta

    logging.basicConfig(level=logging.INFO)

    print("🔍 MEV Detection System Demo")
    print("=" * 60)

    # Initialize detector
    detector = MEVDetector(min_profit_threshold=10.0)

    # Simulate transactions
    print("\n1. Simulating sandwich attack scenario...")
    print("-" * 60)

    # Victim transaction (large swap)
    victim = Transaction(
        tx_hash="0xvictim123",
        from_address="0xuser456",
        to_address="0xuniswap",
        value=5000.0,
        gas_price=50.0,  # 50 Gwei
        gas_limit=200000,
        data="swap",
        timestamp=datetime.now(),
        token_in="USDC",
        token_out="ETH",
        amount_in=5000.0,
        amount_out=2.38,
        dex="Uniswap"
    )

    # Attacker frontrun (higher gas)
    frontrun = Transaction(
        tx_hash="0xattack_front",
        from_address="0xattacker789",
        to_address="0xuniswap",
        value=3000.0,
        gas_price=100.0,  # Higher gas - executes first
        gas_limit=200000,
        data="swap",
        timestamp=datetime.now(),
        token_in="USDC",
        token_out="ETH",
        amount_in=3000.0,
        amount_out=1.43,
        dex="Uniswap"
    )

    # Attacker backrun (lower gas, executes after victim)
    backrun = Transaction(
        tx_hash="0xattack_back",
        from_address="0xattacker789",
        to_address="0xuniswap",
        value=1.43,
        gas_price=45.0,  # Lower gas - executes after victim
        gas_limit=200000,
        data="swap",
        timestamp=datetime.now() + timedelta(seconds=1),
        token_in="ETH",
        token_out="USDC",
        amount_in=1.43,
        amount_out=3100.0,  # Profit!
        dex="Uniswap"
    )

    # Detect sandwich attack
    attack = detector.detect_sandwich_attack(victim, [frontrun, backrun])

    if attack:
        print(f"\n✅ Sandwich attack detected!")
        print(f"  Attack ID: {attack.attack_id}")
        print(f"  Attacker: {attack.attacker_address[:15]}...")
        print(f"  Victim Loss: ${attack.victim_loss:.2f}")
        print(f"  Attacker Profit: ${attack.attacker_profit:.2f}")
        print(f"  Net Profit: ${attack.net_profit:.2f}")
        print(f"  Severity: {attack.severity.value}")

    # Get protection recommendations
    print("\n2. Protection Recommendations")
    print("-" * 60)

    recommendations = detector.get_protection_recommendation(victim)
    print(f"Risk Level: {recommendations['risk_level'].upper()}")
    print(f"Estimated Max Loss: ${recommendations['estimated_max_loss']:.2f}")
    print("\nRecommendations:")
    for rec in recommendations['recommendations']:
        print(f"  • {rec}")

    # Statistics
    print("\n3. Detection Statistics")
    print("-" * 60)
    print(detector.get_summary())

    print("✅ MEV detection demo complete!")
