"""
Whale Following Strategy
Track and follow large wallet movements (smart money) for profitable trades.

Strategy:
- Monitor whale wallets (>$1M holdings)
- Detect accumulation/distribution patterns
- Follow whale trades with confidence scoring
- Exit based on whale behavior or targets
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class WhaleWallet:
    """Represents a whale wallet being tracked."""
    address: str
    label: Optional[str]  # e.g., "Jump Trading", "3AC"
    total_value_usd: float
    success_rate: float  # Historical success rate (0-1)
    avg_hold_time_hours: float
    tokens_held: Dict[str, float]  # token -> balance


@dataclass
class WhaleTransaction:
    """Represents a whale transaction."""
    tx_hash: str
    timestamp: datetime
    whale_address: str
    token: str
    action: str  # 'buy', 'sell', 'transfer_in', 'transfer_out'
    amount: float
    usd_value: float
    exchange: Optional[str]  # If traded on exchange


@dataclass
class WhaleSignal:
    """Trading signal based on whale activity."""
    token: str
    action: str  # 'buy' or 'sell'
    confidence: float  # 0-1
    num_whales: int  # Number of whales supporting this signal
    total_volume_usd: float
    avg_whale_success: float
    entry_price: float
    suggested_hold_time_hours: float
    reasoning: str


class WhaleFollower:
    """
    Whale following strategy for cryptocurrency trading.

    Tracks large holders ("whales") and generates signals based on their
    accumulation/distribution patterns.

    Key Metrics:
    - Whale accumulation score
    - Exchange flow analysis
    - Historical whale success rates
    - Volume and timing patterns
    """

    # Whale classification thresholds
    WHALE_THRESHOLD_USD = 1_000_000  # $1M minimum
    MEGA_WHALE_THRESHOLD_USD = 100_000_000  # $100M+

    def __init__(
        self,
        min_whale_count: int = 3,
        min_confidence: float = 0.65,
        lookback_hours: int = 24
    ):
        """
        Initialize whale follower.

        Args:
            min_whale_count: Minimum number of whales for signal
            min_confidence: Minimum confidence score
            lookback_hours: Hours to look back for patterns
        """
        self.min_whale_count = min_whale_count
        self.min_confidence = min_confidence
        self.lookback_hours = lookback_hours

        # Tracked whales
        self.whale_wallets: Dict[str, WhaleWallet] = {}
        self.whale_transactions: List[WhaleTransaction] = []

        # Historical performance
        self.signal_history: List[Dict] = []

        logger.info(f"Whale follower initialized (min whales: {min_whale_count})")

    def add_whale_wallet(self, wallet: WhaleWallet):
        """Add a whale wallet to tracking list."""
        self.whale_wallets[wallet.address] = wallet
        logger.info(f"Added whale: {wallet.label or wallet.address[:10]}... (${wallet.total_value_usd:,.0f})")

    def record_transaction(self, tx: WhaleTransaction):
        """Record a whale transaction."""
        self.whale_transactions.append(tx)

        # Keep only recent transactions
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours * 2)
        self.whale_transactions = [
            t for t in self.whale_transactions
            if t.timestamp > cutoff_time
        ]

    def analyze_accumulation(
        self,
        token: str,
        current_price: float
    ) -> Dict:
        """
        Analyze whale accumulation patterns for a token.

        Args:
            token: Token symbol (e.g., 'ETH', 'BTC')
            current_price: Current token price

        Returns:
            Accumulation analysis
        """
        # Filter recent transactions for this token
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)
        recent_txs = [
            tx for tx in self.whale_transactions
            if tx.token == token and tx.timestamp > cutoff_time
        ]

        if not recent_txs:
            return {'accumulation_score': 0, 'whales_accumulating': 0}

        # Calculate net flow
        buys = [tx for tx in recent_txs if tx.action in ['buy', 'transfer_in']]
        sells = [tx for tx in recent_txs if tx.action in ['sell', 'transfer_out']]

        total_buy_volume = sum(tx.usd_value for tx in buys)
        total_sell_volume = sum(tx.usd_value for tx in sells)

        net_flow = total_buy_volume - total_sell_volume
        total_volume = total_buy_volume + total_sell_volume

        # Calculate accumulation score (-1 to 1)
        accumulation_score = net_flow / total_volume if total_volume > 0 else 0

        # Count unique whales accumulating
        whale_addresses_buying = set(tx.whale_address for tx in buys)
        whale_addresses_selling = set(tx.whale_address for tx in sells)

        net_whales_accumulating = len(whale_addresses_buying) - len(whale_addresses_selling)

        # Analyze by whale size (mega whales have more weight)
        mega_whale_buys = sum(
            tx.usd_value for tx in buys
            if self.whale_wallets.get(tx.whale_address) and
            self.whale_wallets[tx.whale_address].total_value_usd >= self.MEGA_WHALE_THRESHOLD_USD
        )

        regular_whale_buys = total_buy_volume - mega_whale_buys

        return {
            'accumulation_score': accumulation_score,  # -1 to 1
            'whales_accumulating': net_whales_accumulating,
            'total_buy_volume': total_buy_volume,
            'total_sell_volume': total_sell_volume,
            'net_flow': net_flow,
            'mega_whale_interest': mega_whale_buys / total_volume if total_volume > 0 else 0,
            'num_transactions': len(recent_txs),
            'avg_transaction_size': total_volume / len(recent_txs) if recent_txs else 0
        }

    def detect_smart_money_moves(
        self,
        token: str,
        current_price: float,
        min_transaction_size: float = 100_000
    ) -> List[WhaleTransaction]:
        """
        Detect significant whale transactions.

        Args:
            token: Token to analyze
            current_price: Current price
            min_transaction_size: Minimum transaction size in USD

        Returns:
            List of significant whale transactions
        """
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)

        significant_txs = [
            tx for tx in self.whale_transactions
            if (
                tx.token == token and
                tx.timestamp > cutoff_time and
                tx.usd_value >= min_transaction_size
            )
        ]

        # Sort by size
        significant_txs.sort(key=lambda tx: tx.usd_value, reverse=True)

        return significant_txs

    def generate_signal(
        self,
        token: str,
        current_price: float
    ) -> Optional[WhaleSignal]:
        """
        Generate trading signal based on whale analysis.

        Args:
            token: Token symbol
            current_price: Current market price

        Returns:
            Whale signal if conditions met
        """
        # Analyze accumulation
        analysis = self.analyze_accumulation(token, current_price)

        if analysis['whales_accumulating'] < self.min_whale_count:
            return None

        # Calculate confidence based on multiple factors
        confidence_factors = []

        # 1. Accumulation score
        acc_score = abs(analysis['accumulation_score'])
        confidence_factors.append(acc_score)

        # 2. Number of whales
        whale_factor = min(1.0, analysis['whales_accumulating'] / (self.min_whale_count * 2))
        confidence_factors.append(whale_factor)

        # 3. Mega whale interest
        confidence_factors.append(analysis['mega_whale_interest'])

        # 4. Volume significance
        if analysis['total_buy_volume'] > 0:
            volume_factor = min(1.0, analysis['total_buy_volume'] / 10_000_000)  # $10M baseline
            confidence_factors.append(volume_factor)

        # Average confidence
        confidence = sum(confidence_factors) / len(confidence_factors)

        if confidence < self.min_confidence:
            return None

        # Determine action
        if analysis['accumulation_score'] > 0:
            action = 'buy'
            reasoning = f"{analysis['whales_accumulating']} whales accumulating ${analysis['total_buy_volume']:,.0f}"
        else:
            action = 'sell'
            reasoning = f"{abs(analysis['whales_accumulating'])} whales distributing ${analysis['total_sell_volume']:,.0f}"

        # Estimate hold time based on whale history
        relevant_whales = [
            whale for whale in self.whale_wallets.values()
            if token in whale.tokens_held
        ]

        avg_hold_time = sum(w.avg_hold_time_hours for w in relevant_whales) / len(relevant_whales) if relevant_whales else 72

        # Calculate average whale success rate
        avg_success = sum(w.success_rate for w in relevant_whales) / len(relevant_whales) if relevant_whales else 0.5

        signal = WhaleSignal(
            token=token,
            action=action,
            confidence=confidence,
            num_whales=analysis['whales_accumulating'],
            total_volume_usd=analysis['total_buy_volume'] if action == 'buy' else analysis['total_sell_volume'],
            avg_whale_success=avg_success,
            entry_price=current_price,
            suggested_hold_time_hours=avg_hold_time,
            reasoning=reasoning
        )

        logger.info(f"Generated whale signal: {action.upper()} {token} @ ${current_price:.2f}")
        logger.info(f"Confidence: {confidence:.0%}, Whales: {analysis['whales_accumulating']}")

        return signal

    def get_whale_portfolio_composition(
        self,
        whale_address: str
    ) -> Dict[str, float]:
        """
        Get portfolio composition for a whale.

        Args:
            whale_address: Whale wallet address

        Returns:
            Portfolio composition (token -> percentage)
        """
        whale = self.whale_wallets.get(whale_address)
        if not whale:
            return {}

        total_value = whale.total_value_usd
        composition = {}

        for token, balance in whale.tokens_held.items():
            # This would require token price lookup in production
            composition[token] = (balance / total_value) * 100

        return composition

    def track_whale_performance(
        self,
        whale_address: str,
        token: str,
        days: int = 30
    ) -> Dict:
        """
        Track historical performance of a whale for a specific token.

        Args:
            whale_address: Whale address
            token: Token to analyze
            days: Days to look back

        Returns:
            Performance metrics
        """
        cutoff_time = datetime.now() - timedelta(days=days)

        # Get whale's transactions for this token
        whale_txs = [
            tx for tx in self.whale_transactions
            if (
                tx.whale_address == whale_address and
                tx.token == token and
                tx.timestamp > cutoff_time
            )
        ]

        if not whale_txs:
            return {'trades': 0}

        # Calculate metrics
        buys = [tx for tx in whale_txs if tx.action == 'buy']
        sells = [tx for tx in whale_txs if tx.action == 'sell']

        total_bought = sum(tx.amount for tx in buys)
        total_sold = sum(tx.amount for tx in sells)

        avg_buy_price = sum(tx.usd_value / tx.amount for tx in buys) / len(buys) if buys else 0
        avg_sell_price = sum(tx.usd_value / tx.amount for tx in sells) / len(sells) if sells else 0

        # Calculate P&L if both buys and sells
        pnl_percent = ((avg_sell_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price > 0 else 0

        return {
            'trades': len(whale_txs),
            'total_bought': total_bought,
            'total_sold': total_sold,
            'avg_buy_price': avg_buy_price,
            'avg_sell_price': avg_sell_price,
            'pnl_percent': pnl_percent,
            'is_profitable': pnl_percent > 0
        }

    def get_top_whale_tokens(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get tokens most held by whales.

        Args:
            top_n: Number of top tokens to return

        Returns:
            List of (token, whale_count) tuples
        """
        token_whale_count: Dict[str, int] = {}

        for whale in self.whale_wallets.values():
            for token in whale.tokens_held.keys():
                token_whale_count[token] = token_whale_count.get(token, 0) + 1

        # Sort by whale count
        sorted_tokens = sorted(token_whale_count.items(), key=lambda x: x[1], reverse=True)

        return sorted_tokens[:top_n]


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🐋 Whale Follower Demo")
    print("=" * 60)

    # Initialize follower
    follower = WhaleFollower(
        min_whale_count=3,
        min_confidence=0.65,
        lookback_hours=24
    )

    # Add some whale wallets
    print("\n👥 Adding whale wallets...")

    whales = [
        WhaleWallet(
            address="0x1234...5678",
            label="Jump Trading",
            total_value_usd=500_000_000,
            success_rate=0.75,
            avg_hold_time_hours=120,
            tokens_held={'ETH': 50000, 'BTC': 1000}
        ),
        WhaleWallet(
            address="0xabcd...ef01",
            label="3AC Wallet",
            total_value_usd=200_000_000,
            success_rate=0.68,
            avg_hold_time_hours=72,
            tokens_held={'ETH': 30000, 'MATIC': 10_000_000}
        ),
        WhaleWallet(
            address="0x9876...5432",
            label="DWF Labs",
            total_value_usd=150_000_000,
            success_rate=0.82,
            avg_hold_time_hours=48,
            tokens_held={'ETH': 25000, 'LINK': 500_000}
        ),
        WhaleWallet(
            address="0xfed...cba",
            label="Alameda Research",
            total_value_usd=1_000_000_000,
            success_rate=0.71,
            avg_hold_time_hours=96,
            tokens_held={'ETH': 100000, 'SOL': 5_000_000}
        ),
    ]

    for whale in whales:
        follower.add_whale_wallet(whale)

    # Simulate recent transactions
    print("\n📊 Recording whale transactions...")

    transactions = [
        WhaleTransaction(
            tx_hash="0xabc123",
            timestamp=datetime.now() - timedelta(hours=2),
            whale_address="0x1234...5678",
            token="ETH",
            action="buy",
            amount=500,
            usd_value=1_000_000,
            exchange="Binance"
        ),
        WhaleTransaction(
            tx_hash="0xdef456",
            timestamp=datetime.now() - timedelta(hours=4),
            whale_address="0xabcd...ef01",
            token="ETH",
            action="buy",
            amount=300,
            usd_value=600_000,
            exchange="Coinbase"
        ),
        WhaleTransaction(
            tx_hash="0xghi789",
            timestamp=datetime.now() - timedelta(hours=6),
            whale_address="0x9876...5432",
            token="ETH",
            action="buy",
            amount=400,
            usd_value=800_000,
            exchange="Kraken"
        ),
        WhaleTransaction(
            tx_hash="0xjkl012",
            timestamp=datetime.now() - timedelta(hours=1),
            whale_address="0xfed...cba",
            token="ETH",
            action="buy",
            amount=1000,
            usd_value=2_000_000,
            exchange="Binance"
        ),
    ]

    for tx in transactions:
        follower.record_transaction(tx)

    # Analyze accumulation
    print("\n🔍 Analyzing whale accumulation for ETH...")
    analysis = follower.analyze_accumulation("ETH", 2000.0)

    print(f"\n📈 Accumulation Analysis:")
    print(f"   Score: {analysis['accumulation_score']:.2f} (-1=distribution, +1=accumulation)")
    print(f"   Whales Accumulating: {analysis['whales_accumulating']}")
    print(f"   Buy Volume: ${analysis['total_buy_volume']:,.0f}")
    print(f"   Sell Volume: ${analysis['total_sell_volume']:,.0f}")
    print(f"   Net Flow: ${analysis['net_flow']:,.0f}")
    print(f"   Mega Whale Interest: {analysis['mega_whale_interest']:.1%}")
    print(f"   Transactions: {analysis['num_transactions']}")

    # Generate signal
    print("\n🚨 Generating whale signal...")
    signal = follower.generate_signal("ETH", 2000.0)

    if signal:
        print(f"\n✅ WHALE SIGNAL GENERATED:")
        print(f"   Token: {signal.token}")
        print(f"   Action: {signal.action.upper()}")
        print(f"   Entry Price: ${signal.entry_price:,.2f}")
        print(f"   Confidence: {signal.confidence:.0%}")
        print(f"   Whales Supporting: {signal.num_whales}")
        print(f"   Total Volume: ${signal.total_volume_usd:,.0f}")
        print(f"   Avg Whale Success: {signal.avg_whale_success:.0%}")
        print(f"   Suggested Hold: {signal.suggested_hold_time_hours:.0f} hours")
        print(f"   Reasoning: {signal.reasoning}")
    else:
        print("\n❌ No signal generated (insufficient whale activity)")

    # Show top whale tokens
    print("\n🏆 Top Tokens Held by Whales:")
    top_tokens = follower.get_top_whale_tokens(5)
    for i, (token, count) in enumerate(top_tokens, 1):
        print(f"   {i}. {token}: {count} whales")

    print("\n💡 Strategy Tips:")
    print("   ✅ Follow whales with high success rates")
    print("   ✅ Wait for multiple whales to accumulate")
    print("   ✅ Mega whales (>$100M) have higher weight")
    print("   ✅ Hold times similar to whale averages")
    print("   ⚠️  Whales can manipulate - use with caution")
    print("   ⚠️  Always combine with other analysis")
