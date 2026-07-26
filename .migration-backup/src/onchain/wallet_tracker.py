"""
Wallet Tracker and Analyzer
Track wallet activities, token holdings, and trading patterns
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from blockchain_client import BlockchainClient
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenHolding:
    """Token holding information."""
    token_address: str
    token_symbol: str
    token_name: str
    balance: float
    price_usd: float
    value_usd: float
    decimals: int


@dataclass
class WalletProfile:
    """Comprehensive wallet profile."""
    address: str
    native_balance: float
    token_holdings: List[TokenHolding] = field(default_factory=list)
    total_value_usd: float = 0.0
    transaction_count: int = 0
    first_transaction_date: Optional[datetime] = None
    last_transaction_date: Optional[datetime] = None
    interacted_contracts: Set[str] = field(default_factory=set)
    labels: List[str] = field(default_factory=list)  # "whale", "degen", "trader", etc.


@dataclass
class TransactionPattern:
    """Transaction pattern analysis."""
    address: str
    total_transactions: int
    avg_tx_per_day: float
    most_active_hour: int
    most_traded_tokens: List[str]
    avg_gas_price_gwei: float
    total_gas_spent_eth: float
    profit_loss_usd: float
    win_rate: float


class WalletTracker:
    """
    Track and analyze wallet activities.

    Features:
    - Token balance tracking
    - Transaction history analysis
    - Trading pattern detection
    - Whale watching
    - Smart money tracking
    - Copy trading signals
    """

    def __init__(self, blockchain_client: BlockchainClient):
        """Initialize wallet tracker."""
        self.client = blockchain_client
        self.tracked_wallets: Dict[str, WalletProfile] = {}

    def add_wallet(self, address: str, label: str = None):
        """
        Add wallet to tracking list.

        Args:
            address: Wallet address
            label: Optional label for wallet
        """
        address = self.client.to_checksum_address(address)

        if address not in self.tracked_wallets:
            profile = self.get_wallet_profile(address)
            if label:
                profile.labels.append(label)
            self.tracked_wallets[address] = profile
            logger.info(f"Added wallet to tracking: {address}")

    def remove_wallet(self, address: str):
        """Remove wallet from tracking."""
        address = self.client.to_checksum_address(address)
        if address in self.tracked_wallets:
            del self.tracked_wallets[address]
            logger.info(f"Removed wallet from tracking: {address}")

    def get_wallet_profile(self, address: str) -> WalletProfile:
        """
        Get comprehensive wallet profile.

        Args:
            address: Wallet address

        Returns:
            WalletProfile with all holdings and stats
        """
        address = self.client.to_checksum_address(address)

        # Get native balance
        native_balance = self.client.get_balance(address)

        # Get transaction count
        tx_count = self.client.get_transaction_count(address)

        # Get token holdings
        token_holdings = self._get_token_holdings(address)

        # Calculate total value
        total_value = native_balance  # Native token value (would need price)
        total_value += sum(holding.value_usd for holding in token_holdings)

        # Get transaction history for dates
        transactions = self.client.get_transactions_by_address(address, sort='asc')

        first_tx_date = None
        last_tx_date = None
        interacted_contracts = set()

        if transactions:
            first_tx_date = datetime.fromtimestamp(int(transactions[0]['timeStamp']))
            last_tx_date = datetime.fromtimestamp(int(transactions[-1]['timeStamp']))

            # Collect interacted contracts
            for tx in transactions:
                if tx.get('to') and self.client.is_contract(tx['to']):
                    interacted_contracts.add(tx['to'])

        return WalletProfile(
            address=address,
            native_balance=native_balance,
            token_holdings=token_holdings,
            total_value_usd=total_value,
            transaction_count=tx_count,
            first_transaction_date=first_tx_date,
            last_transaction_date=last_tx_date,
            interacted_contracts=interacted_contracts
        )

    def _get_token_holdings(self, address: str) -> List[TokenHolding]:
        """
        Get all token holdings for wallet.

        Args:
            address: Wallet address

        Returns:
            List of TokenHolding objects
        """
        holdings = []

        # Get token transfers to identify owned tokens
        transfers = self.client.get_token_transfers(address)

        # Group by token contract
        token_balances = defaultdict(float)
        token_info = {}

        for transfer in transfers:
            token_address = transfer.get('contractAddress')
            value = float(transfer.get('value', 0))
            decimals = int(transfer.get('tokenDecimal', 18))

            # Adjust for decimals
            value_adjusted = value / (10 ** decimals)

            # Add or subtract based on from/to
            if transfer['to'].lower() == address.lower():
                token_balances[token_address] += value_adjusted
            elif transfer['from'].lower() == address.lower():
                token_balances[token_address] -= value_adjusted

            # Store token info
            if token_address not in token_info:
                token_info[token_address] = {
                    'symbol': transfer.get('tokenSymbol', 'UNKNOWN'),
                    'name': transfer.get('tokenName', 'Unknown Token'),
                    'decimals': decimals
                }

        # Create holdings for non-zero balances
        for token_address, balance in token_balances.items():
            if balance > 0:
                info = token_info[token_address]
                holdings.append(TokenHolding(
                    token_address=token_address,
                    token_symbol=info['symbol'],
                    token_name=info['name'],
                    balance=balance,
                    price_usd=0.0,  # Would need price oracle
                    value_usd=0.0,
                    decimals=info['decimals']
                ))

        return holdings

    def analyze_trading_pattern(
        self,
        address: str,
        days: int = 30
    ) -> TransactionPattern:
        """
        Analyze trading patterns for wallet.

        Args:
            address: Wallet address
            days: Number of days to analyze

        Returns:
            TransactionPattern with analysis
        """
        address = self.client.to_checksum_address(address)

        # Get recent transactions
        transactions = self.client.get_transactions_by_address(address)

        # Filter by time window
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_txs = [
            tx for tx in transactions
            if datetime.fromtimestamp(int(tx['timeStamp'])) >= cutoff_time
        ]

        if not recent_txs:
            return TransactionPattern(
                address=address,
                total_transactions=0,
                avg_tx_per_day=0,
                most_active_hour=0,
                most_traded_tokens=[],
                avg_gas_price_gwei=0,
                total_gas_spent_eth=0,
                profit_loss_usd=0,
                win_rate=0
            )

        # Calculate metrics
        total_txs = len(recent_txs)
        avg_per_day = total_txs / days

        # Most active hour
        hours = [datetime.fromtimestamp(int(tx['timeStamp'])).hour for tx in recent_txs]
        most_active_hour = max(set(hours), key=hours.count) if hours else 0

        # Gas analysis
        total_gas_used = sum(int(tx.get('gasUsed', 0)) for tx in recent_txs)
        total_gas_price = sum(int(tx.get('gasPrice', 0)) for tx in recent_txs)
        avg_gas_price_gwei = (total_gas_price / len(recent_txs)) / 1e9 if recent_txs else 0
        total_gas_eth = (total_gas_used * (total_gas_price / len(recent_txs))) / 1e18

        # Most traded tokens (from token transfers)
        token_transfers = self.client.get_token_transfers(address)
        recent_token_txs = [
            tx for tx in token_transfers
            if datetime.fromtimestamp(int(tx['timeStamp'])) >= cutoff_time
        ]

        token_counts = defaultdict(int)
        for tx in recent_token_txs:
            token_symbol = tx.get('tokenSymbol', 'UNKNOWN')
            token_counts[token_symbol] += 1

        most_traded = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        most_traded_tokens = [token for token, _ in most_traded]

        return TransactionPattern(
            address=address,
            total_transactions=total_txs,
            avg_tx_per_day=avg_per_day,
            most_active_hour=most_active_hour,
            most_traded_tokens=most_traded_tokens,
            avg_gas_price_gwei=avg_gas_price_gwei,
            total_gas_spent_eth=total_gas_eth,
            profit_loss_usd=0.0,  # Would need price data
            win_rate=0.0  # Would need trade outcome analysis
        )

    def detect_whale_movement(
        self,
        token_address: str,
        threshold_usd: float = 100000
    ) -> List[Dict]:
        """
        Detect large token movements (whale activity).

        Args:
            token_address: Token contract address
            threshold_usd: Minimum transaction value in USD

        Returns:
            List of whale transactions
        """
        # Get recent token transfers
        # This would typically monitor in real-time
        logger.info(f"Monitoring whale movements for {token_address}")

        whale_txs = []
        # Would implement real-time monitoring logic here

        return whale_txs

    def find_smart_money(
        self,
        min_profit: float = 10000,
        min_trades: int = 10,
        min_win_rate: float = 0.7
    ) -> List[str]:
        """
        Identify "smart money" wallets with consistent profits.

        Args:
            min_profit: Minimum total profit in USD
            min_trades: Minimum number of trades
            min_win_rate: Minimum win rate (0-1)

        Returns:
            List of smart money wallet addresses
        """
        smart_wallets = []

        for address, profile in self.tracked_wallets.items():
            pattern = self.analyze_trading_pattern(address)

            # Check criteria
            if (pattern.total_transactions >= min_trades and
                pattern.profit_loss_usd >= min_profit and
                pattern.win_rate >= min_win_rate):

                smart_wallets.append(address)
                logger.info(f"Smart money identified: {address}")

        return smart_wallets

    def generate_copy_trading_signals(
        self,
        smart_wallet: str,
        watch_tokens: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate copy trading signals from smart wallet.

        Args:
            smart_wallet: Smart money wallet to copy
            watch_tokens: Specific tokens to watch (optional)

        Returns:
            List of trading signals
        """
        signals = []

        # Monitor smart wallet transactions
        recent_txs = self.client.get_token_transfers(smart_wallet)

        for tx in recent_txs[:10]:  # Last 10 transactions
            token_symbol = tx.get('tokenSymbol', 'UNKNOWN')

            # Filter by watch list if provided
            if watch_tokens and token_symbol not in watch_tokens:
                continue

            # Determine action (buy if receiving, sell if sending)
            is_receiving = tx['to'].lower() == smart_wallet.lower()

            signals.append({
                'action': 'BUY' if is_receiving else 'SELL',
                'token': token_symbol,
                'token_address': tx['contractAddress'],
                'amount': float(tx['value']) / (10 ** int(tx.get('tokenDecimal', 18))),
                'wallet': smart_wallet,
                'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])),
                'tx_hash': tx['hash']
            })

        return signals

    def get_wallet_labels(self, address: str) -> List[str]:
        """
        Determine wallet labels based on activity.

        Args:
            address: Wallet address

        Returns:
            List of labels (whale, trader, degen, etc.)
        """
        labels = []
        profile = self.get_wallet_profile(address)

        # Whale (>$1M)
        if profile.total_value_usd > 1_000_000:
            labels.append("whale")

        # Active trader (>10 tx/day)
        pattern = self.analyze_trading_pattern(address)
        if pattern.avg_tx_per_day > 10:
            labels.append("active_trader")

        # Degen (high gas spending, many tx)
        if pattern.total_gas_spent_eth > 1.0 and pattern.total_transactions > 100:
            labels.append("degen")

        # Bot (very regular intervals)
        # Would need more sophisticated analysis

        # Smart money (high win rate, profitable)
        if pattern.win_rate > 0.7 and pattern.profit_loss_usd > 10000:
            labels.append("smart_money")

        return labels
