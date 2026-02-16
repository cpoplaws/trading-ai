"""
On-Chain Metrics Calculator
Calculate various on-chain metrics for trading insights
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from blockchain_client import BlockchainClient
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenMetrics:
    """On-chain token metrics."""
    token_address: str
    token_symbol: str
    holder_count: int
    total_supply: float
    circulating_supply: float
    top_10_concentration: float  # % held by top 10
    active_addresses_24h: int
    transfer_count_24h: int
    volume_24h_usd: float
    price_usd: float
    market_cap_usd: float
    liquidity_usd: float
    timestamp: datetime


@dataclass
class NetworkMetrics:
    """Blockchain network metrics."""
    network: str
    block_height: int
    gas_price_gwei: float
    gas_price_percentiles: Dict[int, float]  # 10th, 50th, 90th
    tps: float  # Transactions per second
    active_addresses_24h: int
    new_addresses_24h: int
    transaction_count_24h: int
    avg_block_time: float
    timestamp: datetime


@dataclass
class DeFiMetrics:
    """DeFi ecosystem metrics."""
    total_value_locked: float
    dex_volume_24h: float
    lending_volume_24h: float
    stablecoin_supply: float
    liquidations_24h: float
    new_pools_24h: int
    timestamp: datetime


class OnChainMetricsCalculator:
    """
    Calculate comprehensive on-chain metrics.

    Features:
    - Token holder analysis
    - Network activity metrics
    - DeFi TVL and volume
    - Gas optimization metrics
    - Address activity tracking
    """

    def __init__(self, blockchain_client: BlockchainClient):
        """Initialize metrics calculator."""
        self.client = blockchain_client

    def calculate_token_metrics(
        self,
        token_address: str,
        token_symbol: str
    ) -> TokenMetrics:
        """
        Calculate comprehensive token metrics.

        Args:
            token_address: Token contract address
            token_symbol: Token symbol

        Returns:
            TokenMetrics object
        """
        # Get token transfers to analyze holders
        transfers = self.client.get_token_transfers(
            address=token_address
        )

        # Count unique holders
        holders = set()
        for tx in transfers:
            holders.add(tx['from'])
            holders.add(tx['to'])

        holder_count = len(holders)

        # Calculate 24h activity
        cutoff = datetime.now() - timedelta(days=1)
        recent_transfers = [
            tx for tx in transfers
            if datetime.fromtimestamp(int(tx['timeStamp'])) >= cutoff
        ]

        active_addresses = set()
        for tx in recent_transfers:
            active_addresses.add(tx['from'])
            active_addresses.add(tx['to'])

        transfer_count_24h = len(recent_transfers)
        active_addresses_24h = len(active_addresses)

        # Would need additional data for:
        # - Total/circulating supply (from contract)
        # - Top holder concentration (requires all holder balances)
        # - Volume, price, market cap (from price oracle/DEX)
        # - Liquidity (from DEX pools)

        return TokenMetrics(
            token_address=token_address,
            token_symbol=token_symbol,
            holder_count=holder_count,
            total_supply=0,  # Would query from contract
            circulating_supply=0,
            top_10_concentration=0,
            active_addresses_24h=active_addresses_24h,
            transfer_count_24h=transfer_count_24h,
            volume_24h_usd=0,
            price_usd=0,
            market_cap_usd=0,
            liquidity_usd=0,
            timestamp=datetime.now()
        )

    def calculate_network_metrics(self) -> NetworkMetrics:
        """
        Calculate network-wide metrics.

        Returns:
            NetworkMetrics object
        """
        # Get current block
        latest_block = self.client.get_block()
        block_height = latest_block['number']

        # Get gas price
        gas_price_gwei = self.client.get_gas_price()

        # Calculate gas percentiles
        # Would need historical gas data
        gas_percentiles = {
            10: gas_price_gwei * 0.8,
            50: gas_price_gwei,
            90: gas_price_gwei * 1.5
        }

        # Get recent blocks for TPS calculation
        blocks_to_check = 100
        total_txs = 0
        total_time = 0

        for i in range(blocks_to_check):
            block = self.client.get_block(block_height - i)
            if i > 0:
                prev_block = self.client.get_block(block_height - i + 1)
                block_time = block['timestamp'] - prev_block['timestamp']
                total_time += block_time

            total_txs += len(block['transactions'])

        avg_block_time = total_time / (blocks_to_check - 1) if blocks_to_check > 1 else 12
        tps = (total_txs / blocks_to_check) / avg_block_time if avg_block_time > 0 else 0

        return NetworkMetrics(
            network=self.client.network.value,
            block_height=block_height,
            gas_price_gwei=gas_price_gwei,
            gas_price_percentiles=gas_percentiles,
            tps=tps,
            active_addresses_24h=0,  # Would need full tx data
            new_addresses_24h=0,
            transaction_count_24h=total_txs * (86400 / (avg_block_time * blocks_to_check)),
            avg_block_time=avg_block_time,
            timestamp=datetime.now()
        )

    def calculate_gas_efficiency_score(
        self,
        transactions: List[Dict]
    ) -> float:
        """
        Calculate gas efficiency score for transactions.

        Args:
            transactions: List of transactions

        Returns:
            Efficiency score (0-100)
        """
        if not transactions:
            return 0

        network_gas = self.client.get_gas_price()

        total_overpay = 0
        for tx in transactions:
            tx_gas_price = int(tx.get('gasPrice', 0)) / 1e9  # Convert to Gwei
            overpay = max(0, tx_gas_price - network_gas)
            total_overpay += overpay

        avg_overpay = total_overpay / len(transactions)

        # Score: 100 if no overpay, decreases with overpayment
        # 50% overpay = 50 score, 100% overpay = 0 score
        score = max(0, 100 - (avg_overpay / network_gas) * 100)

        return score

    def analyze_concentration_risk(
        self,
        token_address: str,
        top_n: int = 10
    ) -> Dict:
        """
        Analyze token holder concentration risk.

        Args:
            token_address: Token contract address
            top_n: Number of top holders to analyze

        Returns:
            Concentration risk analysis
        """
        # Get all token holders and their balances
        # This would require querying all transfer events
        # and calculating current balances

        logger.info(f"Analyzing concentration risk for {token_address}")

        # Example structure
        return {
            'top_10_percentage': 0,  # % held by top 10
            'top_50_percentage': 0,  # % held by top 50
            'gini_coefficient': 0,   # Wealth distribution (0=equal, 1=unequal)
            'risk_level': 'medium',  # low/medium/high
            'holder_count': 0
        }

    def calculate_nvt_ratio(
        self,
        network_value: float,
        daily_transaction_volume: float
    ) -> float:
        """
        Calculate Network Value to Transactions (NVT) ratio.

        Similar to P/E ratio for stocks.
        Low NVT = undervalued, High NVT = overvalued

        Args:
            network_value: Market cap in USD
            daily_transaction_volume: 24h transaction volume in USD

        Returns:
            NVT ratio
        """
        if daily_transaction_volume == 0:
            return float('inf')

        return network_value / daily_transaction_volume

    def calculate_mvrv_ratio(
        self,
        market_cap: float,
        realized_cap: float
    ) -> float:
        """
        Calculate Market Value to Realized Value (MVRV) ratio.

        MVRV > 1: Market cap > realized value (profit territory)
        MVRV < 1: Market cap < realized value (loss territory)

        Args:
            market_cap: Current market cap
            realized_cap: Realized cap (based on last move price)

        Returns:
            MVRV ratio
        """
        if realized_cap == 0:
            return 0

        return market_cap / realized_cap

    def calculate_velocity(
        self,
        transaction_volume: float,
        market_cap: float
    ) -> float:
        """
        Calculate token velocity.

        Velocity = Transaction Volume / Market Cap

        High velocity = token changes hands frequently
        Low velocity = token held for longer periods

        Args:
            transaction_volume: Daily transaction volume
            market_cap: Market capitalization

        Returns:
            Token velocity
        """
        if market_cap == 0:
            return 0

        return transaction_volume / market_cap

    def detect_anomalies(
        self,
        token_address: str,
        lookback_days: int = 30
    ) -> List[Dict]:
        """
        Detect anomalous on-chain activity.

        Args:
            token_address: Token contract address
            lookback_days: Days to analyze

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Get historical transfers
        transfers = self.client.get_token_transfers(token_address)

        # Calculate daily volumes
        daily_volumes = {}
        for tx in transfers:
            date = datetime.fromtimestamp(int(tx['timeStamp'])).date()
            value = float(tx.get('value', 0)) / (10 ** int(tx.get('tokenDecimal', 18)))

            if date not in daily_volumes:
                daily_volumes[date] = 0
            daily_volumes[date] += value

        # Find anomalies (volume > 3x standard deviation)
        if len(daily_volumes) > 7:
            import statistics
            volumes = list(daily_volumes.values())
            mean_volume = statistics.mean(volumes)
            stdev_volume = statistics.stdev(volumes)

            for date, volume in daily_volumes.items():
                if volume > mean_volume + (3 * stdev_volume):
                    anomalies.append({
                        'date': date,
                        'type': 'high_volume',
                        'volume': volume,
                        'mean': mean_volume,
                        'deviation': (volume - mean_volume) / stdev_volume
                    })

        return anomalies

    def calculate_defi_metrics(self) -> DeFiMetrics:
        """
        Calculate DeFi ecosystem metrics.

        Returns:
            DeFiMetrics object
        """
        # Would integrate with DeFi Llama API or aggregate from protocols

        return DeFiMetrics(
            total_value_locked=0,
            dex_volume_24h=0,
            lending_volume_24h=0,
            stablecoin_supply=0,
            liquidations_24h=0,
            new_pools_24h=0,
            timestamp=datetime.now()
        )
