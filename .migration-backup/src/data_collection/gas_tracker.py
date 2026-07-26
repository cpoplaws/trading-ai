"""
Gas Price Tracker
Monitors real-time gas prices and predicts optimal gas for DEX transactions.
"""
import os
import requests
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from web3 import Web3
import statistics

logger = logging.getLogger(__name__)


@dataclass
class GasPrice:
    """Gas price information."""
    timestamp: datetime
    slow: float  # Gwei
    standard: float  # Gwei
    fast: float  # Gwei
    instant: float  # Gwei
    base_fee: Optional[float] = None  # Gwei (EIP-1559)
    priority_fee: Optional[float] = None  # Gwei (EIP-1559)


@dataclass
class GasCost:
    """Estimated gas cost for a transaction."""
    gas_limit: int
    gas_price_gwei: float
    cost_eth: float
    cost_usd: float


class GasTracker:
    """
    Track and predict gas prices.

    Features:
    - Real-time gas prices from multiple sources
    - EIP-1559 support (base fee + priority fee)
    - Historical gas price tracking
    - Optimal gas prediction
    - Cost estimation for DEX swaps
    - Gas price alerts

    Sources:
    - Etherscan Gas Tracker API
    - Blocknative Gas API
    - Web3 eth_gasPrice
    - Flashbots gas API
    """

    # Typical gas limits for common operations
    GAS_LIMITS = {
        'eth_transfer': 21000,
        'erc20_transfer': 65000,
        'uniswap_v2_swap': 150000,
        'uniswap_v3_swap': 180000,
        'approve': 46000,
        '1inch_swap': 200000,
        'aave_deposit': 300000,
        'aave_borrow': 350000
    }

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        etherscan_api_key: Optional[str] = None
    ):
        """
        Initialize gas tracker.

        Args:
            rpc_url: Ethereum RPC URL
            etherscan_api_key: Etherscan API key (for gas oracle)
        """
        self.rpc_url = rpc_url or os.getenv('ETHEREUM_RPC_URL', 'https://eth.llamarpc.com')
        self.etherscan_api_key = etherscan_api_key or os.getenv('ETHERSCAN_API_KEY')

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if self.w3.is_connected():
            logger.info("Gas tracker connected to Ethereum")
        else:
            logger.warning("Gas tracker not connected to Ethereum RPC")

        # Gas price history
        self.history: List[GasPrice] = []

    def get_gas_price_web3(self) -> Optional[float]:
        """
        Get current gas price from Web3 RPC.

        Returns:
            Gas price in Gwei
        """
        try:
            gas_price_wei = self.w3.eth.gas_price
            gas_price_gwei = self.w3.from_wei(gas_price_wei, 'gwei')
            return float(gas_price_gwei)

        except Exception as e:
            logger.error(f"Failed to get gas price from Web3: {e}")
            return None

    def get_gas_price_etherscan(self) -> Optional[GasPrice]:
        """
        Get gas prices from Etherscan Gas Tracker API.

        Returns:
            GasPrice object with multiple speed options
        """
        if not self.etherscan_api_key:
            logger.warning("No Etherscan API key provided")
            return None

        try:
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'gastracker',
                'action': 'gasoracle',
                'apikey': self.etherscan_api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data['status'] != '1':
                logger.error(f"Etherscan API error: {data.get('message')}")
                return None

            result = data['result']

            return GasPrice(
                timestamp=datetime.now(),
                slow=float(result['SafeGasPrice']),
                standard=float(result['ProposeGasPrice']),
                fast=float(result['FastGasPrice']),
                instant=float(result['FastGasPrice']) * 1.2,  # Estimate
                base_fee=float(result.get('suggestBaseFee', 0)),
                priority_fee=float(result.get('suggestBaseFee', 0)) * 0.1
            )

        except Exception as e:
            logger.error(f"Failed to get gas price from Etherscan: {e}")
            return None

    def get_gas_price_blocknative(self) -> Optional[GasPrice]:
        """
        Get gas prices from Blocknative Gas Platform API.

        Returns:
            GasPrice object with EIP-1559 data
        """
        try:
            url = "https://api.blocknative.com/gasprices/blockprices"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            prices = data['blockPrices'][0]['estimatedPrices']

            # Find different confidence levels
            confidence_90 = next((p for p in prices if p['confidence'] == 90), prices[0])
            confidence_70 = next((p for p in prices if p['confidence'] == 70), prices[0])
            confidence_99 = next((p for p in prices if p['confidence'] == 99), prices[-1])

            base_fee = data['blockPrices'][0]['baseFeePerGas']

            return GasPrice(
                timestamp=datetime.now(),
                slow=confidence_70['maxFeePerGas'],
                standard=confidence_90['maxFeePerGas'],
                fast=confidence_99['maxFeePerGas'],
                instant=confidence_99['maxFeePerGas'] * 1.1,
                base_fee=base_fee,
                priority_fee=confidence_90['maxPriorityFeePerGas']
            )

        except Exception as e:
            logger.error(f"Failed to get gas price from Blocknative: {e}")
            return None

    def get_current_gas_price(
        self,
        speed: str = 'standard'
    ) -> Optional[GasPrice]:
        """
        Get current gas price from best available source.

        Args:
            speed: Desired speed ('slow', 'standard', 'fast', 'instant')

        Returns:
            GasPrice object
        """
        # Try Blocknative first (most accurate)
        gas_price = self.get_gas_price_blocknative()
        if gas_price:
            logger.info(f"Got gas price from Blocknative: {gas_price.standard:.2f} Gwei")
            self.history.append(gas_price)
            return gas_price

        # Try Etherscan
        gas_price = self.get_gas_price_etherscan()
        if gas_price:
            logger.info(f"Got gas price from Etherscan: {gas_price.standard:.2f} Gwei")
            self.history.append(gas_price)
            return gas_price

        # Fallback to Web3
        web3_price = self.get_gas_price_web3()
        if web3_price:
            logger.info(f"Got gas price from Web3: {web3_price:.2f} Gwei")
            gas_price = GasPrice(
                timestamp=datetime.now(),
                slow=web3_price * 0.8,
                standard=web3_price,
                fast=web3_price * 1.2,
                instant=web3_price * 1.5
            )
            self.history.append(gas_price)
            return gas_price

        logger.error("Failed to get gas price from any source")
        return None

    def estimate_transaction_cost(
        self,
        operation: str,
        gas_price_gwei: Optional[float] = None,
        eth_price_usd: float = 2000.0
    ) -> GasCost:
        """
        Estimate cost for a transaction.

        Args:
            operation: Operation type (e.g., 'uniswap_v2_swap')
            gas_price_gwei: Gas price in Gwei (current price if None)
            eth_price_usd: ETH price in USD

        Returns:
            GasCost with estimates
        """
        # Get gas limit for operation
        gas_limit = self.GAS_LIMITS.get(operation, 200000)

        # Get current gas price if not provided
        if gas_price_gwei is None:
            current_gas = self.get_current_gas_price()
            if current_gas:
                gas_price_gwei = current_gas.standard
            else:
                gas_price_gwei = 50.0  # Fallback

        # Calculate cost
        cost_eth = (gas_limit * gas_price_gwei) / 1e9
        cost_usd = cost_eth * eth_price_usd

        return GasCost(
            gas_limit=gas_limit,
            gas_price_gwei=gas_price_gwei,
            cost_eth=cost_eth,
            cost_usd=cost_usd
        )

    def is_gas_price_good(
        self,
        current_gwei: float,
        threshold_gwei: float = 50.0,
        lookback_minutes: int = 60
    ) -> bool:
        """
        Check if current gas price is favorable.

        Args:
            current_gwei: Current gas price in Gwei
            threshold_gwei: Absolute threshold
            lookback_minutes: Historical comparison window

        Returns:
            True if gas price is good for trading
        """
        # Check absolute threshold
        if current_gwei > threshold_gwei:
            return False

        # Check against recent history
        if len(self.history) > 0:
            cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
            recent_prices = [
                gp.standard for gp in self.history
                if gp.timestamp > cutoff
            ]

            if recent_prices:
                avg_recent = statistics.mean(recent_prices)
                # Current price should be below average
                if current_gwei > avg_recent * 1.1:
                    return False

        return True

    def predict_next_block_gas(self) -> Optional[float]:
        """
        Predict gas price for next block using recent trends.

        Returns:
            Predicted gas price in Gwei
        """
        if len(self.history) < 3:
            return None

        # Use last 10 data points
        recent = self.history[-10:]
        prices = [gp.standard for gp in recent]

        # Simple prediction: weighted moving average
        weights = list(range(1, len(prices) + 1))
        weighted_avg = sum(p * w for p, w in zip(prices, weights)) / sum(weights)

        return weighted_avg

    def get_optimal_gas_price(
        self,
        urgency: str = 'medium'
    ) -> Optional[float]:
        """
        Get optimal gas price based on urgency.

        Args:
            urgency: 'low', 'medium', 'high', or 'critical'

        Returns:
            Optimal gas price in Gwei
        """
        current = self.get_current_gas_price()
        if not current:
            return None

        urgency_map = {
            'low': current.slow,
            'medium': current.standard,
            'high': current.fast,
            'critical': current.instant
        }

        return urgency_map.get(urgency, current.standard)

    def calculate_max_profitable_gas(
        self,
        profit_eth: float,
        gas_limit: int = 200000
    ) -> float:
        """
        Calculate maximum gas price that keeps trade profitable.

        Args:
            profit_eth: Expected profit in ETH
            gas_limit: Gas limit for transaction

        Returns:
            Maximum gas price in Gwei
        """
        # Max gas cost = profit
        # gas_limit * gas_price_wei = profit_eth * 1e18
        # gas_price_wei = (profit_eth * 1e18) / gas_limit
        # gas_price_gwei = gas_price_wei / 1e9

        max_gas_price_gwei = (profit_eth * 1e9) / gas_limit
        return max_gas_price_gwei

    def get_statistics(self) -> Dict:
        """
        Get gas price statistics from history.

        Returns:
            Dict with min, max, avg, median
        """
        if not self.history:
            return {}

        prices = [gp.standard for gp in self.history]

        return {
            'count': len(prices),
            'min': min(prices),
            'max': max(prices),
            'mean': statistics.mean(prices),
            'median': statistics.median(prices),
            'stdev': statistics.stdev(prices) if len(prices) > 1 else 0
        }


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("⛽ Gas Price Tracker Demo")
    print("=" * 60)

    # Initialize tracker
    tracker = GasTracker()

    # Get current gas prices
    print("\n1. Getting current gas prices...")
    gas_price = tracker.get_current_gas_price()

    if gas_price:
        print(f"✓ Current gas prices (Gwei):")
        print(f"  Slow:     {gas_price.slow:>6.1f} Gwei")
        print(f"  Standard: {gas_price.standard:>6.1f} Gwei ⭐")
        print(f"  Fast:     {gas_price.fast:>6.1f} Gwei")
        print(f"  Instant:  {gas_price.instant:>6.1f} Gwei")

        if gas_price.base_fee:
            print(f"\n  EIP-1559:")
            print(f"  Base Fee:     {gas_price.base_fee:>6.1f} Gwei")
            print(f"  Priority Fee: {gas_price.priority_fee:>6.1f} Gwei")

    # Estimate transaction costs
    print("\n2. Estimating transaction costs (ETH @ $2,000)...")
    operations = [
        'eth_transfer',
        'erc20_transfer',
        'uniswap_v2_swap',
        'approve'
    ]

    for op in operations:
        cost = tracker.estimate_transaction_cost(op, eth_price_usd=2000)
        print(f"\n  {op}:")
        print(f"    Gas limit: {cost.gas_limit:,}")
        print(f"    Gas price: {cost.gas_price_gwei:.1f} Gwei")
        print(f"    Cost: {cost.cost_eth:.6f} ETH (${cost.cost_usd:.2f})")

    # Check if gas is good for trading
    print("\n3. Checking if gas price is favorable...")
    if gas_price:
        is_good = tracker.is_gas_price_good(gas_price.standard, threshold_gwei=50)
        print(f"  Current: {gas_price.standard:.1f} Gwei")
        print(f"  Threshold: 50 Gwei")
        print(f"  Status: {'✅ GOOD' if is_good else '❌ TOO HIGH'} for trading")

    # Calculate max profitable gas
    print("\n4. Calculating maximum profitable gas price...")
    profit_scenarios = [0.01, 0.05, 0.1, 0.5]  # ETH profit

    for profit in profit_scenarios:
        max_gas = tracker.calculate_max_profitable_gas(
            profit_eth=profit,
            gas_limit=150000  # Uniswap swap
        )
        print(f"  Profit {profit} ETH → Max gas: {max_gas:.1f} Gwei")

    # Get optimal gas for different urgencies
    print("\n5. Optimal gas price by urgency...")
    urgencies = ['low', 'medium', 'high', 'critical']

    for urgency in urgencies:
        optimal = tracker.get_optimal_gas_price(urgency)
        if optimal:
            cost = tracker.estimate_transaction_cost('uniswap_v2_swap', optimal)
            print(f"  {urgency.capitalize():>8}: {optimal:>6.1f} Gwei (${cost.cost_usd:.2f})")

    print("\n✅ Gas tracker demo complete!")
    print("\nNext steps:")
    print("1. Set ETHERSCAN_API_KEY for better gas estimates")
    print("2. Monitor gas prices before executing DEX swaps")
    print("3. Use calculate_max_profitable_gas() to ensure profitability")
