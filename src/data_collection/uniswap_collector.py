"""
Uniswap V2/V3 Data Collector
Collects pool data, swap events, and liquidity information from Uniswap DEX.
"""
import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal

from web3 import Web3
from web3.contract import Contract

logger = logging.getLogger(__name__)


# Uniswap V2 Router address (Ethereum mainnet)
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
UNISWAP_V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

# Uniswap V3 Router address
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

# Common token addresses
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
DAI_ADDRESS = "0x6B175474E89094C44Da98b954EedeAC495271d0F"


@dataclass
class UniswapPool:
    """Uniswap liquidity pool."""
    address: str
    token0: str
    token1: str
    token0_symbol: str
    token1_symbol: str
    reserve0: float
    reserve1: float
    price: float  # token1/token0
    liquidity_usd: float
    version: str  # 'v2' or 'v3'


@dataclass
class UniswapSwap:
    """Uniswap swap event."""
    timestamp: datetime
    tx_hash: str
    pool: str
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    price: float
    trader: str


class UniswapCollector:
    """
    Collect data from Uniswap V2 and V3.

    Features:
    - Get pool reserves and liquidity
    - Calculate prices with low slippage
    - Track swap events
    - Monitor large transactions
    - Calculate price impact for different sizes
    """

    # Minimal ABI for ERC20 tokens
    ERC20_ABI = [
        {
            "constant": True,
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "symbol",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "type": "function"
        }
    ]

    # Uniswap V2 Pair ABI (minimal)
    UNISWAP_V2_PAIR_ABI = [
        {
            "constant": True,
            "inputs": [],
            "name": "getReserves",
            "outputs": [
                {"name": "reserve0", "type": "uint112"},
                {"name": "reserve1", "type": "uint112"},
                {"name": "blockTimestampLast", "type": "uint32"}
            ],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "token0",
            "outputs": [{"name": "", "type": "address"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "token1",
            "outputs": [{"name": "", "type": "address"}],
            "type": "function"
        }
    ]

    # Uniswap V2 Factory ABI (minimal)
    UNISWAP_V2_FACTORY_ABI = [
        {
            "constant": True,
            "inputs": [
                {"name": "tokenA", "type": "address"},
                {"name": "tokenB", "type": "address"}
            ],
            "name": "getPair",
            "outputs": [{"name": "pair", "type": "address"}],
            "type": "function"
        }
    ]

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        network: str = "ethereum"
    ):
        """
        Initialize Uniswap collector.

        Args:
            rpc_url: Ethereum RPC URL (or set ETHEREUM_RPC_URL env var)
            network: Network name ('ethereum', 'polygon', etc.)
        """
        self.network = network
        self.rpc_url = rpc_url or os.getenv('ETHEREUM_RPC_URL', 'https://eth.llamarpc.com')

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if not self.w3.is_connected():
            logger.warning(f"Not connected to {network} RPC")
        else:
            logger.info(f"Connected to {network} at block {self.w3.eth.block_number:,}")

        # Initialize contracts
        self.v2_factory = self.w3.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_V2_FACTORY),
            abi=self.UNISWAP_V2_FACTORY_ABI
        )

    def get_token_info(self, token_address: str) -> Dict:
        """
        Get token information (symbol, decimals).

        Args:
            token_address: Token contract address

        Returns:
            Dict with symbol and decimals
        """
        try:
            token = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.ERC20_ABI
            )

            symbol = token.functions.symbol().call()
            decimals = token.functions.decimals().call()

            return {
                'address': token_address,
                'symbol': symbol,
                'decimals': decimals
            }

        except Exception as e:
            logger.error(f"Failed to get token info for {token_address}: {e}")
            return {
                'address': token_address,
                'symbol': 'UNKNOWN',
                'decimals': 18
            }

    def get_pair_address(
        self,
        token_a: str,
        token_b: str
    ) -> Optional[str]:
        """
        Get Uniswap V2 pair address for two tokens.

        Args:
            token_a: First token address
            token_b: Second token address

        Returns:
            Pair contract address or None if doesn't exist
        """
        try:
            pair_address = self.v2_factory.functions.getPair(
                Web3.to_checksum_address(token_a),
                Web3.to_checksum_address(token_b)
            ).call()

            # Check if pair exists (non-zero address)
            if pair_address == "0x0000000000000000000000000000000000000000":
                return None

            return pair_address

        except Exception as e:
            logger.error(f"Failed to get pair address: {e}")
            return None

    def get_pool_info(
        self,
        token_a: str,
        token_b: str
    ) -> Optional[UniswapPool]:
        """
        Get Uniswap V2 pool information.

        Args:
            token_a: First token address
            token_b: Second token address

        Returns:
            Pool information with reserves and price
        """
        try:
            # Get pair address
            pair_address = self.get_pair_address(token_a, token_b)
            if not pair_address:
                logger.warning(f"No pool found for {token_a}/{token_b}")
                return None

            # Create pair contract
            pair = self.w3.eth.contract(
                address=Web3.to_checksum_address(pair_address),
                abi=self.UNISWAP_V2_PAIR_ABI
            )

            # Get reserves
            reserves = pair.functions.getReserves().call()
            reserve0_raw = reserves[0]
            reserve1_raw = reserves[1]

            # Get token addresses
            token0_address = pair.functions.token0().call()
            token1_address = pair.functions.token1().call()

            # Get token info
            token0_info = self.get_token_info(token0_address)
            token1_info = self.get_token_info(token1_address)

            # Convert reserves to human-readable
            reserve0 = reserve0_raw / (10 ** token0_info['decimals'])
            reserve1 = reserve1_raw / (10 ** token1_info['decimals'])

            # Calculate price (token1 per token0)
            price = reserve1 / reserve0 if reserve0 > 0 else 0

            return UniswapPool(
                address=pair_address,
                token0=token0_address,
                token1=token1_address,
                token0_symbol=token0_info['symbol'],
                token1_symbol=token1_info['symbol'],
                reserve0=reserve0,
                reserve1=reserve1,
                price=price,
                liquidity_usd=0,  # TODO: Calculate USD value
                version='v2'
            )

        except Exception as e:
            logger.error(f"Failed to get pool info: {e}")
            return None

    def get_amount_out(
        self,
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee: float = 0.003  # 0.3% for Uniswap V2
    ) -> float:
        """
        Calculate output amount for a swap using constant product formula.

        Args:
            amount_in: Input amount
            reserve_in: Input token reserve
            reserve_out: Output token reserve
            fee: Trading fee (0.003 = 0.3%)

        Returns:
            Output amount after fees
        """
        if reserve_in == 0 or reserve_out == 0:
            return 0

        # Apply fee
        amount_in_with_fee = amount_in * (1 - fee)

        # Constant product formula: x * y = k
        # amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee

        return numerator / denominator

    def calculate_price_impact(
        self,
        amount_in: float,
        token_in: str,
        token_out: str
    ) -> Dict:
        """
        Calculate price impact for a swap.

        Args:
            amount_in: Amount to swap
            token_in: Input token address
            token_out: Output token address

        Returns:
            Dict with amount_out, price, and price_impact
        """
        try:
            pool = self.get_pool_info(token_in, token_out)
            if not pool:
                return {'error': 'Pool not found'}

            # Determine direction
            if pool.token0.lower() == token_in.lower():
                reserve_in = pool.reserve0
                reserve_out = pool.reserve1
                spot_price = pool.price
            else:
                reserve_in = pool.reserve1
                reserve_out = pool.reserve0
                spot_price = 1 / pool.price if pool.price > 0 else 0

            # Calculate output amount
            amount_out = self.get_amount_out(amount_in, reserve_in, reserve_out)

            # Calculate execution price
            execution_price = amount_out / amount_in if amount_in > 0 else 0

            # Calculate price impact
            price_impact = ((execution_price - spot_price) / spot_price * 100) if spot_price > 0 else 0

            return {
                'amount_in': amount_in,
                'amount_out': amount_out,
                'spot_price': spot_price,
                'execution_price': execution_price,
                'price_impact_percent': abs(price_impact),
                'reserve_in': reserve_in,
                'reserve_out': reserve_out
            }

        except Exception as e:
            logger.error(f"Failed to calculate price impact: {e}")
            return {'error': str(e)}

    def get_best_path(
        self,
        token_in: str,
        token_out: str,
        amount_in: float
    ) -> List[str]:
        """
        Find best swap path (may route through WETH).

        Args:
            token_in: Input token
            token_out: Output token
            amount_in: Input amount

        Returns:
            List of token addresses representing the path
        """
        # Try direct path first
        direct_pool = self.get_pool_info(token_in, token_out)
        if direct_pool and direct_pool.reserve0 > 0:
            return [token_in, token_out]

        # Try path through WETH
        if token_in.lower() != WETH_ADDRESS.lower() and token_out.lower() != WETH_ADDRESS.lower():
            path_via_weth = [token_in, WETH_ADDRESS, token_out]

            # Check if both pools exist
            pool1 = self.get_pool_info(token_in, WETH_ADDRESS)
            pool2 = self.get_pool_info(WETH_ADDRESS, token_out)

            if pool1 and pool2:
                return path_via_weth

        logger.warning(f"No path found from {token_in} to {token_out}")
        return [token_in, token_out]


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🦄 Uniswap Collector Demo")
    print("=" * 60)

    # Initialize collector
    collector = UniswapCollector()

    if not collector.w3.is_connected():
        print("❌ Not connected to Ethereum RPC")
        print("Set ETHEREUM_RPC_URL environment variable or use Alchemy/Infura")
        exit(1)

    print(f"✓ Connected to Ethereum")
    print(f"  Block: {collector.w3.eth.block_number:,}")

    # Get WETH/USDC pool info
    print("\n1. Getting WETH/USDC pool information...")
    pool = collector.get_pool_info(WETH_ADDRESS, USDC_ADDRESS)

    if pool:
        print(f"✓ Pool found: {pool.address}")
        print(f"  {pool.token0_symbol}: {pool.reserve0:,.2f}")
        print(f"  {pool.token1_symbol}: {pool.reserve1:,.2f}")
        print(f"  Price: {pool.price:,.2f} {pool.token1_symbol}/{pool.token0_symbol}")

        # Calculate price impact for different sizes
        print("\n2. Calculating price impact for different swap sizes...")
        test_amounts = [0.1, 1, 10, 100]  # ETH amounts

        for amount in test_amounts:
            impact = collector.calculate_price_impact(
                amount,
                WETH_ADDRESS,
                USDC_ADDRESS
            )

            if 'error' not in impact:
                print(f"\n  Swap {amount} ETH:")
                print(f"    Receive: {impact['amount_out']:,.2f} USDC")
                print(f"    Spot price: ${impact['spot_price']:,.2f}")
                print(f"    Execution price: ${impact['execution_price']:,.2f}")
                print(f"    Price impact: {impact['price_impact_percent']:.3f}%")

    # Get DAI/USDC pool
    print("\n3. Getting DAI/USDC pool (stablecoin pair)...")
    stable_pool = collector.get_pool_info(DAI_ADDRESS, USDC_ADDRESS)

    if stable_pool:
        print(f"✓ Pool found: {stable_pool.address}")
        print(f"  {stable_pool.token0_symbol}: {stable_pool.reserve0:,.2f}")
        print(f"  {stable_pool.token1_symbol}: {stable_pool.reserve1:,.2f}")
        print(f"  Price: {stable_pool.price:.6f} (should be ~1.0)")

    # Find best path
    print("\n4. Finding best swap path...")
    path = collector.get_best_path(DAI_ADDRESS, USDC_ADDRESS, 1000)
    print(f"✓ Path: {' → '.join([collector.get_token_info(addr)['symbol'] for addr in path])}")

    print("\n✅ Uniswap collector demo complete!")
    print("\nNext steps:")
    print("1. Set ETHEREUM_RPC_URL (Alchemy or Infura) for better reliability")
    print("2. Use collector to monitor pools and find arbitrage")
    print("3. Integrate with gas tracker to calculate profitability")
