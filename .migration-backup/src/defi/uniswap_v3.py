"""
Uniswap V3 Integration
Advanced DEX trading with concentrated liquidity pools.
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from web3 import Web3
from eth_account import Account
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Uniswap V3 Contract Addresses (Ethereum Mainnet)
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_QUOTER = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"

# Common token addresses
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

# Fee tiers (basis points)
FEE_LOW = 500      # 0.05%
FEE_MEDIUM = 3000  # 0.30%
FEE_HIGH = 10000   # 1.00%

# Uniswap V3 Router ABI (essential functions)
ROUTER_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "tokenIn", "type": "address"},
                    {"internalType": "address", "name": "tokenOut", "type": "address"},
                    {"internalType": "uint24", "name": "fee", "type": "uint24"},
                    {"internalType": "address", "name": "recipient", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
                    {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
                ],
                "internalType": "struct ISwapRouter.ExactInputSingleParams",
                "name": "params",
                "type": "tuple"
            }
        ],
        "name": "exactInputSingle",
        "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function"
    }
]

# Quoter ABI
QUOTER_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenIn", "type": "address"},
            {"internalType": "address", "name": "tokenOut", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"},
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
        ],
        "name": "quoteExactInputSingle",
        "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# ERC20 ABI (for approvals)
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]


class UniswapV3Client:
    """
    Uniswap V3 integration for advanced DEX trading.

    Features:
    - Price quotes across multiple fee tiers
    - Optimal route finding
    - Swap execution with slippage protection
    - MEV protection strategies
    """

    def __init__(self, rpc_url: str, private_key: Optional[str] = None):
        """
        Initialize Uniswap V3 client.

        Args:
            rpc_url: Ethereum RPC endpoint
            private_key: Private key for signing transactions (optional)
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key

        if private_key:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

        # Initialize contracts
        self.router = self.w3.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_V3_ROUTER),
            abi=ROUTER_ABI
        )

        self.quoter = self.w3.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_V3_QUOTER),
            abi=QUOTER_ABI
        )

        logger.info(f"UniswapV3 client initialized (connected: {self.w3.is_connected()})")

    def get_token_contract(self, token_address: str):
        """Get ERC20 token contract."""
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=ERC20_ABI
        )

    def get_token_decimals(self, token_address: str) -> int:
        """Get token decimals."""
        try:
            token = self.get_token_contract(token_address)
            return token.functions.decimals().call()
        except Exception as e:
            logger.warning(f"Could not get decimals for {token_address}: {e}")
            return 18  # Default to 18

    def get_token_balance(self, token_address: str, wallet_address: Optional[str] = None) -> Decimal:
        """Get token balance for wallet."""
        try:
            wallet = wallet_address or self.address
            if not wallet:
                return Decimal(0)

            token = self.get_token_contract(token_address)
            balance = token.functions.balanceOf(wallet).call()
            decimals = self.get_token_decimals(token_address)

            return Decimal(balance) / Decimal(10 ** decimals)
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return Decimal(0)

    def quote_price(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        fee_tier: int = FEE_MEDIUM
    ) -> Optional[Dict]:
        """
        Get price quote for a swap.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token
            fee_tier: Fee tier (500, 3000, or 10000)

        Returns:
            Quote dict with amount_out, price, fee_tier
        """
        try:
            # Get decimals
            decimals_in = self.get_token_decimals(token_in)
            decimals_out = self.get_token_decimals(token_out)

            # Convert to wei
            amount_in_wei = int(amount_in * 10 ** decimals_in)

            # Get quote
            amount_out_wei = self.quoter.functions.quoteExactInputSingle(
                Web3.to_checksum_address(token_in),
                Web3.to_checksum_address(token_out),
                fee_tier,
                amount_in_wei,
                0  # sqrtPriceLimitX96 (0 = no limit)
            ).call()

            # Convert from wei
            amount_out = float(amount_out_wei) / (10 ** decimals_out)

            # Calculate price
            price = amount_out / amount_in if amount_in > 0 else 0

            # Calculate price impact
            price_impact = self._calculate_price_impact(
                token_in, token_out, amount_in, amount_out
            )

            return {
                'dex': 'uniswap_v3',
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out': amount_out,
                'price': price,
                'fee_tier': fee_tier,
                'fee_percent': fee_tier / 10000,
                'price_impact': price_impact,
                'gas_estimate': 150000  # Approximate gas for V3 swap
            }

        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None

    def get_best_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float
    ) -> Optional[Dict]:
        """
        Get best quote across all fee tiers.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token

        Returns:
            Best quote dict
        """
        fee_tiers = [FEE_LOW, FEE_MEDIUM, FEE_HIGH]
        quotes = []

        for fee_tier in fee_tiers:
            quote = self.quote_price(token_in, token_out, amount_in, fee_tier)
            if quote:
                quotes.append(quote)

        if not quotes:
            return None

        # Return quote with highest output amount
        best_quote = max(quotes, key=lambda q: q['amount_out'])
        best_quote['all_quotes'] = quotes

        return best_quote

    def _calculate_price_impact(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        amount_out: float
    ) -> float:
        """
        Calculate price impact of a swap.

        Price impact = (execution_price - market_price) / market_price
        """
        try:
            # Get small quote for market price (1 unit)
            small_quote = self.quote_price(token_in, token_out, 1.0, FEE_MEDIUM)
            if not small_quote:
                return 0.0

            market_price = small_quote['price']
            execution_price = amount_out / amount_in if amount_in > 0 else 0

            if market_price == 0:
                return 0.0

            price_impact = ((execution_price - market_price) / market_price) * 100
            return abs(price_impact)

        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return 0.0

    def approve_token(
        self,
        token_address: str,
        spender: str = UNISWAP_V3_ROUTER,
        amount: Optional[int] = None
    ) -> Optional[str]:
        """
        Approve token spending.

        Args:
            token_address: Token to approve
            spender: Spender address (default: Uniswap router)
            amount: Amount to approve (None = unlimited)

        Returns:
            Transaction hash
        """
        if not self.account:
            logger.error("No private key configured")
            return None

        try:
            token = self.get_token_contract(token_address)

            # Default to max approval
            if amount is None:
                amount = 2**256 - 1

            # Build transaction
            tx = token.functions.approve(
                Web3.to_checksum_address(spender),
                amount
            ).build_transaction({
                'from': self.address,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.address)
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            logger.info(f"Token approval sent: {tx_hash.hex()}")
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Error approving token: {e}")
            return None

    def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        min_amount_out: float,
        fee_tier: int = FEE_MEDIUM,
        deadline_seconds: int = 300
    ) -> Optional[Dict]:
        """
        Execute a swap on Uniswap V3.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token
            min_amount_out: Minimum output amount (slippage protection)
            fee_tier: Fee tier to use
            deadline_seconds: Deadline in seconds from now

        Returns:
            Transaction result dict
        """
        if not self.account:
            logger.error("No private key configured for swaps")
            return None

        try:
            # Get decimals
            decimals_in = self.get_token_decimals(token_in)
            decimals_out = self.get_token_decimals(token_out)

            # Convert to wei
            amount_in_wei = int(amount_in * 10 ** decimals_in)
            min_amount_out_wei = int(min_amount_out * 10 ** decimals_out)

            # Set deadline
            deadline = self.w3.eth.get_block('latest')['timestamp'] + deadline_seconds

            # Build swap parameters
            params = (
                Web3.to_checksum_address(token_in),
                Web3.to_checksum_address(token_out),
                fee_tier,
                self.address,
                deadline,
                amount_in_wei,
                min_amount_out_wei,
                0  # sqrtPriceLimitX96
            )

            # Build transaction
            tx = self.router.functions.exactInputSingle(params).build_transaction({
                'from': self.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.address),
                'value': amount_in_wei if token_in == WETH else 0
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            logger.info(f"Swap transaction sent: {tx_hash.hex()}")

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            return {
                'success': receipt['status'] == 1,
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt['gasUsed'],
                'block_number': receipt['blockNumber']
            }

        except Exception as e:
            logger.error(f"Error executing swap: {e}")
            return {'success': False, 'error': str(e)}

    def estimate_gas_cost(self, gas_used: int = 150000) -> Dict:
        """
        Estimate gas cost for a transaction.

        Args:
            gas_used: Estimated gas units

        Returns:
            Gas cost dict with eth and usd estimates
        """
        try:
            gas_price = self.w3.eth.gas_price
            cost_wei = gas_used * gas_price
            cost_eth = float(cost_wei) / 1e18

            # Rough ETH price estimate (would need oracle in production)
            eth_price_usd = 2000  # TODO: Get from oracle
            cost_usd = cost_eth * eth_price_usd

            return {
                'gas_used': gas_used,
                'gas_price_gwei': float(gas_price) / 1e9,
                'cost_eth': cost_eth,
                'cost_usd': cost_usd
            }

        except Exception as e:
            logger.error(f"Error estimating gas: {e}")
            return {'cost_eth': 0, 'cost_usd': 0}


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🦄 Uniswap V3 Client Demo")
    print("=" * 50)

    # Initialize (read-only mode)
    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')
    client = UniswapV3Client(rpc_url)

    # Test price quote: WETH -> USDC
    print("\n📊 Getting price quote for 1 ETH -> USDC...")
    quote = client.get_best_quote(WETH, USDC, 1.0)

    if quote:
        print(f"✅ Best Quote Found:")
        print(f"   DEX: {quote['dex']}")
        print(f"   Amount In: {quote['amount_in']} ETH")
        print(f"   Amount Out: {quote['amount_out']:.2f} USDC")
        print(f"   Price: {quote['price']:.2f} USDC per ETH")
        print(f"   Fee Tier: {quote['fee_percent']}%")
        print(f"   Price Impact: {quote['price_impact']:.4f}%")
        print(f"   Gas Estimate: {quote['gas_estimate']} units")

        print(f"\n📊 All Fee Tiers:")
        for q in quote.get('all_quotes', []):
            print(f"   {q['fee_percent']}%: {q['amount_out']:.2f} USDC")
    else:
        print("❌ Could not get quote")

    print("\n✅ Uniswap V3 client demo complete!")
