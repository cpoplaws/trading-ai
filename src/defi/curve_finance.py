"""
Curve Finance Integration
Specialized for stablecoin and similar-asset swaps with low slippage.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from decimal import Decimal
from web3 import Web3
from eth_account import Account
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Curve pool addresses (Ethereum Mainnet)
CURVE_3POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"  # DAI/USDC/USDT
CURVE_STETH = "0xDC24316b9AE028F1497c275EB9192a3Ea0f67022"  # ETH/stETH
CURVE_TRICRYPTO = "0xD51a44d3FaE010294C616388b506AcdA1bfAAE46"  # USDT/WBTC/WETH

# Common stablecoins
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

# Simplified Curve Pool ABI
CURVE_POOL_ABI = [
    {
        "name": "get_dy",
        "outputs": [{"type": "uint256", "name": ""}],
        "inputs": [
            {"type": "int128", "name": "i"},
            {"type": "int128", "name": "j"},
            {"type": "uint256", "name": "dx"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "name": "exchange",
        "outputs": [{"type": "uint256", "name": ""}],
        "inputs": [
            {"type": "int128", "name": "i"},
            {"type": "int128", "name": "j"},
            {"type": "uint256", "name": "dx"},
            {"type": "uint256", "name": "min_dy"}
        ],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "name": "coins",
        "outputs": [{"type": "address", "name": ""}],
        "inputs": [{"type": "uint256", "name": "arg0"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "name": "balances",
        "outputs": [{"type": "uint256", "name": ""}],
        "inputs": [{"type": "uint256", "name": "arg0"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# ERC20 ABI
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
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]


class CurveFinanceClient:
    """
    Curve Finance integration for low-slippage stable asset swaps.

    Features:
    - Optimized for stablecoin swaps
    - Minimal price impact
    - Low fees for similar assets
    - Multiple pool support
    """

    # Pool configurations
    POOLS = {
        '3pool': {
            'address': CURVE_3POOL,
            'coins': [DAI, USDC, USDT],
            'name': '3Pool (DAI/USDC/USDT)',
            'type': 'stableswap'
        },
        'steth': {
            'address': CURVE_STETH,
            'coins': ['ETH', '0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84'],  # ETH, stETH
            'name': 'stETH Pool',
            'type': 'eth'
        }
    }

    def __init__(self, rpc_url: str, private_key: Optional[str] = None):
        """
        Initialize Curve Finance client.

        Args:
            rpc_url: Ethereum RPC endpoint
            private_key: Private key for signing transactions
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key

        if private_key:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

        logger.info(f"Curve Finance client initialized (connected: {self.w3.is_connected()})")

    def get_pool_contract(self, pool_address: str):
        """Get Curve pool contract."""
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=CURVE_POOL_ABI
        )

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
            return 18

    def find_pool_for_pair(self, token_in: str, token_out: str) -> Optional[Dict]:
        """
        Find Curve pool that supports a token pair.

        Args:
            token_in: Input token address
            token_out: Output token address

        Returns:
            Pool config dict if found
        """
        token_in = token_in.lower()
        token_out = token_out.lower()

        for pool_id, pool_config in self.POOLS.items():
            coins = [c.lower() if c != 'ETH' else 'eth' for c in pool_config['coins']]

            # Check if both tokens are in this pool
            if token_in in coins and token_out in coins:
                return {
                    'pool_id': pool_id,
                    'pool_address': pool_config['address'],
                    'pool_name': pool_config['name'],
                    'coin_index_in': coins.index(token_in),
                    'coin_index_out': coins.index(token_out),
                    'coins': pool_config['coins']
                }

        return None

    def quote_price(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        pool_address: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get price quote for a swap on Curve.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token
            pool_address: Specific pool to use (optional)

        Returns:
            Quote dict
        """
        try:
            # Find pool if not specified
            if pool_address:
                pool_info = None
                for pool_id, config in self.POOLS.items():
                    if config['address'].lower() == pool_address.lower():
                        pool_info = {
                            'pool_address': config['address'],
                            'pool_name': config['name'],
                            'coins': config['coins']
                        }
                        # Find coin indices
                        coins_lower = [c.lower() for c in config['coins']]
                        pool_info['coin_index_in'] = coins_lower.index(token_in.lower())
                        pool_info['coin_index_out'] = coins_lower.index(token_out.lower())
                        break
            else:
                pool_info = self.find_pool_for_pair(token_in, token_out)

            if not pool_info:
                logger.warning(f"No Curve pool found for {token_in}/{token_out}")
                return None

            # Get pool contract
            pool = self.get_pool_contract(pool_info['pool_address'])

            # Get decimals
            decimals_in = self.get_token_decimals(token_in)
            decimals_out = self.get_token_decimals(token_out)

            # Convert to wei
            amount_in_wei = int(amount_in * 10 ** decimals_in)

            # Get quote using get_dy
            amount_out_wei = pool.functions.get_dy(
                pool_info['coin_index_in'],
                pool_info['coin_index_out'],
                amount_in_wei
            ).call()

            # Convert from wei
            amount_out = float(amount_out_wei) / (10 ** decimals_out)

            # Calculate price
            price = amount_out / amount_in if amount_in > 0 else 0

            # Curve typically has 0.04% fee for stableswaps
            fee_percent = 0.04

            # Calculate effective exchange rate
            exchange_rate = amount_out / amount_in if amount_in > 0 else 0

            # Price impact (should be very low on Curve)
            price_impact = abs(1 - exchange_rate) * 100 if exchange_rate > 0 else 0

            return {
                'dex': 'curve_finance',
                'pool_name': pool_info['pool_name'],
                'pool_address': pool_info['pool_address'],
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out': amount_out,
                'price': price,
                'exchange_rate': exchange_rate,
                'fee_percent': fee_percent,
                'price_impact': price_impact,
                'gas_estimate': 100000,  # Curve is gas-efficient
                'coin_index_in': pool_info['coin_index_in'],
                'coin_index_out': pool_info['coin_index_out']
            }

        except Exception as e:
            logger.error(f"Error getting Curve quote: {e}")
            return None

    def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        min_amount_out: float,
        pool_address: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Execute swap on Curve Finance.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount to swap
            min_amount_out: Minimum output (slippage protection)
            pool_address: Specific pool to use

        Returns:
            Transaction result dict
        """
        if not self.account:
            logger.error("No private key configured for swaps")
            return None

        try:
            # Get quote to find pool and indices
            quote = self.quote_price(token_in, token_out, amount_in, pool_address)
            if not quote:
                return {'success': False, 'error': 'No pool found'}

            # Get pool contract
            pool = self.get_pool_contract(quote['pool_address'])

            # Get decimals
            decimals_in = self.get_token_decimals(token_in)
            decimals_out = self.get_token_decimals(token_out)

            # Convert to wei
            amount_in_wei = int(amount_in * 10 ** decimals_in)
            min_amount_out_wei = int(min_amount_out * 10 ** decimals_out)

            # Build transaction
            tx = pool.functions.exchange(
                quote['coin_index_in'],
                quote['coin_index_out'],
                amount_in_wei,
                min_amount_out_wei
            ).build_transaction({
                'from': self.address,
                'gas': 150000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.address)
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            logger.info(f"Curve swap sent: {tx_hash.hex()}")

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            return {
                'success': receipt['status'] == 1,
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt['gasUsed'],
                'block_number': receipt['blockNumber'],
                'pool_used': quote['pool_name']
            }

        except Exception as e:
            logger.error(f"Error executing Curve swap: {e}")
            return {'success': False, 'error': str(e)}

    def get_pool_balances(self, pool_address: str) -> Dict:
        """
        Get balances for all coins in a pool.

        Args:
            pool_address: Curve pool address

        Returns:
            Dict of coin balances
        """
        try:
            pool = self.get_pool_contract(pool_address)

            # Find pool config
            pool_config = None
            for config in self.POOLS.values():
                if config['address'].lower() == pool_address.lower():
                    pool_config = config
                    break

            if not pool_config:
                return {}

            balances = {}
            for i, coin_address in enumerate(pool_config['coins']):
                try:
                    balance_wei = pool.functions.balances(i).call()
                    decimals = self.get_token_decimals(coin_address) if coin_address != 'ETH' else 18
                    balance = float(balance_wei) / (10 ** decimals)
                    balances[coin_address] = balance
                except Exception as e:
                    logger.warning(f"Could not get balance for coin {i}: {e}")

            return balances

        except Exception as e:
            logger.error(f"Error getting pool balances: {e}")
            return {}

    def compare_to_1_1_rate(self, token_in: str, token_out: str, amount: float = 1.0) -> Dict:
        """
        Compare Curve rate to ideal 1:1 rate (useful for stablecoins).

        Args:
            token_in: Input token
            token_out: Output token
            amount: Amount to check

        Returns:
            Comparison dict
        """
        quote = self.quote_price(token_in, token_out, amount)
        if not quote:
            return {}

        ideal_rate = 1.0
        actual_rate = quote['exchange_rate']
        deviation = ((actual_rate - ideal_rate) / ideal_rate) * 100

        return {
            'ideal_rate': ideal_rate,
            'actual_rate': actual_rate,
            'deviation_percent': deviation,
            'amount_in': amount,
            'amount_out': quote['amount_out'],
            'loss_to_ideal': amount - quote['amount_out'],
            'is_favorable': actual_rate > ideal_rate
        }


if __name__ == '__main__':
    import os
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🌊 Curve Finance Client Demo")
    print("=" * 50)

    # Initialize
    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')
    client = CurveFinanceClient(rpc_url)

    # Test stablecoin swap: USDC -> DAI
    print("\n📊 Getting quote for 1000 USDC -> DAI...")
    quote = client.quote_price(USDC, DAI, 1000.0)

    if quote:
        print(f"✅ Quote Found:")
        print(f"   DEX: {quote['dex']}")
        print(f"   Pool: {quote['pool_name']}")
        print(f"   Amount In: {quote['amount_in']:.2f} USDC")
        print(f"   Amount Out: {quote['amount_out']:.2f} DAI")
        print(f"   Exchange Rate: {quote['exchange_rate']:.6f}")
        print(f"   Fee: {quote['fee_percent']}%")
        print(f"   Price Impact: {quote['price_impact']:.4f}%")
        print(f"   Gas Estimate: {quote['gas_estimate']} units")

        # Compare to ideal 1:1 rate
        print(f"\n📊 Comparing to ideal 1:1 rate...")
        comparison = client.compare_to_1_1_rate(USDC, DAI, 1000.0)
        print(f"   Ideal: 1.0")
        print(f"   Actual: {comparison['actual_rate']:.6f}")
        print(f"   Deviation: {comparison['deviation_percent']:.4f}%")
        print(f"   Loss to ideal: ${comparison['loss_to_ideal']:.4f}")
    else:
        print("❌ Could not get quote")

    # Get pool balances
    print(f"\n💰 3Pool Balances:")
    balances = client.get_pool_balances(CURVE_3POOL)
    for coin, balance in balances.items():
        print(f"   {coin}: {balance:,.2f}")

    print("\n✅ Curve Finance demo complete!")
