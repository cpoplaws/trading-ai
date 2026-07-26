"""
Coinbase Base L2 interface.
"""
import os
import logging
from typing import Optional
from .ethereum_interface import EthereumInterface
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BaseInterface(EthereumInterface):
    """
    Coinbase Base L2 blockchain interface.
    Inherits from EthereumInterface since Base is EVM-compatible.
    """
    
    # Base-specific token addresses
    TOKENS = {
        'WETH': '0x4200000000000000000000000000000000000006',
        'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
        'DAI': '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
    }
    
    def __init__(self, rpc_url: Optional[str] = None, private_key: Optional[str] = None):
        """
        Initialize Base interface.
        
        Args:
            rpc_url: RPC endpoint URL
            private_key: Private key for transactions
        """
        rpc = rpc_url or os.getenv('BASE_RPC_URL', 'https://mainnet.base.org')
        super().__init__(rpc_url=rpc, chain_id=8453, private_key=private_key)
        
        logger.info("Base interface initialized")
    
    def get_eth_balance(self, address: Optional[str] = None) -> float:
        """
        Get ETH balance on Base (convenience method).
        
        Args:
            address: Wallet address
            
        Returns:
            ETH balance
        """
        return self.get_balance(address)
    
    def estimate_gas_cost_usd(self, gas_units: int, eth_price_usd: float = 3000.0) -> float:
        """
        Estimate transaction cost in USD.
        
        Args:
            gas_units: Estimated gas units
            eth_price_usd: Current ETH price in USD
            
        Returns:
            Estimated cost in USD
        """
        gas_price = self.get_gas_price()['standard']
        cost_eth = (gas_units * gas_price) / 1e18
        return cost_eth * eth_price_usd


if __name__ == "__main__":
    # Test Base connection
    base = BaseInterface()
    
    print("=== Base Interface Test ===")
    print(f"Connected: {base.w3.is_connected()}")
    print(f"Block number: {base.get_block_number()}")
    print(f"Chain ID: {base.chain_id}")
    
    gas_prices = base.get_gas_price()
    print(f"Gas prices (Gwei): slow={gas_prices['slow']/1e9:.2f}, standard={gas_prices['standard']/1e9:.2f}, fast={gas_prices['fast']/1e9:.2f}")
    
    print("✅ Base interface test completed!")
