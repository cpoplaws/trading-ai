"""
Avalanche C-Chain interface.
"""
import os
import logging
from typing import Optional
from .ethereum_interface import EthereumInterface
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AvalancheInterface(EthereumInterface):
    """
    Avalanche C-Chain blockchain interface.
    Inherits from EthereumInterface since Avalanche is EVM-compatible.
    """
    
    # Avalanche-specific token addresses
    TOKENS = {
        'WAVAX': '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7',
        'USDC': '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
        'USDT': '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7',
        'DAI': '0xd586E7F844cEa2F87f50152665BCbc2C279D8d70',
        'WETH': '0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB',
        'WBTC': '0x50b7545627a5162F82A992c33b87aDc75187B218',
    }
    
    def __init__(self, rpc_url: Optional[str] = None, private_key: Optional[str] = None):
        """
        Initialize Avalanche interface.
        
        Args:
            rpc_url: RPC endpoint URL
            private_key: Private key for transactions
        """
        rpc = rpc_url or os.getenv('AVALANCHE_RPC_URL', 'https://api.avax.network/ext/bc/C/rpc')
        super().__init__(rpc_url=rpc, chain_id=43114, private_key=private_key)
        
        logger.info("Avalanche interface initialized")
    
    def get_avax_balance(self, address: Optional[str] = None) -> float:
        """
        Get AVAX balance (convenience method).
        
        Args:
            address: Wallet address
            
        Returns:
            AVAX balance
        """
        return self.get_balance(address)
    
    def estimate_gas_cost_usd(self, gas_units: int, avax_price_usd: float = 35.0) -> float:
        """
        Estimate transaction cost in USD.
        
        Args:
            gas_units: Estimated gas units
            avax_price_usd: Current AVAX price in USD
            
        Returns:
            Estimated cost in USD
        """
        gas_price = self.get_gas_price()['standard']
        cost_avax = (gas_units * gas_price) / 1e18
        return cost_avax * avax_price_usd


if __name__ == "__main__":
    # Test Avalanche connection
    avax = AvalancheInterface()
    
    print("=== Avalanche Interface Test ===")
    print(f"Connected: {avax.w3.is_connected()}")
    print(f"Block number: {avax.get_block_number()}")
    print(f"Chain ID: {avax.chain_id}")
    
    gas_prices = avax.get_gas_price()
    print(f"Gas prices (nAVAX): slow={gas_prices['slow']/1e9:.2f}, standard={gas_prices['standard']/1e9:.2f}, fast={gas_prices['fast']/1e9:.2f}")
    
    print("✅ Avalanche interface test completed!")
