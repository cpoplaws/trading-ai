"""
Polygon/MATIC blockchain interface.
"""
import os
import logging
from typing import Optional
from .ethereum_interface import EthereumInterface
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PolygonInterface(EthereumInterface):
    """
    Polygon (MATIC) blockchain interface.
    Inherits from EthereumInterface since Polygon is EVM-compatible.
    """
    
    # Polygon-specific token addresses
    TOKENS = {
        'WMATIC': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
        'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
        'DAI': '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063',
        'WETH': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
        'WBTC': '0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6',
    }
    
    def __init__(self, rpc_url: Optional[str] = None, private_key: Optional[str] = None):
        """
        Initialize Polygon interface.
        
        Args:
            rpc_url: RPC endpoint URL
            private_key: Private key for transactions
        """
        rpc = rpc_url or os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com')
        super().__init__(rpc_url=rpc, chain_id=137, private_key=private_key)
        
        logger.info("Polygon interface initialized")
    
    def get_matic_balance(self, address: Optional[str] = None) -> float:
        """
        Get MATIC balance (convenience method).
        
        Args:
            address: Wallet address
            
        Returns:
            MATIC balance
        """
        return self.get_balance(address)
    
    def estimate_gas_cost_usd(self, gas_units: int, matic_price_usd: float = 0.8) -> float:
        """
        Estimate transaction cost in USD.
        
        Args:
            gas_units: Estimated gas units
            matic_price_usd: Current MATIC price in USD
            
        Returns:
            Estimated cost in USD
        """
        gas_price = self.get_gas_price()['standard']
        cost_matic = (gas_units * gas_price) / 1e18
        return cost_matic * matic_price_usd


if __name__ == "__main__":
    # Test Polygon connection
    polygon = PolygonInterface()
    
    print("=== Polygon Interface Test ===")
    print(f"Connected: {polygon.w3.is_connected()}")
    print(f"Block number: {polygon.get_block_number()}")
    print(f"Chain ID: {polygon.chain_id}")
    
    gas_prices = polygon.get_gas_price()
    print(f"Gas prices (Gwei): slow={gas_prices['slow']/1e9:.2f}, standard={gas_prices['standard']/1e9:.2f}, fast={gas_prices['fast']/1e9:.2f}")
    
    print("✅ Polygon interface test completed!")
