"""
Unified multi-chain abstraction layer for managing connections across multiple blockchains.
"""
import logging
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    BASE = "base"
    SOLANA = "solana"


@dataclass
class ChainConfig:
    """Configuration for a blockchain network."""
    chain_id: int
    name: str
    rpc_urls: List[str]
    native_token: str
    explorer_url: str
    is_evm: bool = True


class ChainManager:
    """
    Unified interface for managing multiple blockchain connections.
    Provides fallback RPC endpoints, connection pooling, and automatic switching.
    """
    
    CHAIN_CONFIGS = {
        Chain.ETHEREUM: ChainConfig(
            chain_id=1,
            name="Ethereum Mainnet",
            rpc_urls=[
                "https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
                "https://mainnet.infura.io/v3/${INFURA_KEY}",
                "https://cloudflare-eth.com"
            ],
            native_token="ETH",
            explorer_url="https://etherscan.io"
        ),
        Chain.POLYGON: ChainConfig(
            chain_id=137,
            name="Polygon",
            rpc_urls=[
                "https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
                "https://polygon-rpc.com",
                "https://rpc-mainnet.matic.network"
            ],
            native_token="MATIC",
            explorer_url="https://polygonscan.com"
        ),
        Chain.ARBITRUM: ChainConfig(
            chain_id=42161,
            name="Arbitrum One",
            rpc_urls=[
                "https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
                "https://arb1.arbitrum.io/rpc",
                "https://arbitrum.llamarpc.com"
            ],
            native_token="ETH",
            explorer_url="https://arbiscan.io"
        ),
        Chain.OPTIMISM: ChainConfig(
            chain_id=10,
            name="Optimism",
            rpc_urls=[
                "https://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
                "https://mainnet.optimism.io",
                "https://optimism.llamarpc.com"
            ],
            native_token="ETH",
            explorer_url="https://optimistic.etherscan.io"
        ),
        Chain.BSC: ChainConfig(
            chain_id=56,
            name="BNB Smart Chain",
            rpc_urls=[
                "https://bsc-dataseed1.binance.org",
                "https://bsc-dataseed2.binance.org",
                "https://bsc-dataseed3.binance.org"
            ],
            native_token="BNB",
            explorer_url="https://bscscan.com"
        ),
        Chain.AVALANCHE: ChainConfig(
            chain_id=43114,
            name="Avalanche C-Chain",
            rpc_urls=[
                "https://api.avax.network/ext/bc/C/rpc",
                "https://avalanche.public-rpc.com",
                "https://avax.meowrpc.com"
            ],
            native_token="AVAX",
            explorer_url="https://snowtrace.io"
        ),
        Chain.BASE: ChainConfig(
            chain_id=8453,
            name="Base",
            rpc_urls=[
                "https://mainnet.base.org",
                "https://base.llamarpc.com",
                "https://base.meowrpc.com"
            ],
            native_token="ETH",
            explorer_url="https://basescan.org"
        ),
        Chain.SOLANA: ChainConfig(
            chain_id=0,  # Solana doesn't use chain_id
            name="Solana",
            rpc_urls=[
                "https://api.mainnet-beta.solana.com",
                "https://solana-api.projectserum.com"
            ],
            native_token="SOL",
            explorer_url="https://explorer.solana.com",
            is_evm=False
        )
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize chain manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.connections: Dict[Chain, Any] = {}
        self.active_rpcs: Dict[Chain, str] = {}
        
        logger.info("ChainManager initialized")
    
    def get_chain_config(self, chain: Chain) -> ChainConfig:
        """
        Get configuration for a specific chain.
        
        Args:
            chain: Blockchain network
            
        Returns:
            Chain configuration
        """
        return self.CHAIN_CONFIGS[chain]
    
    def connect(self, chain: Chain, rpc_url: Optional[str] = None) -> bool:
        """
        Connect to a blockchain network.
        
        Args:
            chain: Blockchain to connect to
            rpc_url: Optional specific RPC URL to use
            
        Returns:
            True if connection successful
        """
        try:
            config = self.get_chain_config(chain)
            
            # Try provided RPC or fallback to configured ones
            rpcs_to_try = [rpc_url] if rpc_url else config.rpc_urls
            
            for rpc in rpcs_to_try:
                if rpc is None:
                    continue
                    
                try:
                    # Import chain-specific interface
                    if chain.value in ['ethereum', 'polygon', 'arbitrum', 'optimism', 'base']:
                        from .ethereum_interface import EthereumInterface
                        interface = EthereumInterface(rpc_url=rpc, chain_id=config.chain_id)
                    elif chain == Chain.BSC:
                        from .bsc_interface import BSCInterface
                        interface = BSCInterface(testnet=False)
                    elif chain == Chain.AVALANCHE:
                        from .avalanche_interface import AvalancheInterface
                        interface = AvalancheInterface(rpc_url=rpc)
                    elif chain == Chain.SOLANA:
                        from .solana_interface import SolanaInterface
                        interface = SolanaInterface(rpc_url=rpc)
                    else:
                        logger.warning(f"Chain {chain.value} not yet implemented")
                        continue
                    
                    # Test connection
                    if hasattr(interface, 'w3') and interface.w3.is_connected():
                        self.connections[chain] = interface
                        self.active_rpcs[chain] = rpc
                        logger.info(f"Connected to {config.name} via {rpc}")
                        return True
                    elif hasattr(interface, 'client'):  # Solana
                        self.connections[chain] = interface
                        self.active_rpcs[chain] = rpc
                        logger.info(f"Connected to {config.name} via {rpc}")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Failed to connect to {config.name} via {rpc}: {e}")
                    continue
            
            logger.error(f"Failed to connect to {config.name} with any RPC")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to {chain.value}: {e}")
            return False
    
    def get_connection(self, chain: Chain) -> Optional[Any]:
        """
        Get active connection for a chain.
        
        Args:
            chain: Blockchain network
            
        Returns:
            Chain interface instance or None
        """
        if chain not in self.connections:
            logger.warning(f"No connection for {chain.value}, attempting to connect...")
            self.connect(chain)
        
        return self.connections.get(chain)
    
    def disconnect(self, chain: Chain) -> None:
        """
        Disconnect from a blockchain.
        
        Args:
            chain: Blockchain to disconnect from
        """
        if chain in self.connections:
            del self.connections[chain]
            del self.active_rpcs[chain]
            logger.info(f"Disconnected from {chain.value}")
    
    def get_balance(self, chain: Chain, address: str, token_address: Optional[str] = None) -> float:
        """
        Get balance on any chain.
        
        Args:
            chain: Blockchain network
            address: Wallet address
            token_address: Optional token contract address
            
        Returns:
            Balance amount
        """
        connection = self.get_connection(chain)
        if not connection:
            raise ValueError(f"No connection to {chain.value}")
        
        return connection.get_balance(address, token_address)
    
    def get_all_balances(self, address: str) -> Dict[str, float]:
        """
        Get native token balances across all connected chains.
        
        Args:
            address: Wallet address
            
        Returns:
            Dictionary of chain -> balance
        """
        balances = {}
        
        for chain in self.connections.keys():
            try:
                balance = self.get_balance(chain, address)
                config = self.get_chain_config(chain)
                balances[f"{config.name} ({config.native_token})"] = balance
            except Exception as e:
                logger.warning(f"Error getting balance on {chain.value}: {e}")
                balances[chain.value] = 0.0
        
        return balances
    
    def is_connected(self, chain: Chain) -> bool:
        """
        Check if connected to a chain.
        
        Args:
            chain: Blockchain network
            
        Returns:
            True if connected
        """
        return chain in self.connections
    
    def list_connected_chains(self) -> List[str]:
        """
        Get list of connected chains.
        
        Returns:
            List of connected chain names
        """
        return [chain.value for chain in self.connections.keys()]
    
    def reconnect_all(self) -> Dict[str, bool]:
        """
        Attempt to reconnect to all chains.
        
        Returns:
            Dictionary of chain -> connection status
        """
        results = {}
        
        for chain in Chain:
            try:
                results[chain.value] = self.connect(chain)
            except Exception as e:
                logger.error(f"Error reconnecting to {chain.value}: {e}")
                results[chain.value] = False
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize chain manager
    manager = ChainManager()
    
    print("=== Multi-Chain Manager Test ===")
    
    # Connect to multiple chains
    chains_to_test = [Chain.ETHEREUM, Chain.POLYGON, Chain.BSC, Chain.ARBITRUM]
    
    for chain in chains_to_test:
        success = manager.connect(chain)
        print(f"{chain.value}: {'✅ Connected' if success else '❌ Failed'}")
    
    # List connected chains
    print(f"\nConnected chains: {manager.list_connected_chains()}")
    
    print("\n✅ ChainManager test completed!")
