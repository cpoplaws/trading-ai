"""
Chain Configuration - Multi-chain RPC endpoints and chain metadata

Manages:
- RPC endpoints for all supported chains
- Chain metadata (chain ID, block explorer, native token)
- Testnet/mainnet configuration
- Connection pooling and failover
"""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Network(Enum):
    """Network type"""
    MAINNET = "mainnet"
    TESTNET = "testnet"


class Chain(Enum):
    """Supported chains"""
    # EVM L2s
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    # EVM L1
    ETHEREUM = "ethereum"
    # Non-EVM
    SOLANA = "solana"


@dataclass
class ChainMetadata:
    """Chain metadata and configuration"""
    chain: Chain
    chain_id: int
    name: str
    native_token: str
    native_token_decimals: int
    block_explorer: str
    rpc_endpoints: List[str]
    ws_endpoints: List[str]
    supports_eip1559: bool
    average_block_time: float  # seconds
    finality_blocks: int


class ChainConfig:
    """
    Chain configuration manager.

    Features:
    - RPC endpoint management with failover
    - Testnet/mainnet switching
    - Environment variable configuration
    - Chain metadata access
    """

    # Mainnet configurations
    MAINNET_CONFIGS = {
        Chain.BASE: ChainMetadata(
            chain=Chain.BASE,
            chain_id=8453,
            name="Base",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://basescan.org",
            rpc_endpoints=[
                "https://mainnet.base.org",
                "https://base.llamarpc.com",
            ],
            ws_endpoints=[
                "wss://base.llamarpc.com",
            ],
            supports_eip1559=True,
            average_block_time=2.0,
            finality_blocks=64
        ),

        Chain.ARBITRUM: ChainMetadata(
            chain=Chain.ARBITRUM,
            chain_id=42161,
            name="Arbitrum One",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://arbiscan.io",
            rpc_endpoints=[
                "https://arb1.arbitrum.io/rpc",
                "https://arbitrum.llamarpc.com",
            ],
            ws_endpoints=[
                "wss://arbitrum.llamarpc.com",
            ],
            supports_eip1559=True,
            average_block_time=0.25,
            finality_blocks=256
        ),

        Chain.OPTIMISM: ChainMetadata(
            chain=Chain.OPTIMISM,
            chain_id=10,
            name="Optimism",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://optimistic.etherscan.io",
            rpc_endpoints=[
                "https://mainnet.optimism.io",
                "https://optimism.llamarpc.com",
            ],
            ws_endpoints=[
                "wss://optimism.llamarpc.com",
            ],
            supports_eip1559=True,
            average_block_time=2.0,
            finality_blocks=64
        ),

        Chain.POLYGON: ChainMetadata(
            chain=Chain.POLYGON,
            chain_id=137,
            name="Polygon",
            native_token="MATIC",
            native_token_decimals=18,
            block_explorer="https://polygonscan.com",
            rpc_endpoints=[
                "https://polygon-rpc.com",
                "https://polygon.llamarpc.com",
            ],
            ws_endpoints=[
                "wss://polygon.llamarpc.com",
            ],
            supports_eip1559=True,
            average_block_time=2.0,
            finality_blocks=128
        ),

        Chain.ETHEREUM: ChainMetadata(
            chain=Chain.ETHEREUM,
            chain_id=1,
            name="Ethereum",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://etherscan.io",
            rpc_endpoints=[
                "https://eth.llamarpc.com",
            ],
            ws_endpoints=[
                "wss://eth.llamarpc.com",
            ],
            supports_eip1559=True,
            average_block_time=12.0,
            finality_blocks=32
        ),

        Chain.SOLANA: ChainMetadata(
            chain=Chain.SOLANA,
            chain_id=0,  # Solana doesn't use chain ID
            name="Solana",
            native_token="SOL",
            native_token_decimals=9,
            block_explorer="https://explorer.solana.com",
            rpc_endpoints=[
                "https://api.mainnet-beta.solana.com",
            ],
            ws_endpoints=[
                "wss://api.mainnet-beta.solana.com",
            ],
            supports_eip1559=False,
            average_block_time=0.4,
            finality_blocks=31
        ),
    }

    # Testnet configurations
    TESTNET_CONFIGS = {
        Chain.BASE: ChainMetadata(
            chain=Chain.BASE,
            chain_id=84532,
            name="Base Sepolia",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://sepolia.basescan.org",
            rpc_endpoints=[
                "https://sepolia.base.org",
            ],
            ws_endpoints=[],
            supports_eip1559=True,
            average_block_time=2.0,
            finality_blocks=64
        ),

        Chain.ARBITRUM: ChainMetadata(
            chain=Chain.ARBITRUM,
            chain_id=421614,
            name="Arbitrum Sepolia",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://sepolia.arbiscan.io",
            rpc_endpoints=[
                "https://sepolia-rollup.arbitrum.io/rpc",
            ],
            ws_endpoints=[],
            supports_eip1559=True,
            average_block_time=0.25,
            finality_blocks=256
        ),

        Chain.OPTIMISM: ChainMetadata(
            chain=Chain.OPTIMISM,
            chain_id=11155420,
            name="Optimism Sepolia",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://sepolia-optimism.etherscan.io",
            rpc_endpoints=[
                "https://sepolia.optimism.io",
            ],
            ws_endpoints=[],
            supports_eip1559=True,
            average_block_time=2.0,
            finality_blocks=64
        ),

        Chain.POLYGON: ChainMetadata(
            chain=Chain.POLYGON,
            chain_id=80002,
            name="Polygon Amoy",
            native_token="MATIC",
            native_token_decimals=18,
            block_explorer="https://amoy.polygonscan.com",
            rpc_endpoints=[
                "https://rpc-amoy.polygon.technology",
            ],
            ws_endpoints=[],
            supports_eip1559=True,
            average_block_time=2.0,
            finality_blocks=128
        ),

        Chain.ETHEREUM: ChainMetadata(
            chain=Chain.ETHEREUM,
            chain_id=11155111,
            name="Ethereum Sepolia",
            native_token="ETH",
            native_token_decimals=18,
            block_explorer="https://sepolia.etherscan.io",
            rpc_endpoints=[
                "https://rpc.sepolia.org",
            ],
            ws_endpoints=[],
            supports_eip1559=True,
            average_block_time=12.0,
            finality_blocks=32
        ),

        Chain.SOLANA: ChainMetadata(
            chain=Chain.SOLANA,
            chain_id=0,
            name="Solana Devnet",
            native_token="SOL",
            native_token_decimals=9,
            block_explorer="https://explorer.solana.com?cluster=devnet",
            rpc_endpoints=[
                "https://api.devnet.solana.com",
            ],
            ws_endpoints=[
                "wss://api.devnet.solana.com",
            ],
            supports_eip1559=False,
            average_block_time=0.4,
            finality_blocks=31
        ),
    }

    def __init__(self, network: Network = Network.TESTNET, use_custom_rpcs: bool = True):
        """
        Initialize chain configuration.

        Args:
            network: Network type (mainnet or testnet)
            use_custom_rpcs: Whether to use custom RPC endpoints from environment
        """
        self.network = network
        self.use_custom_rpcs = use_custom_rpcs

        # Select config based on network
        self.configs = (
            self.MAINNET_CONFIGS if network == Network.MAINNET
            else self.TESTNET_CONFIGS
        )

        # Override with custom RPCs from environment if enabled
        if use_custom_rpcs:
            self._load_custom_rpcs()

        logger.info(f"Chain config initialized for {network.value}")
        logger.info(f"Supported chains: {[c.value for c in self.configs.keys()]}")

    def _load_custom_rpcs(self) -> None:
        """Load custom RPC endpoints from environment variables."""
        for chain in self.configs.keys():
            env_var = f"{chain.value.upper()}_RPC_URL"
            custom_rpc = os.getenv(env_var)

            if custom_rpc:
                # Prepend custom RPC to the list
                self.configs[chain].rpc_endpoints.insert(0, custom_rpc)
                logger.info(f"Using custom RPC for {chain.value}: {custom_rpc}")

    def get_metadata(self, chain: Chain) -> Optional[ChainMetadata]:
        """Get metadata for a chain."""
        return self.configs.get(chain)

    def get_rpc_url(self, chain: Chain, index: int = 0) -> Optional[str]:
        """
        Get RPC URL for a chain.

        Args:
            chain: Chain
            index: Index of RPC endpoint (for failover)

        Returns:
            RPC URL or None
        """
        metadata = self.get_metadata(chain)
        if not metadata or not metadata.rpc_endpoints:
            return None

        if index >= len(metadata.rpc_endpoints):
            index = 0  # Wrap around

        return metadata.rpc_endpoints[index]

    def get_ws_url(self, chain: Chain, index: int = 0) -> Optional[str]:
        """Get WebSocket URL for a chain."""
        metadata = self.get_metadata(chain)
        if not metadata or not metadata.ws_endpoints:
            return None

        if index >= len(metadata.ws_endpoints):
            return None

        return metadata.ws_endpoints[index]

    def get_explorer_url(self, chain: Chain, tx_hash: str = None, address: str = None) -> str:
        """
        Get block explorer URL.

        Args:
            chain: Chain
            tx_hash: Optional transaction hash
            address: Optional address

        Returns:
            Explorer URL
        """
        metadata = self.get_metadata(chain)
        if not metadata:
            return ""

        base_url = metadata.block_explorer

        if tx_hash:
            return f"{base_url}/tx/{tx_hash}"
        elif address:
            return f"{base_url}/address/{address}"
        else:
            return base_url

    def is_evm_chain(self, chain: Chain) -> bool:
        """Check if chain is EVM-compatible."""
        return chain != Chain.SOLANA

    def get_chain_by_id(self, chain_id: int) -> Optional[Chain]:
        """Get chain by chain ID."""
        for chain, metadata in self.configs.items():
            if metadata.chain_id == chain_id:
                return chain
        return None

    def get_all_evm_chains(self) -> List[Chain]:
        """Get all EVM chains."""
        return [chain for chain in self.configs.keys() if self.is_evm_chain(chain)]

    def get_all_chains(self) -> List[Chain]:
        """Get all supported chains."""
        return list(self.configs.keys())

    def get_summary(self) -> Dict:
        """Get configuration summary."""
        return {
            "network": self.network.value,
            "total_chains": len(self.configs),
            "evm_chains": len(self.get_all_evm_chains()),
            "chains": {
                chain.value: {
                    "name": metadata.name,
                    "chain_id": metadata.chain_id,
                    "native_token": metadata.native_token,
                    "rpc_count": len(metadata.rpc_endpoints),
                    "avg_block_time": metadata.average_block_time,
                }
                for chain, metadata in self.configs.items()
            }
        }


# Singleton instance
_config_instance: Optional[ChainConfig] = None


def get_chain_config(
    network: Network = None,
    use_custom_rpcs: bool = True,
    force_new: bool = False
) -> ChainConfig:
    """
    Get chain configuration singleton.

    Args:
        network: Network type (defaults to env or testnet)
        use_custom_rpcs: Whether to use custom RPCs
        force_new: Force create new instance

    Returns:
        ChainConfig instance
    """
    global _config_instance

    if _config_instance is None or force_new:
        # Determine network from environment if not specified
        if network is None:
            env_network = os.getenv("NETWORK", "testnet").lower()
            network = Network.MAINNET if env_network == "mainnet" else Network.TESTNET

        _config_instance = ChainConfig(network, use_custom_rpcs)

    return _config_instance


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("CHAIN CONFIGURATION TEST")
    print("="*70)

    # Test testnet configuration
    print("\n--- Testnet Configuration ---")
    config = get_chain_config(Network.TESTNET)

    summary = config.get_summary()
    print(f"Network: {summary['network']}")
    print(f"Total chains: {summary['total_chains']}")
    print(f"EVM chains: {summary['evm_chains']}")

    print("\n--- Chain Details ---")
    for chain_name, details in summary['chains'].items():
        print(f"\n{details['name']}:")
        print(f"  Chain ID: {details['chain_id']}")
        print(f"  Native Token: {details['native_token']}")
        print(f"  RPC Endpoints: {details['rpc_count']}")
        print(f"  Block Time: {details['avg_block_time']}s")

    # Test RPC URLs
    print("\n--- RPC URLs ---")
    for chain in [Chain.BASE, Chain.ARBITRUM, Chain.OPTIMISM]:
        rpc_url = config.get_rpc_url(chain)
        print(f"{chain.value}: {rpc_url}")

    # Test explorer URLs
    print("\n--- Block Explorers ---")
    mock_tx = "0x1234567890abcdef"
    for chain in [Chain.BASE, Chain.ARBITRUM]:
        explorer_url = config.get_explorer_url(chain, tx_hash=mock_tx)
        print(f"{chain.value}: {explorer_url}")

    print("\n" + "="*70)
    print("✅ Chain configuration ready!")
    print("="*70)
