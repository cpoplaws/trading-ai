"""
Blockchain Client
Unified interface for interacting with multiple blockchain networks
"""
from web3 import Web3
from web3.middleware import geth_poa_middleware
from typing import Dict, List, Optional, Any
import requests
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Network(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"


@dataclass
class NetworkConfig:
    """Network configuration."""
    name: str
    rpc_url: str
    chain_id: int
    explorer_url: str
    explorer_api_key: str
    native_token: str
    is_poa: bool = False  # Proof of Authority


class BlockchainClient:
    """
    Unified blockchain client for multiple networks.

    Features:
    - Multiple network support (Ethereum, BSC, Polygon, etc.)
    - Smart contract interactions
    - Transaction queries
    - Block data retrieval
    - Account balance and nonce queries
    - Gas estimation
    """

    DEFAULT_CONFIGS = {
        Network.ETHEREUM: NetworkConfig(
            name="Ethereum",
            rpc_url="https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY",
            chain_id=1,
            explorer_url="https://api.etherscan.io/api",
            explorer_api_key="YOUR_ETHERSCAN_API_KEY",
            native_token="ETH"
        ),
        Network.BSC: NetworkConfig(
            name="Binance Smart Chain",
            rpc_url="https://bsc-dataseed.binance.org/",
            chain_id=56,
            explorer_url="https://api.bscscan.com/api",
            explorer_api_key="YOUR_BSCSCAN_API_KEY",
            native_token="BNB",
            is_poa=True
        ),
        Network.POLYGON: NetworkConfig(
            name="Polygon",
            rpc_url="https://polygon-rpc.com/",
            chain_id=137,
            explorer_url="https://api.polygonscan.com/api",
            explorer_api_key="YOUR_POLYGONSCAN_API_KEY",
            native_token="MATIC",
            is_poa=True
        ),
        Network.ARBITRUM: NetworkConfig(
            name="Arbitrum",
            rpc_url="https://arb1.arbitrum.io/rpc",
            chain_id=42161,
            explorer_url="https://api.arbiscan.io/api",
            explorer_api_key="YOUR_ARBISCAN_API_KEY",
            native_token="ETH"
        ),
    }

    def __init__(self, network: Network, config: Optional[NetworkConfig] = None):
        """
        Initialize blockchain client.

        Args:
            network: Target blockchain network
            config: Custom network configuration (optional)
        """
        self.network = network
        self.config = config or self.DEFAULT_CONFIGS.get(network)

        if not self.config:
            raise ValueError(f"No configuration for network: {network}")

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))

        # Add middleware for PoA chains
        if self.config.is_poa:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        # Verify connection
        if not self.w3.is_connected():
            logger.warning(f"Failed to connect to {self.config.name}")
        else:
            logger.info(f"Connected to {self.config.name} (Chain ID: {self.config.chain_id})")

    # ==========================================
    # Block and Transaction Queries
    # ==========================================

    def get_block(self, block_number: int = None) -> Dict:
        """
        Get block data.

        Args:
            block_number: Block number (None for latest)

        Returns:
            Block data dictionary
        """
        block = self.w3.eth.get_block(block_number or 'latest', full_transactions=True)
        return dict(block)

    def get_transaction(self, tx_hash: str) -> Dict:
        """
        Get transaction data.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction data dictionary
        """
        tx = self.w3.eth.get_transaction(tx_hash)
        return dict(tx)

    def get_transaction_receipt(self, tx_hash: str) -> Dict:
        """
        Get transaction receipt.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction receipt dictionary
        """
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        return dict(receipt)

    def get_transaction_count(self, address: str, block: str = 'latest') -> int:
        """
        Get transaction count (nonce) for address.

        Args:
            address: Wallet address
            block: Block number or 'latest'

        Returns:
            Transaction count
        """
        return self.w3.eth.get_transaction_count(
            self.w3.to_checksum_address(address),
            block
        )

    # ==========================================
    # Account Queries
    # ==========================================

    def get_balance(self, address: str, block: str = 'latest') -> float:
        """
        Get native token balance for address.

        Args:
            address: Wallet address
            block: Block number or 'latest'

        Returns:
            Balance in native token (ETH, BNB, etc.)
        """
        balance_wei = self.w3.eth.get_balance(
            self.w3.to_checksum_address(address),
            block
        )
        return float(self.w3.from_wei(balance_wei, 'ether'))

    def get_code(self, address: str) -> str:
        """
        Get contract code at address.

        Args:
            address: Contract address

        Returns:
            Contract bytecode (hex string)
        """
        code = self.w3.eth.get_code(self.w3.to_checksum_address(address))
        return code.hex()

    def is_contract(self, address: str) -> bool:
        """
        Check if address is a contract.

        Args:
            address: Address to check

        Returns:
            True if contract, False if EOA
        """
        code = self.get_code(address)
        return code != '0x' and code != '0x0'

    # ==========================================
    # Smart Contract Interactions
    # ==========================================

    def get_contract(self, address: str, abi: List[Dict]) -> Any:
        """
        Get contract instance.

        Args:
            address: Contract address
            abi: Contract ABI

        Returns:
            Web3 contract instance
        """
        return self.w3.eth.contract(
            address=self.w3.to_checksum_address(address),
            abi=abi
        )

    def call_contract_function(
        self,
        contract_address: str,
        abi: List[Dict],
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call contract view/pure function.

        Args:
            contract_address: Contract address
            abi: Contract ABI
            function_name: Function to call
            *args: Function arguments
            **kwargs: Additional parameters

        Returns:
            Function return value
        """
        contract = self.get_contract(contract_address, abi)
        func = getattr(contract.functions, function_name)
        return func(*args).call(**kwargs)

    # ==========================================
    # Gas Estimation
    # ==========================================

    def estimate_gas(
        self,
        transaction: Dict
    ) -> int:
        """
        Estimate gas for transaction.

        Args:
            transaction: Transaction parameters

        Returns:
            Estimated gas units
        """
        return self.w3.eth.estimate_gas(transaction)

    def get_gas_price(self) -> float:
        """
        Get current gas price.

        Returns:
            Gas price in Gwei
        """
        gas_price_wei = self.w3.eth.gas_price
        return float(self.w3.from_wei(gas_price_wei, 'gwei'))

    # ==========================================
    # Explorer API Integration
    # ==========================================

    def get_transactions_by_address(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = 'desc'
    ) -> List[Dict]:
        """
        Get transactions for address using explorer API.

        Args:
            address: Wallet address
            start_block: Starting block number
            end_block: Ending block number
            sort: Sort order ('asc' or 'desc')

        Returns:
            List of transactions
        """
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': sort,
            'apikey': self.config.explorer_api_key
        }

        response = requests.get(self.config.explorer_url, params=params)
        data = response.json()

        if data.get('status') == '1':
            return data.get('result', [])
        else:
            logger.error(f"Explorer API error: {data.get('message')}")
            return []

    def get_token_transfers(
        self,
        address: str,
        contract_address: Optional[str] = None
    ) -> List[Dict]:
        """
        Get ERC-20 token transfers for address.

        Args:
            address: Wallet address
            contract_address: Specific token contract (optional)

        Returns:
            List of token transfers
        """
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'sort': 'desc',
            'apikey': self.config.explorer_api_key
        }

        if contract_address:
            params['contractaddress'] = contract_address

        response = requests.get(self.config.explorer_url, params=params)
        data = response.json()

        if data.get('status') == '1':
            return data.get('result', [])
        else:
            logger.error(f"Explorer API error: {data.get('message')}")
            return []

    def get_contract_abi(self, contract_address: str) -> List[Dict]:
        """
        Get contract ABI from explorer.

        Args:
            contract_address: Contract address

        Returns:
            Contract ABI
        """
        params = {
            'module': 'contract',
            'action': 'getabi',
            'address': contract_address,
            'apikey': self.config.explorer_api_key
        }

        response = requests.get(self.config.explorer_url, params=params)
        data = response.json()

        if data.get('status') == '1':
            import json
            return json.loads(data.get('result', '[]'))
        else:
            logger.error(f"Failed to get ABI: {data.get('message')}")
            return []

    # ==========================================
    # Utility Methods
    # ==========================================

    def to_checksum_address(self, address: str) -> str:
        """Convert address to checksum format."""
        return self.w3.to_checksum_address(address)

    def is_address(self, address: str) -> bool:
        """Check if string is valid address."""
        return self.w3.is_address(address)

    def to_wei(self, amount: float, unit: str = 'ether') -> int:
        """Convert amount to wei."""
        return self.w3.to_wei(amount, unit)

    def from_wei(self, amount: int, unit: str = 'ether') -> float:
        """Convert wei to amount."""
        return float(self.w3.from_wei(amount, unit))

    def keccak(self, text: str) -> str:
        """Calculate Keccak-256 hash."""
        return self.w3.keccak(text=text).hex()
