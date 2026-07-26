"""
Ethereum mainnet and L2 (Arbitrum, Optimism, Base) interface.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class EthereumInterface:
    """
    Ethereum and L2 blockchain interface supporting:
    - Ethereum mainnet
    - Arbitrum
    - Optimism  
    - Base
    """
    
    # Common token addresses (Ethereum mainnet)
    TOKENS = {
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
    }
    
    def __init__(self, rpc_url: Optional[str] = None, chain_id: int = 1, private_key: Optional[str] = None):
        """
        Initialize Ethereum interface.
        
        Args:
            rpc_url: RPC endpoint URL
            chain_id: Chain ID (1=mainnet, 42161=arbitrum, 10=optimism, 8453=base)
            private_key: Private key for transactions
        """
        self.chain_id = chain_id
        self.rpc_url = rpc_url or os.getenv('ETHEREUM_RPC_URL', 'https://cloudflare-eth.com')
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Add middleware for L2s that use PoA
        if chain_id in [10, 8453]:  # Optimism, Base
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        # Wallet setup
        self.private_key = private_key or os.getenv('ETH_PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            logger.warning("No private key provided - read-only mode")
            self.account = None
            self.address = None
        
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum (chain_id={chain_id})")
        
        logger.info(f"Connected to Ethereum chain_id={chain_id}")
        if self.address:
            logger.info(f"Wallet address: {self.address}")
    
    def get_balance(self, address: Optional[str] = None, token_address: Optional[str] = None) -> float:
        """
        Get ETH or ERC-20 token balance.
        
        Args:
            address: Wallet address (uses default if None)
            token_address: Token contract address (ETH if None)
            
        Returns:
            Balance in token units
        """
        addr = address or self.address
        if not addr:
            raise ValueError("No address provided")
        
        try:
            if token_address is None:
                # Get ETH balance
                balance_wei = self.w3.eth.get_balance(addr)
                return balance_wei / 1e18
            else:
                # Get ERC-20 balance
                contract = self._get_erc20_contract(token_address)
                balance = contract.functions.balanceOf(addr).call()
                decimals = contract.functions.decimals().call()
                return balance / (10 ** decimals)
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get ERC-20 token information.
        
        Args:
            token_address: Token contract address
            
        Returns:
            Token information dictionary
        """
        try:
            contract = self._get_erc20_contract(token_address)
            return {
                'address': token_address,
                'name': contract.functions.name().call(),
                'symbol': contract.functions.symbol().call(),
                'decimals': contract.functions.decimals().call(),
                'total_supply': contract.functions.totalSupply().call()
            }
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return {}
    
    def estimate_gas(self, transaction: Dict) -> int:
        """
        Estimate gas for a transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Estimated gas units
        """
        try:
            gas = self.w3.eth.estimate_gas(transaction)
            # Add 20% buffer
            return int(gas * 1.2)
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default")
            return 200000  # Default fallback
    
    def get_gas_price(self) -> Dict[str, int]:
        """
        Get current gas prices with priority levels.
        
        Returns:
            Dictionary with slow, standard, fast gas prices in wei
        """
        try:
            base_price = self.w3.eth.gas_price
            
            # For L2s, gas is usually stable and low
            if self.chain_id in [42161, 10, 8453]:  # Arbitrum, Optimism, Base
                return {
                    'slow': base_price,
                    'standard': int(base_price * 1.1),
                    'fast': int(base_price * 1.2)
                }
            
            # For mainnet, provide varied options
            return {
                'slow': base_price,
                'standard': int(base_price * 1.2),
                'fast': int(base_price * 1.5)
            }
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            # Fallback prices
            if self.chain_id == 1:  # Mainnet
                return {'slow': 20e9, 'standard': 30e9, 'fast': 50e9}
            else:  # L2s
                return {'slow': 0.1e9, 'standard': 0.2e9, 'fast': 0.5e9}
    
    def send_transaction(self, to: str, value: float = 0, data: str = '0x', 
                        gas_limit: Optional[int] = None, gas_price: Optional[int] = None) -> Optional[str]:
        """
        Send a transaction.
        
        Args:
            to: Recipient address
            value: Amount in ETH
            data: Transaction data (hex string)
            gas_limit: Gas limit
            gas_price: Gas price in wei
            
        Returns:
            Transaction hash if successful
        """
        if not self.account:
            raise ValueError("No wallet configured")
        
        try:
            nonce = self.w3.eth.get_transaction_count(self.address)
            
            # Build transaction
            transaction = {
                'to': Web3.to_checksum_address(to),
                'value': Web3.to_wei(value, 'ether'),
                'data': data,
                'nonce': nonce,
                'chainId': self.chain_id,
            }
            
            # Add gas settings
            if gas_price is None:
                gas_price = self.get_gas_price()['standard']
            transaction['gasPrice'] = gas_price
            
            if gas_limit is None:
                gas_limit = self.estimate_gas(transaction)
            transaction['gas'] = gas_limit
            
            # Sign and send
            signed = self.account.sign_transaction(transaction)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            logger.info(f"Transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return None
    
    def approve_token(self, token_address: str, spender: str, 
                     amount: Optional[float] = None) -> Optional[str]:
        """
        Approve ERC-20 token spending.
        
        Args:
            token_address: Token contract address
            spender: Spender address
            amount: Amount to approve (unlimited if None)
            
        Returns:
            Transaction hash if successful
        """
        if not self.account:
            raise ValueError("No wallet configured")
        
        try:
            contract = self._get_erc20_contract(token_address)
            decimals = contract.functions.decimals().call()
            
            # Set approval amount
            if amount is None:
                approve_amount = 2**256 - 1  # Unlimited
            else:
                approve_amount = int(amount * (10 ** decimals))
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.get_gas_price()['standard']
            
            transaction = contract.functions.approve(
                Web3.to_checksum_address(spender),
                approve_amount
            ).build_transaction({
                'chainId': self.chain_id,
                'gas': 100000,
                'gasPrice': gas_price,
                'nonce': nonce,
            })
            
            # Sign and send
            signed = self.account.sign_transaction(transaction)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            logger.info(f"Approval sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error approving token: {e}")
            return None
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> Optional[Dict]:
        """
        Wait for transaction confirmation.
        
        Args:
            tx_hash: Transaction hash
            timeout: Timeout in seconds
            
        Returns:
            Transaction receipt if successful
        """
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            if receipt['status'] == 1:
                logger.info(f"Transaction confirmed: {tx_hash}")
                return dict(receipt)
            else:
                logger.error(f"Transaction failed: {tx_hash}")
                return None
        except Exception as e:
            logger.error(f"Error waiting for transaction: {e}")
            return None
    
    def get_block_number(self) -> int:
        """Get current block number."""
        return self.w3.eth.block_number
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """
        Get transaction details.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction dictionary
        """
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            return dict(tx)
        except Exception as e:
            logger.error(f"Error getting transaction: {e}")
            return None
    
    def _get_erc20_contract(self, token_address: str):
        """Get ERC-20 contract instance."""
        erc20_abi = [
            {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
            {"constant": False, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
            {"constant": True, "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
        ]
        
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=erc20_abi
        )


if __name__ == "__main__":
    # Test connection (read-only)
    eth = EthereumInterface()
    
    print("=== Ethereum Interface Test ===")
    print(f"Connected: {eth.w3.is_connected()}")
    print(f"Block number: {eth.get_block_number()}")
    print(f"Chain ID: {eth.chain_id}")
    
    # Get gas prices
    gas_prices = eth.get_gas_price()
    print(f"Gas prices: {gas_prices}")
    
    print("✅ Ethereum interface test completed!")
