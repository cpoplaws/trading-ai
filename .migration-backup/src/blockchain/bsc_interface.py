"""
Binance Smart Chain (BSC) Web3 interface for DeFi trading.
"""
import os
import json
from web3 import Web3
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from dotenv import load_dotenv
# We'll handle wallet operations later
Account = None
from decimal import Decimal

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class BSCInterface:
    """
    Binance Smart Chain interface for Web3 interactions.
    """
    
    # BSC Network Configuration
    BSC_MAINNET_RPC = "https://bsc-dataseed1.binance.org/"
    BSC_TESTNET_RPC = "https://data-seed-prebsc-1-s1.binance.org:8545/"
    
    # Common BSC Token Addresses
    TOKENS = {
        'BNB': '0x0000000000000000000000000000000000000000',  # Native BNB
        'BUSD': '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56',
        'USDT': '0x55d398326f99059fF775485246999027B3197955', 
        'ETH': '0x2170Ed0807C2f9CE5E5E1c22Fc8e6A4Bb3d24Fe4',
        'CAKE': '0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82',
        'ADA': '0x3EE2200Efb3400fAbB9AacF31297cBdD1d435D47',
        'DOT': '0x7083609fCE4d1d8Dc0C979AAb8c869Ea2C873402',
    }
    
    def __init__(self, testnet: bool = True, private_key: Optional[str] = None):
        """
        Initialize BSC connection.
        
        Args:
            testnet: Whether to use BSC testnet (True) or mainnet (False)
            private_key: Private key for wallet operations
        """
        self.testnet = testnet
        self.rpc_url = self.BSC_TESTNET_RPC if testnet else self.BSC_MAINNET_RPC
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        # BSC uses Proof of Authority, no middleware needed for basic operations
        
        # Wallet setup (simplified for demo)
        self.private_key = private_key or os.getenv('BSC_PRIVATE_KEY')
        if self.private_key and Account:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            logger.warning("No private key provided or eth-account not available - read-only mode")
            self.account = None
            self.address = None
            
        # Verify connection
        try:
            # Test connection by getting latest block
            self.w3.eth.block_number
            connected = True
        except Exception:
            connected = False
            
        if not connected:
            raise ConnectionError(f"Failed to connect to BSC {'testnet' if testnet else 'mainnet'}")
            
        logger.info(f"Connected to BSC {'testnet' if testnet else 'mainnet'}")
        if self.address:
            logger.info(f"Wallet address: {self.address}")
    
    def get_balance(self, address: Optional[str] = None, token_address: Optional[str] = None) -> float:
        """
        Get BNB or token balance.
        
        Args:
            address: Wallet address (uses default if None)
            token_address: Token contract address (BNB if None)
            
        Returns:
            Balance in token units
        """
        addr = address or self.address
        if not addr:
            raise ValueError("No address provided and no wallet configured")
            
        try:
            if token_address is None:
                # Get BNB balance
                balance_wei = self.w3.eth.get_balance(addr)
                return balance_wei / 1e18
            else:
                # Get token balance
                contract = self._get_token_contract(token_address)
                balance = contract.functions.balanceOf(addr).call()
                decimals = contract.functions.decimals().call()
                return balance / (10 ** decimals)
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_token_info(self, token_address: str) -> Dict[str, Union[str, int]]:
        """
        Get token information (name, symbol, decimals).
        
        Args:
            token_address: Token contract address
            
        Returns:
            Dictionary with token info
        """
        try:
            contract = self._get_token_contract(token_address)
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
    
    def estimate_gas_price(self) -> int:
        """
        Estimate current gas price with buffer.
        
        Returns:
            Gas price in wei
        """
        try:
            gas_price = self.w3.eth.gas_price
            # Add 10% buffer for BSC
            return int(gas_price * 1.1)
        except Exception as e:
            # Fallback to 5 Gwei for BSC
            logger.warning(f"Error estimating gas: {e}, using fallback")
            return int(5 * 1e9)  # 5 Gwei in Wei
    
    def send_bnb(self, to_address: str, amount: float, gas_limit: int = 21000) -> Optional[str]:
        """
        Send BNB to another address.
        
        Args:
            to_address: Recipient address
            amount: Amount in BNB
            gas_limit: Gas limit for transaction
            
        Returns:
            Transaction hash if successful
        """
        if not self.account:
            raise ValueError("No wallet configured for sending transactions")
            
        try:
            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.estimate_gas_price()
            
            transaction = {
                'to': to_address,
                'value': int(amount * 1e18),
                'gas': gas_limit,
                'gasPrice': gas_price,
                'nonce': nonce,
                'chainId': 97 if self.testnet else 56  # BSC testnet: 97, mainnet: 56
            }
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"BNB transfer sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending BNB: {e}")
            return None
    
    def approve_token(self, token_address: str, spender_address: str, amount: Optional[float] = None) -> Optional[str]:
        """
        Approve token spending by another contract.
        
        Args:
            token_address: Token contract address
            spender_address: Address allowed to spend tokens
            amount: Amount to approve (None for unlimited)
            
        Returns:
            Transaction hash if successful
        """
        if not self.account:
            raise ValueError("No wallet configured for sending transactions")
            
        try:
            contract = self._get_token_contract(token_address)
            decimals = contract.functions.decimals().call()
            
            # Set approval amount
            if amount is None:
                # Unlimited approval
                approve_amount = 2**256 - 1
            else:
                approve_amount = int(amount * (10 ** decimals))
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.estimate_gas_price()
            
            transaction = contract.functions.approve(
                spender_address, approve_amount
            ).buildTransaction({
                'chainId': 97 if self.testnet else 56,
                'gas': 100000,  # Standard gas limit for approval
                'gasPrice': gas_price,
                'nonce': nonce,
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Token approval sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error approving token: {e}")
            return None
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> Optional[Dict]:
        """
        Wait for transaction confirmation.
        
        Args:
            tx_hash: Transaction hash to wait for
            timeout: Timeout in seconds
            
        Returns:
            Transaction receipt if successful
        """
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            if receipt.status == 1:
                logger.info(f"Transaction confirmed: {tx_hash}")
                return dict(receipt)
            else:
                logger.error(f"Transaction failed: {tx_hash}")
                return None
        except Exception as e:
            logger.error(f"Error waiting for transaction: {e}")
            return None
    
    def _get_token_contract(self, token_address: str):
        """
        Get token contract instance.
        
        Args:
            token_address: Token contract address
            
        Returns:
            Web3 contract instance
        """
        # Standard ERC-20 ABI (minimal)
        erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
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
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
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
        
        return self.w3.eth.contract(
            address=self.w3.toChecksumAddress(token_address),
            abi=erc20_abi
        )

# Example usage and testing
if __name__ == "__main__":
    # Test BSC connection (read-only)
    bsc = BSCInterface(testnet=True)
    
    print("=== BSC Interface Test ===")
    print(f"Connected to BSC: {bsc.w3.isConnected()}")
    print(f"Latest block: {bsc.w3.eth.block_number}")
    print(f"Chain ID: {bsc.w3.eth.chain_id}")
    
    # Test token info
    busd_info = bsc.get_token_info(bsc.TOKENS['BUSD'])
    print(f"BUSD Info: {busd_info}")
    
    print("✅ BSC Interface test completed!")