"""
Wallet Manager - Secure multi-chain wallet management

Responsibilities:
1. Encrypted private key storage (Fernet encryption)
2. Multi-chain signing (EVM and Solana)
3. Gas reserves per chain (never trade reserved gas funds)
4. Balance tracking across chains
5. Transaction signing and submission

Security:
- Private keys encrypted at rest
- Master password required
- Keys never logged or exposed
- Separate wallets per chain for isolation
"""

import os
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported chains"""
    # EVM chains
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    ETHEREUM = "ethereum"
    # Non-EVM
    SOLANA = "solana"


@dataclass
class WalletInfo:
    """Wallet information"""
    chain: Chain
    address: str
    balance_native: float  # ETH, SOL, etc.
    balance_usd: float
    gas_reserve: float  # Amount reserved for gas
    available: float  # balance_native - gas_reserve


@dataclass
class Transaction:
    """Transaction data"""
    chain: Chain
    from_address: str
    to_address: str
    value: float
    data: Optional[str] = None
    gas_limit: Optional[int] = None
    gas_price: Optional[int] = None
    nonce: Optional[int] = None


class WalletManager:
    """
    Secure multi-chain wallet manager.

    Features:
    - Encrypted key storage with Fernet
    - Multi-chain support (EVM + Solana)
    - Gas reserve management
    - Balance tracking
    - Transaction signing

    Security:
    - Keys encrypted with master password
    - PBKDF2 key derivation (100k iterations)
    - Keys stored in encrypted file
    - Never logs private keys
    """

    # Gas reserves per chain (in native tokens)
    GAS_RESERVES = {
        Chain.BASE: 0.01,      # 0.01 ETH reserved
        Chain.ARBITRUM: 0.01,
        Chain.OPTIMISM: 0.01,
        Chain.POLYGON: 5.0,    # 5 MATIC reserved
        Chain.ETHEREUM: 0.05,  # 0.05 ETH (mainnet is expensive)
        Chain.SOLANA: 0.1,     # 0.1 SOL reserved
    }

    def __init__(
        self,
        wallet_file: str = "~/.trading-ai/wallets.enc",
        master_password: Optional[str] = None
    ):
        """
        Initialize wallet manager.

        Args:
            wallet_file: Path to encrypted wallet file
            master_password: Master password for encryption
        """
        self.wallet_file = os.path.expanduser(wallet_file)
        self.master_password = master_password or os.getenv("WALLET_MASTER_PASSWORD")

        if not self.master_password:
            logger.warning("⚠️  No master password - wallets will not be encrypted!")
            logger.warning("   Set WALLET_MASTER_PASSWORD environment variable")

        # Storage for decrypted keys (in-memory only)
        self._keys: Dict[Chain, str] = {}
        self._addresses: Dict[Chain, str] = {}

        # Ensure wallet directory exists
        os.makedirs(os.path.dirname(self.wallet_file), exist_ok=True)

        logger.info(f"Wallet Manager initialized | File: {self.wallet_file}")

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Master password
            salt: Salt for key derivation

        Returns:
            Derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _get_cipher(self, salt: Optional[bytes] = None) -> tuple:
        """
        Get Fernet cipher for encryption/decryption.

        Returns:
            (cipher, salt)
        """
        if not self.master_password:
            raise ValueError("Master password required for encryption")

        if salt is None:
            salt = os.urandom(16)

        key = self._derive_key(self.master_password, salt)
        cipher = Fernet(key)
        return cipher, salt

    def generate_wallet(self, chain: Chain) -> str:
        """
        Generate new wallet for chain.

        Args:
            chain: Chain to generate wallet for

        Returns:
            Wallet address
        """
        if chain == Chain.SOLANA:
            # Solana wallet generation
            try:
                from solders.keypair import Keypair
                keypair = Keypair()
                private_key = bytes(keypair).hex()
                address = str(keypair.pubkey())

                self._keys[chain] = private_key
                self._addresses[chain] = address

                logger.info(f"✅ Generated Solana wallet: {address}")
                return address

            except ImportError:
                logger.error("solders library not installed - cannot generate Solana wallet")
                logger.error("Install with: pip install solders")
                raise

        else:
            # EVM wallet generation
            try:
                from eth_account import Account
                Account.enable_unaudited_hdwallet_features()
                account = Account.create()

                private_key = account.key.hex()
                address = account.address

                self._keys[chain] = private_key
                self._addresses[chain] = address

                logger.info(f"✅ Generated {chain.value} wallet: {address}")
                return address

            except ImportError:
                logger.error("eth_account library not installed - cannot generate EVM wallet")
                logger.error("Install with: pip install eth-account")
                raise

    def import_wallet(self, chain: Chain, private_key: str) -> str:
        """
        Import existing wallet from private key.

        Args:
            chain: Chain
            private_key: Private key (hex string)

        Returns:
            Wallet address
        """
        if chain == Chain.SOLANA:
            try:
                from solders.keypair import Keypair
                keypair = Keypair.from_bytes(bytes.fromhex(private_key))
                address = str(keypair.pubkey())

                self._keys[chain] = private_key
                self._addresses[chain] = address

                logger.info(f"✅ Imported Solana wallet: {address}")
                return address

            except Exception as e:
                logger.error(f"Failed to import Solana wallet: {e}")
                raise

        else:
            try:
                from eth_account import Account
                account = Account.from_key(private_key)
                address = account.address

                self._keys[chain] = private_key
                self._addresses[chain] = address

                logger.info(f"✅ Imported {chain.value} wallet: {address}")
                return address

            except Exception as e:
                logger.error(f"Failed to import EVM wallet: {e}")
                raise

    def save_wallets(self) -> bool:
        """
        Save wallets to encrypted file.

        Returns:
            Success status
        """
        if not self.master_password:
            logger.warning("Cannot save wallets without master password")
            return False

        try:
            # Prepare wallet data
            wallet_data = {
                chain.value: {
                    "private_key": key,
                    "address": self._addresses.get(chain, "")
                }
                for chain, key in self._keys.items()
            }

            # Encrypt
            cipher, salt = self._get_cipher()
            encrypted = cipher.encrypt(json.dumps(wallet_data).encode())

            # Save to file
            with open(self.wallet_file, 'wb') as f:
                f.write(salt)  # First 16 bytes are salt
                f.write(encrypted)

            logger.info(f"✅ Wallets saved to {self.wallet_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save wallets: {e}")
            return False

    def load_wallets(self) -> bool:
        """
        Load wallets from encrypted file.

        Returns:
            Success status
        """
        if not os.path.exists(self.wallet_file):
            logger.warning(f"Wallet file not found: {self.wallet_file}")
            return False

        if not self.master_password:
            logger.error("Master password required to load wallets")
            return False

        try:
            # Read file
            with open(self.wallet_file, 'rb') as f:
                salt = f.read(16)
                encrypted = f.read()

            # Decrypt
            cipher, _ = self._get_cipher(salt)
            decrypted = cipher.decrypt(encrypted)
            wallet_data = json.loads(decrypted)

            # Load wallets
            for chain_name, data in wallet_data.items():
                chain = Chain(chain_name)
                self._keys[chain] = data["private_key"]
                self._addresses[chain] = data["address"]

            logger.info(f"✅ Loaded {len(self._keys)} wallets from {self.wallet_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load wallets: {e}")
            return False

    def get_address(self, chain: Chain) -> Optional[str]:
        """Get wallet address for chain."""
        return self._addresses.get(chain)

    def has_wallet(self, chain: Chain) -> bool:
        """Check if wallet exists for chain."""
        return chain in self._keys

    def get_balance(self, chain: Chain) -> WalletInfo:
        """
        Get wallet balance for chain.

        Args:
            chain: Chain to check

        Returns:
            Wallet info with balance
        """
        if not self.has_wallet(chain):
            return WalletInfo(
                chain=chain,
                address="",
                balance_native=0.0,
                balance_usd=0.0,
                gas_reserve=self.GAS_RESERVES.get(chain, 0.0),
                available=0.0
            )

        address = self.get_address(chain)
        balance = self._fetch_balance(chain, address)
        gas_reserve = self.GAS_RESERVES.get(chain, 0.0)

        return WalletInfo(
            chain=chain,
            address=address,
            balance_native=balance,
            balance_usd=balance * self._get_price(chain),  # TODO: Real price
            gas_reserve=gas_reserve,
            available=max(0.0, balance - gas_reserve)
        )

    def _fetch_balance(self, chain: Chain, address: str) -> float:
        """
        Fetch balance from chain.

        Args:
            chain: Chain
            address: Wallet address

        Returns:
            Balance in native token
        """
        # TODO: Implement real balance fetching
        # For now, return mock balance
        logger.warning(f"Mock balance fetch for {chain.value}:{address}")
        return 1.0

    def _get_price(self, chain: Chain) -> float:
        """Get native token price in USD (mock for now)."""
        prices = {
            Chain.BASE: 3000.0,      # ETH
            Chain.ARBITRUM: 3000.0,
            Chain.OPTIMISM: 3000.0,
            Chain.ETHEREUM: 3000.0,
            Chain.POLYGON: 0.80,     # MATIC
            Chain.SOLANA: 120.0,     # SOL
        }
        return prices.get(chain, 1.0)

    def sign_transaction(self, chain: Chain, transaction: Transaction) -> str:
        """
        Sign transaction for chain.

        Args:
            chain: Chain
            transaction: Transaction to sign

        Returns:
            Signed transaction (hex string)
        """
        if not self.has_wallet(chain):
            raise ValueError(f"No wallet for {chain.value}")

        private_key = self._keys[chain]

        if chain == Chain.SOLANA:
            # Solana transaction signing
            logger.warning("Solana transaction signing not yet implemented")
            raise NotImplementedError("Solana signing coming in next update")

        else:
            # EVM transaction signing
            try:
                from eth_account import Account
                from web3 import Web3

                account = Account.from_key(private_key)

                # Build transaction dict
                tx = {
                    'from': transaction.from_address,
                    'to': transaction.to_address,
                    'value': Web3.to_wei(transaction.value, 'ether'),
                    'gas': transaction.gas_limit or 21000,
                    'gasPrice': transaction.gas_price or 0,
                    'nonce': transaction.nonce or 0,
                }

                if transaction.data:
                    tx['data'] = transaction.data

                # Sign transaction
                signed = account.sign_transaction(tx)
                return signed.rawTransaction.hex()

            except Exception as e:
                logger.error(f"Failed to sign transaction: {e}")
                raise

    def get_all_wallets(self) -> List[WalletInfo]:
        """Get info for all wallets."""
        return [self.get_balance(chain) for chain in self._keys.keys()]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("WALLET MANAGER TEST")
    print("="*70)

    # Initialize wallet manager
    manager = WalletManager(
        wallet_file="~/.trading-ai-test/wallets.enc",
        master_password="test_password_123"  # Never hardcode in production!
    )

    # Generate wallets
    print("\n--- Generating Wallets ---")
    try:
        base_addr = manager.generate_wallet(Chain.BASE)
        print(f"Base: {base_addr}")
    except Exception as e:
        print(f"Base wallet generation failed: {e}")

    try:
        sol_addr = manager.generate_wallet(Chain.SOLANA)
        print(f"Solana: {sol_addr}")
    except Exception as e:
        print(f"Solana wallet generation failed: {e}")

    # Save wallets
    print("\n--- Saving Wallets ---")
    if manager.save_wallets():
        print("✅ Wallets saved successfully")

    # Load wallets
    print("\n--- Loading Wallets ---")
    manager2 = WalletManager(
        wallet_file="~/.trading-ai-test/wallets.enc",
        master_password="test_password_123"
    )

    if manager2.load_wallets():
        print("✅ Wallets loaded successfully")
        print(f"Loaded {len(manager2.get_all_wallets())} wallets")

    # Get balances
    print("\n--- Wallet Balances ---")
    for wallet in manager2.get_all_wallets():
        print(f"{wallet.chain.value}:")
        print(f"  Address: {wallet.address}")
        print(f"  Balance: {wallet.balance_native:.4f}")
        print(f"  Gas Reserve: {wallet.gas_reserve}")
        print(f"  Available: {wallet.available:.4f}")

    print("\n" + "="*70)
    print("✅ Wallet Manager ready!")
    print("="*70)
