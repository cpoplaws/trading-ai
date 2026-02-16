"""
Secure settings and API key management for the Trading-AI system.
Handles reading/writing encrypted credentials.
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import logging

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manage application settings and API keys securely."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize settings manager."""
        if config_dir is None:
            config_dir = Path.home() / '.trading-ai'

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.env_file = self.config_dir / '.env'
        self.settings_file = self.config_dir / 'settings.json'
        self.encrypted_file = self.config_dir / 'credentials.enc'

        # Generate or load encryption key
        self.key_file = self.config_dir / '.key'
        self._init_encryption()

    def _init_encryption(self):
        """Initialize encryption key."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                self.cipher_key = f.read()
        else:
            # Generate new key from machine-specific salt
            salt = os.urandom(16)
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(b"trading-ai-secret"))
            self.cipher_key = key

            with open(self.key_file, 'wb') as f:
                f.write(self.cipher_key)

            # Secure the key file
            os.chmod(self.key_file, 0o600)

        self.cipher = Fernet(self.cipher_key)

    def save_api_key(self, service: str, key_name: str, key_value: str) -> bool:
        """
        Save an API key securely.

        Args:
            service: Service name (e.g., 'alpaca', 'binance')
            key_name: Key identifier (e.g., 'api_key', 'secret_key')
            key_value: The actual key value

        Returns:
            bool: Success status
        """
        try:
            # Load existing credentials
            credentials = self._load_encrypted_credentials()

            # Update credentials
            if service not in credentials:
                credentials[service] = {}

            credentials[service][key_name] = key_value

            # Save encrypted
            self._save_encrypted_credentials(credentials)

            # Also update .env file for compatibility
            self._update_env_file(service, key_name, key_value)

            logger.info(f"Saved {service}.{key_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving API key: {e}")
            return False

    def get_api_key(self, service: str, key_name: str) -> Optional[str]:
        """
        Retrieve an API key.

        Args:
            service: Service name
            key_name: Key identifier

        Returns:
            str or None: The key value if found
        """
        try:
            credentials = self._load_encrypted_credentials()
            return credentials.get(service, {}).get(key_name)
        except Exception as e:
            logger.error(f"Error retrieving API key: {e}")
            return None

    def get_all_keys(self, service: str) -> Dict[str, str]:
        """Get all keys for a service."""
        try:
            credentials = self._load_encrypted_credentials()
            return credentials.get(service, {})
        except Exception as e:
            logger.error(f"Error retrieving keys for {service}: {e}")
            return {}

    def delete_api_key(self, service: str, key_name: str) -> bool:
        """Delete an API key."""
        try:
            credentials = self._load_encrypted_credentials()

            if service in credentials and key_name in credentials[service]:
                del credentials[service][key_name]

                if not credentials[service]:
                    del credentials[service]

                self._save_encrypted_credentials(credentials)
                return True

            return False
        except Exception as e:
            logger.error(f"Error deleting API key: {e}")
            return False

    def _load_encrypted_credentials(self) -> Dict:
        """Load and decrypt credentials."""
        if not self.encrypted_file.exists():
            return {}

        try:
            with open(self.encrypted_file, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Error loading encrypted credentials: {e}")
            return {}

    def _save_encrypted_credentials(self, credentials: Dict):
        """Encrypt and save credentials."""
        try:
            json_data = json.dumps(credentials).encode()
            encrypted_data = self.cipher.encrypt(json_data)

            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)

            # Secure the file
            os.chmod(self.encrypted_file, 0o600)

        except Exception as e:
            logger.error(f"Error saving encrypted credentials: {e}")
            raise

    def _update_env_file(self, service: str, key_name: str, key_value: str):
        """Update .env file for backward compatibility."""
        env_var_name = f"{service.upper()}_{key_name.upper()}"

        # Read existing .env
        env_content = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        var, val = line.split('=', 1)
                        env_content[var.strip()] = val.strip()

        # Update
        env_content[env_var_name] = key_value

        # Write back
        with open(self.env_file, 'w') as f:
            f.write("# Trading-AI Environment Configuration\n")
            f.write("# Auto-generated - Do not edit manually\n\n")
            for var, val in sorted(env_content.items()):
                f.write(f"{var}={val}\n")

        os.chmod(self.env_file, 0o600)

    def save_setting(self, key: str, value: any) -> bool:
        """Save a general setting."""
        try:
            settings = self._load_settings()
            settings[key] = value
            self._save_settings(settings)
            return True
        except Exception as e:
            logger.error(f"Error saving setting: {e}")
            return False

    def get_setting(self, key: str, default: any = None) -> any:
        """Get a general setting."""
        settings = self._load_settings()
        return settings.get(key, default)

    def _load_settings(self) -> Dict:
        """Load settings from JSON."""
        if not self.settings_file.exists():
            return {}

        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return {}

    def _save_settings(self, settings: Dict):
        """Save settings to JSON."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            raise

    def export_env_file(self, target_path: Path) -> bool:
        """Export current configuration to a .env file."""
        try:
            credentials = self._load_encrypted_credentials()

            with open(target_path, 'w') as f:
                f.write("# Trading-AI Environment Configuration\n")
                f.write(f"# Exported on {os.popen('date').read().strip()}\n\n")

                for service, keys in credentials.items():
                    f.write(f"# {service.upper()}\n")
                    for key_name, key_value in keys.items():
                        env_var = f"{service.upper()}_{key_name.upper()}"
                        f.write(f"{env_var}={key_value}\n")
                    f.write("\n")

            logger.info(f"Exported configuration to {target_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting env file: {e}")
            return False

    def is_configured(self, service: str) -> bool:
        """Check if a service has any keys configured."""
        credentials = self._load_encrypted_credentials()
        return service in credentials and bool(credentials[service])

    def get_status(self) -> Dict:
        """Get configuration status for all services."""
        credentials = self._load_encrypted_credentials()

        status = {}
        services = ['alpaca', 'binance', 'coingecko', 'newsapi', 'reddit', 'ethereum', 'bsc', 'polygon', 'solana']

        for service in services:
            status[service] = {
                'configured': service in credentials,
                'key_count': len(credentials.get(service, {}))
            }

        return status
