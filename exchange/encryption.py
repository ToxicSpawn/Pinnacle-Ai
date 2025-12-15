"""
API Key Encryption/Decryption Utilities
Securely store and retrieve encrypted API keys
"""
from __future__ import annotations

import os
import logging
from typing import Optional
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class APIKeyEncryption:
    """
    Utility class for encrypting and decrypting API keys.
    
    Uses Fernet symmetric encryption from the cryptography library.
    """
    
    def __init__(self, key_file: str = ".encryption_key"):
        """
        Initialize encryption manager.
        
        Args:
            key_file: Path to file storing the encryption key
        """
        self.key_file = key_file
        self.key: Optional[bytes] = None
        self.cipher_suite: Optional[Fernet] = None
        self._load_or_generate_key()
    
    def _load_or_generate_key(self) -> None:
        """Load existing encryption key or generate a new one."""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'rb') as f:
                    self.key = f.read()
                self.cipher_suite = Fernet(self.key)
                logger.info("✅ Loaded existing encryption key")
            except Exception as e:
                logger.error(f"Error loading encryption key: {e}")
                self._generate_key()
        else:
            self._generate_key()
    
    def _generate_key(self) -> None:
        """Generate a new encryption key."""
        try:
            self.key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.key)
            
            # Save key to file (user should secure this!)
            with open(self.key_file, 'wb') as f:
                f.write(self.key)
            
            logger.warning(f"⚠️  Generated new encryption key. Store {self.key_file} securely!")
            logger.warning("⚠️  DO NOT commit this file to version control!")
        except Exception as e:
            logger.error(f"Error generating encryption key: {e}")
            raise
    
    def encrypt(self, plaintext: str) -> Optional[str]:
        """
        Encrypt a plaintext string.
        
        Args:
            plaintext: String to encrypt
            
        Returns:
            Encrypted string (base64 encoded) or None on error
        """
        if not self.cipher_suite:
            logger.error("Encryption not initialized")
            return None
        
        try:
            encrypted_bytes = self.cipher_suite.encrypt(plaintext.encode())
            return encrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Error encrypting: {e}")
            return None
    
    def decrypt(self, ciphertext: str) -> Optional[str]:
        """
        Decrypt an encrypted string.
        
        Args:
            ciphertext: Encrypted string (base64 encoded)
            
        Returns:
            Decrypted plaintext string or None on error
        """
        if not self.cipher_suite:
            logger.error("Decryption not initialized")
            return None
        
        try:
            decrypted_bytes = self.cipher_suite.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Error decrypting: {e}")
            return None
    
    def encrypt_api_key(self, api_key: str) -> Optional[str]:
        """Convenience method to encrypt an API key."""
        return self.encrypt(api_key)
    
    def decrypt_api_key(self, encrypted_key: str) -> Optional[str]:
        """Convenience method to decrypt an API key."""
        return self.decrypt(encrypted_key)


def encrypt_config_values(config_path: str = "config/exchanges.yaml") -> None:
    """
    Utility function to encrypt API keys in config file.
    
    This is a helper script that can be run to encrypt existing API keys.
    """
    import yaml
    
    encryption = APIKeyEncryption()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exchanges = config.get('exchanges', {})
        modified = False
        
        for exchange_name, exchange_config in exchanges.items():
            if exchange_config.get('encrypted', False):
                continue  # Already encrypted
            
            api_key = exchange_config.get('api_key')
            secret = exchange_config.get('secret')
            
            if api_key and not api_key.startswith('${'):  # Not an env var
                encrypted_key = encryption.encrypt(api_key)
                if encrypted_key:
                    exchange_config['api_key'] = encrypted_key
                    exchange_config['encrypted'] = True
                    modified = True
                    logger.info(f"Encrypted API key for {exchange_name}")
            
            if secret and not secret.startswith('${'):  # Not an env var
                encrypted_secret = encryption.encrypt(secret)
                if encrypted_secret:
                    exchange_config['secret'] = encrypted_secret
                    exchange_config['encrypted'] = True
                    modified = True
                    logger.info(f"Encrypted secret for {exchange_name}")
        
        if modified:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"✅ Updated {config_path} with encrypted values")
        else:
            logger.info("No values to encrypt")
    
    except Exception as e:
        logger.error(f"Error encrypting config: {e}")

