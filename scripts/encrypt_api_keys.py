#!/usr/bin/env python3
"""
Utility script to encrypt API keys in exchange configuration file.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exchange.encryption import encrypt_config_values, APIKeyEncryption
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to encrypt API keys."""
    config_path = "config/exchanges.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create config/exchanges.yaml first")
        return 1
    
    print("🔐 API Key Encryption Utility")
    print("=" * 50)
    print(f"\nThis will encrypt API keys in: {config_path}")
    print("⚠️  Make sure you have a backup of your API keys!")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return 0
    
    try:
        encrypt_config_values(config_path)
        print("\n✅ Encryption complete!")
        print("\n⚠️  IMPORTANT:")
        print("   1. Store .encryption_key file securely (DO NOT commit to git)")
        print("   2. Keep a backup of your original API keys")
        print("   3. Test the configuration before using in production")
        return 0
    except Exception as e:
        logger.error(f"Error during encryption: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

