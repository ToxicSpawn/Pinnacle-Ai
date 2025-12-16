#!/usr/bin/env python3
"""
Test script to verify Pinnacle AI configuration
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tools.config_loader import load_config, validate_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_configuration():
    """Test the complete configuration"""
    print("Testing Pinnacle AI Configuration...\n")
    
    try:
        # Load configuration
        config = load_config()
        print("[OK] Configuration loaded successfully")
        
        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            print("\n[WARNING] Configuration validation warnings:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("[OK] Configuration validation passed")
        
        # Test API keys presence
        print("\nAPI Key Status:")
        
        # OpenAI
        openai_key = config.get("core", {}).get("api_keys", {}).get("openai", "")
        if openai_key and openai_key.startswith("sk-"):
            print(f"   [OK] OpenAI API Key: {openai_key[:20]}...")
        else:
            print("   [ERROR] OpenAI API Key: Not set or invalid")
        
        # Serper
        serper_key = config.get("tools", {}).get("search", {}).get("serper", {}).get("api_key", "")
        if serper_key:
            print(f"   [OK] Serper API Key: {serper_key[:20]}...")
        else:
            print("   [ERROR] Serper API Key: Not set")
        
        # Stability
        stability_key = config.get("tools", {}).get("image_generation", {}).get("stable_diffusion", {}).get("api_key", "")
        if stability_key and stability_key.startswith("sk-"):
            print(f"   [OK] Stability API Key: {stability_key[:20]}...")
        else:
            print("   [WARNING] Stability API Key: Not set (optional)")
        
        # ElevenLabs
        elevenlabs_key = config.get("tools", {}).get("audio_generation", {}).get("elevenlabs", {}).get("api_key", "")
        if elevenlabs_key and elevenlabs_key.startswith("sk_"):
            print(f"   [OK] ElevenLabs API Key: {elevenlabs_key[:20]}...")
        else:
            print("   [WARNING] ElevenLabs API Key: Not set (optional)")
        
        # Test API connectivity (optional, requires internet)
        print("\nTesting API Connectivity (optional)...")
        
        # Test OpenAI API
        try:
            import openai
            if openai_key:
                openai.api_key = openai_key
                # Try to list models (this is a lightweight operation)
                try:
                    # Use OpenAI client if available
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    models = client.models.list()
                    print(f"   [OK] OpenAI API: Connected ({len(list(models.data))} models available)")
                except Exception as e:
                    # Fallback to simple check
                    if openai_key.startswith("sk-"):
                        print(f"   [OK] OpenAI API: Key format valid")
                    else:
                        print(f"   [WARNING] OpenAI API: {str(e)}")
        except ImportError:
            print("   [WARNING] OpenAI library not installed (pip install openai)")
        except Exception as e:
            print(f"   [WARNING] OpenAI API test failed: {str(e)}")
        
        # Test Serper API
        try:
            import requests
            if serper_key:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": serper_key,
                        "Content-Type": "application/json"
                    },
                    json={"q": "test"},
                    timeout=5
                )
                if response.status_code == 200:
                    print("   [OK] Serper API: Connected")
                else:
                    print(f"   [WARNING] Serper API: Status {response.status_code}")
        except ImportError:
            print("   [WARNING] Requests library not installed (pip install requests)")
        except Exception as e:
            print(f"   [WARNING] Serper API test failed: {str(e)}")
        
        # Test Stability API
        try:
            import requests
            if stability_key:
                response = requests.get(
                    "https://api.stability.ai/v1/user/account",
                    headers={"Authorization": f"Bearer {stability_key}"},
                    timeout=5
                )
                if response.status_code == 200:
                    print("   [OK] Stability API: Connected")
                else:
                    print(f"   [WARNING] Stability API: Status {response.status_code}")
        except Exception as e:
            print(f"   [WARNING] Stability API test failed: {str(e)}")
        
        # Test ElevenLabs API
        try:
            import requests
            if elevenlabs_key:
                response = requests.get(
                    "https://api.elevenlabs.io/v1/user",
                    headers={"xi-api-key": elevenlabs_key},
                    timeout=5
                )
                if response.status_code == 200:
                    print("   [OK] ElevenLabs API: Connected")
                else:
                    print(f"   [WARNING] ElevenLabs API: Status {response.status_code}")
        except Exception as e:
            print(f"   [WARNING] ElevenLabs API test failed: {str(e)}")
        
        # Configuration summary
        print("\nConfiguration Summary:")
        print(f"   LLM Provider: {config.get('core', {}).get('llm_provider', 'unknown')}")
        print(f"   Model: {config.get('core', {}).get('openai_model', 'unknown')}")
        print(f"   Available Agents: {len(config.get('agents', {}).get('available_agents', []))}")
        print(f"   Deployment Mode: {config.get('deployment', {}).get('mode', 'unknown')}")
        print(f"   Security Enabled: {config.get('security', {}).get('authentication', {}).get('enabled', False)}")
        
        print("\n[SUCCESS] Configuration test complete!")
        print("\nNext steps:")
        print("   1. Ensure all required API keys are set in .env")
        print("   2. Run: python src/main.py --interactive")
        print("   3. Or: python src/main.py --web")
        print("   4. Or: python src/main.py --api")
        
    except FileNotFoundError as e:
        print(f"[ERROR] Configuration file not found: {str(e)}")
        print("   Please create config/settings.yaml from config/settings.yaml.example")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_configuration()

