"""
Configuration loader with environment variable resolution
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not installed. Environment variables must be set manually.")


def load_dotenv_if_available():
    """Load environment variables from .env file if dotenv is available"""
    if DOTENV_AVAILABLE:
        load_dotenv()


def resolve_env_vars(config: Any) -> Any:
    """
    Resolve environment variables in configuration
    
    Supports ${VARIABLE_NAME} syntax for environment variable substitution
    """
    if isinstance(config, dict):
        resolved = {}
        for key, value in config.items():
            resolved[key] = resolve_env_vars(value)
        return resolved
    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Check if it's an environment variable reference
        if config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            value = os.getenv(env_var, "")
            if not value:
                logging.warning(f"Environment variable {env_var} not set, using empty string")
            return value
        return config
    else:
        return config


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load and resolve configuration from YAML or JSON file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the resolved configuration
    """
    # Load environment variables from .env file if available
    load_dotenv_if_available()
    
    path = Path(config_path)
    
    if not path.exists():
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f) or {}
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                # Try YAML first, then JSON
                content = f.read()
                try:
                    config = yaml.safe_load(content) or {}
                except:
                    config = json.loads(content)
        
        # Resolve environment variables
        resolved_config = resolve_env_vars(config)
        
        return resolved_config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict:
    """Get default configuration."""
    return {
        "agents": {
            "available_agents": [
                "planner",
                "researcher",
                "coder",
                "creative",
                "robotic",
                "scientist",
                "philosopher",
                "meta_agent"
            ]
        },
        "neurosymbolic": {},
        "hyper_modal": {},
        "memory": {},
        "self_evolution": {},
        "logging": {
            "level": "INFO"
        }
    }


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation path
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., "core.api_keys.openai")
        default: Default value if path doesn't exist
        
    Returns:
        Configuration value or default
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate configuration for required fields
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for required API keys
    required_keys = [
        "core.api_keys.openai",
        "tools.search.serper.api_key",
    ]
    
    for key_path in required_keys:
        value = get_config_value(config, key_path)
        if not value or value == "":
            errors.append(f"Missing or empty configuration: {key_path}")
    
    # Check for valid LLM provider
    llm_provider = get_config_value(config, "core.llm_provider", "openai")
    if llm_provider not in ["openai", "anthropic", "ollama", "mistral"]:
        errors.append(f"Invalid LLM provider: {llm_provider}")
    
    # Check for valid deployment mode
    deployment_mode = get_config_value(config, "deployment.mode", "local")
    if deployment_mode not in ["local", "docker", "kubernetes", "cloud"]:
        errors.append(f"Invalid deployment mode: {deployment_mode}")
    
    return len(errors) == 0, errors
