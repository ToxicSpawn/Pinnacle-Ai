"""
Config Loader - Loads configuration files.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    
    if not path.exists():
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif path.suffix == '.json':
                return json.load(f)
            else:
                # Try YAML first, then JSON
                content = f.read()
                try:
                    return yaml.safe_load(content) or {}
                except:
                    return json.loads(content)
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

