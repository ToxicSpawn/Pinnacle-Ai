"""
Tests for Config Loader
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.tools.config_loader import load_config, get_default_config

def test_get_default_config():
    """Test default configuration."""
    config = get_default_config()
    assert "agents" in config
    assert "neurosymbolic" in config

def test_load_config_yaml(tmp_path):
    """Test loading YAML config."""
    config_file = tmp_path / "test.yaml"
    test_config = {"test": "value", "agents": {"available_agents": ["test"]}}
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    config = load_config(str(config_file))
    assert config["test"] == "value"

def test_load_config_missing():
    """Test loading missing config returns defaults."""
    config = load_config("nonexistent.yaml")
    assert "agents" in config

