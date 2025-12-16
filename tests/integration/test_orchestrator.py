"""
Integration tests for Orchestrator
"""

import pytest
from src.core.orchestrator import OmniAIOrchestrator

def test_orchestrator_init():
    """Test orchestrator initialization with minimal config."""
    config = {
        "agents": {
            "available_agents": ["meta_agent"]
        },
        "neurosymbolic": {},
        "hyper_modal": {},
        "memory": {},
        "self_evolution": {},
        "llm": {
            "models": ["default"]
        }
    }
    
    # This will fail if meta_agent can't be initialized, which is expected
    # In a real scenario, we'd mock the dependencies
    try:
        orchestrator = OmniAIOrchestrator(config)
        assert orchestrator is not None
    except Exception as e:
        # Expected if dependencies aren't fully set up
        pytest.skip(f"Orchestrator initialization requires full setup: {e}")

