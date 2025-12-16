"""
Tests for Planner Agent
"""

import pytest
from unittest.mock import Mock
from src.agents.planner import PlannerAgent
from src.models.llm_manager import LLMManager

def test_planner_agent_init():
    """Test PlannerAgent initialization."""
    llm_manager = Mock(spec=LLMManager)
    config = {}
    agent = PlannerAgent(llm_manager, config)
    assert agent is not None
    assert agent.name == "PlannerAgent"

def test_planner_execute():
    """Test planner execution."""
    llm_manager = Mock(spec=LLMManager)
    llm_manager.generate.return_value = "Test plan"
    config = {}
    agent = PlannerAgent(llm_manager, config)
    
    result = agent.execute("Test task")
    assert "agent" in result
    assert result["agent"] == "planner"
    assert "plan" in result
    assert "subtasks" in result

