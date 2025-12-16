"""
Tests for Logic Engine
"""

import pytest
from src.core.neurosymbolic.logic_engine import LogicEngine

def test_logic_engine_init():
    """Test LogicEngine initialization."""
    engine = LogicEngine({})
    assert engine is not None
    assert engine.rules == []
    assert engine.constraints == []

def test_add_rule():
    """Test adding a rule."""
    engine = LogicEngine({})
    engine.add_rule("If A then B")
    assert len(engine.rules) == 1
    assert "If A then B" in engine.rules

def test_add_constraint():
    """Test adding a constraint."""
    engine = LogicEngine({})
    engine.add_constraint("A must be positive")
    assert len(engine.constraints) == 1
    assert "A must be positive" in engine.constraints

def test_reason():
    """Test reasoning."""
    engine = LogicEngine({})
    premises = ["A is true", "B is true"]
    conclusions = engine.reason(premises)
    assert len(conclusions) == len(premises)

def test_verify():
    """Test verification."""
    engine = LogicEngine({})
    result = engine.verify("A is true")
    assert isinstance(result, bool)

def test_improve():
    """Test improvement."""
    engine = LogicEngine({})
    engine.add_rule("Test rule")
    result = engine.improve()
    assert "status" in result
    assert result["status"] == "improved"

