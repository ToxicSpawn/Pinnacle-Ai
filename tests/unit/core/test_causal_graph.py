"""
Tests for Causal Graph
"""

import pytest
from src.core.neurosymbolic.causal_graph import CausalGraph

def test_causal_graph_init():
    """Test CausalGraph initialization."""
    graph = CausalGraph({})
    assert graph is not None
    assert graph.nodes == {}
    assert graph.edges == []

def test_add_node():
    """Test adding a node."""
    graph = CausalGraph({})
    graph.add_node("concept1", {"property": "value"})
    assert "concept1" in graph.nodes
    assert graph.nodes["concept1"]["property"] == "value"

def test_add_edge():
    """Test adding an edge."""
    graph = CausalGraph({})
    graph.add_edge("cause", "effect", 0.8)
    assert len(graph.edges) == 1
    assert graph.edges[0]["cause"] == "cause"
    assert graph.edges[0]["effect"] == "effect"
    assert graph.edges[0]["strength"] == 0.8

def test_find_causes():
    """Test finding causes."""
    graph = CausalGraph({})
    graph.add_edge("cause1", "effect", 0.8)
    graph.add_edge("cause2", "effect", 0.9)
    causes = graph.find_causes("effect")
    assert "cause1" in causes
    assert "cause2" in causes

def test_find_effects():
    """Test finding effects."""
    graph = CausalGraph({})
    graph.add_edge("cause", "effect1", 0.8)
    graph.add_edge("cause", "effect2", 0.9)
    effects = graph.find_effects("cause")
    assert "effect1" in effects
    assert "effect2" in effects

def test_reason_about():
    """Test reasoning about a concept."""
    graph = CausalGraph({})
    graph.add_node("concept")
    graph.add_edge("cause", "concept", 0.8)
    graph.add_edge("concept", "effect", 0.9)
    result = graph.reason_about("concept")
    assert "causes" in result
    assert "effects" in result

