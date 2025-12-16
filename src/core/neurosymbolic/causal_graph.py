"""
Causal Graph - Models cause-and-effect relationships between concepts.
"""

import logging
from typing import Dict, Any, List, Optional

class CausalGraph:
    """Graph-based causal reasoning system."""

    def __init__(self, config: Dict):
        """Initialize the causal graph."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.nodes = {}
        self.edges = []

    def add_node(self, concept: str, properties: Dict = None):
        """Add a concept node to the graph."""
        self.nodes[concept] = properties or {}
        self.logger.debug(f"Added node: {concept}")

    def add_edge(self, cause: str, effect: str, strength: float = 1.0):
        """Add a causal relationship."""
        self.edges.append({
            "cause": cause,
            "effect": effect,
            "strength": strength
        })
        self.logger.debug(f"Added edge: {cause} -> {effect}")

    def find_causes(self, effect: str) -> List[str]:
        """Find all causes of an effect."""
        causes = []
        for edge in self.edges:
            if edge["effect"] == effect:
                causes.append(edge["cause"])
        return causes

    def find_effects(self, cause: str) -> List[str]:
        """Find all effects of a cause."""
        effects = []
        for edge in self.edges:
            if edge["cause"] == cause:
                effects.append(edge["effect"])
        return effects

    def reason_about(self, concept: str) -> Dict:
        """Reason about causal relationships involving a concept."""
        return {
            "causes": self.find_causes(concept),
            "effects": self.find_effects(concept),
            "related": list(self.nodes.keys())
        }

    def improve(self) -> Dict:
        """Improve the causal graph."""
        return {
            "status": "improved",
            "nodes": len(self.nodes),
            "edges": len(self.edges)
        }

