from typing import Dict, List, Optional
from loguru import logger

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Causal reasoning will be limited.")


class CausalEngine:
    """
    Causal Reasoning Engine
    
    Enables:
    - Causal graph construction
    - Why questions
    - Counterfactual reasoning
    """
    
    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = {}  # Fallback dict structure
        self.explanations = {}
        
        # Add some default causal knowledge
        self._init_default_knowledge()
        
        logger.info("Causal Engine initialized")
    
    def _init_default_knowledge(self):
        """Initialize with default causal relationships"""
        default_causes = [
            ("learning", "knowledge", "leads_to"),
            ("knowledge", "understanding", "enables"),
            ("understanding", "wisdom", "develops"),
            ("practice", "skill", "improves"),
            ("effort", "success", "contributes_to"),
            ("curiosity", "learning", "motivates"),
            ("failure", "learning", "can_lead_to"),
            ("collaboration", "innovation", "enables")
        ]
        
        for cause, effect, relation in default_causes:
            self.add_cause(cause, effect, relation)
    
    def add_cause(self, cause: str, effect: str, relation: str = "causes"):
        """Add a causal relationship"""
        if NETWORKX_AVAILABLE:
            self.graph.add_edge(cause, effect, relation=relation)
        else:
            if cause not in self.graph:
                self.graph[cause] = {}
            self.graph[cause][effect] = relation
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze causal structure in text
        
        Args:
            text: Text to analyze
        
        Returns:
            Causal analysis
        """
        # Extract potential causes and effects
        causal_words = ["because", "therefore", "so", "thus", "hence", "since", "as a result"]
        
        text_lower = text.lower()
        has_causal = any(word in text_lower for word in causal_words)
        
        # Find related concepts in graph
        related = []
        if NETWORKX_AVAILABLE:
            for node in self.graph.nodes:
                if node.lower() in text_lower:
                    related.append(node)
        else:
            for node in self.graph.keys():
                if node.lower() in text_lower:
                    related.append(node)
        
        return {
            "has_causal_language": has_causal,
            "related_concepts": related,
            "potential_causes": [list(self.graph.predecessors(n)) if NETWORKX_AVAILABLE else [] for n in related] if NETWORKX_AVAILABLE else [],
            "potential_effects": [list(self.graph.successors(n)) if NETWORKX_AVAILABLE else [] for n in related] if NETWORKX_AVAILABLE else []
        }
    
    def why(self, effect: str) -> str:
        """
        Answer why something happens
        
        Args:
            effect: The effect to explain
        
        Returns:
            Explanation
        """
        effect_lower = effect.lower()
        
        # Find matching node
        matching = None
        if NETWORKX_AVAILABLE:
            for node in self.graph.nodes:
                if node.lower() in effect_lower or effect_lower in node.lower():
                    matching = node
                    break
        else:
            for node in self.graph.keys():
                if node.lower() in effect_lower or effect_lower in node.lower():
                    matching = node
                    break
        
        if not matching:
            return f"I don't have causal knowledge about '{effect}' yet."
        
        # Get causes
        if NETWORKX_AVAILABLE:
            causes = list(self.graph.predecessors(matching))
        else:
            causes = [c for c in self.graph.keys() if matching in self.graph.get(c, {})]
        
        if not causes:
            return f"'{matching}' appears to be a root cause with no known causes."
        
        # Build explanation
        explanations = []
        for cause in causes:
            if NETWORKX_AVAILABLE:
                edge_data = self.graph.edges[cause, matching]
                relation = edge_data.get("relation", "causes")
            else:
                relation = self.graph.get(cause, {}).get(matching, "causes")
            explanations.append(f"'{cause}' {relation} '{matching}'")
        
        return f"Why {effect}? " + "; ".join(explanations)
    
    def counterfactual(self, scenario: str) -> Dict:
        """
        Counterfactual reasoning
        
        Args:
            scenario: "What if X?" scenario
        
        Returns:
            Counterfactual analysis
        """
        # Simple counterfactual analysis
        return {
            "scenario": scenario,
            "analysis": f"If we consider the counterfactual '{scenario}', we would need to trace the causal implications through our knowledge graph.",
            "confidence": 0.7
        }
    
    def explain(self, concept: str) -> Dict:
        """Get full causal explanation for a concept"""
        concept_lower = concept.lower()
        
        # Find matching node
        matching = None
        if NETWORKX_AVAILABLE:
            for node in self.graph.nodes:
                if node.lower() == concept_lower:
                    matching = node
                    break
        else:
            for node in self.graph.keys():
                if node.lower() == concept_lower:
                    matching = node
                    break
        
        if not matching:
            return {"error": f"Concept '{concept}' not found"}
        
        if NETWORKX_AVAILABLE:
            return {
                "concept": matching,
                "causes": list(self.graph.predecessors(matching)),
                "effects": list(self.graph.successors(matching)),
                "all_ancestors": list(nx.ancestors(self.graph, matching)),
                "all_descendants": list(nx.descendants(self.graph, matching))
            }
        else:
            causes = [c for c in self.graph.keys() if matching in self.graph.get(c, {})]
            effects = list(self.graph.get(matching, {}).keys())
            return {
                "concept": matching,
                "causes": causes,
                "effects": effects,
                "all_ancestors": causes,
                "all_descendants": effects
            }
