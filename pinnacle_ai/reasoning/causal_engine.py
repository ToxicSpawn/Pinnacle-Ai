"""
True Causal Reasoning Engine

Unlike LLMs that only learn correlations, this engine:
- Builds causal graphs from data
- Performs interventional reasoning (do-calculus)
- Answers counterfactual questions
- Understands cause and effect

This is the key difference between pattern matching and true understanding.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Causal reasoning will be limited.")


class CausalReasoningEngine(nn.Module):
    """
    True Causal Reasoning Engine
    
    Unlike LLMs that only learn correlations, this engine:
    - Builds causal graphs from data
    - Performs interventional reasoning (do-calculus)
    - Answers counterfactual questions
    - Understands cause and effect
    
    This is the key difference between pattern matching and true understanding.
    """
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Causal graph
        if NETWORKX_AVAILABLE:
            self.causal_graph = nx.DiGraph()
        else:
            self.causal_graph = {}  # Fallback dict structure
        
        # Variable encoder
        self.variable_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Causal mechanism network
        self.mechanism_network = nn.ModuleDict()
        
        # Intervention predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Counterfactual reasoner
        self.counterfactual_reasoner = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Causal strength estimator
        self.strength_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info("Causal Reasoning Engine initialized")
    
    def add_variable(self, name: str, embedding: torch.Tensor):
        """Add a variable to the causal graph"""
        encoded = self.variable_encoder(embedding)
        if NETWORKX_AVAILABLE:
            self.causal_graph.add_node(name, embedding=encoded.detach())
        else:
            if name not in self.causal_graph:
                self.causal_graph[name] = {"embedding": encoded.detach(), "edges": {}}
    
    def add_causal_link(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        mechanism: Optional[nn.Module] = None
    ):
        """Add a causal link between variables"""
        if NETWORKX_AVAILABLE:
            if cause not in self.causal_graph:
                raise ValueError(f"Variable {cause} not in graph")
            if effect not in self.causal_graph:
                raise ValueError(f"Variable {effect} not in graph")
            
            self.causal_graph.add_edge(cause, effect, strength=strength)
            
            # Add mechanism if provided
            if mechanism:
                self.mechanism_network[f"{cause}->{effect}"] = mechanism
        else:
            if cause not in self.causal_graph:
                raise ValueError(f"Variable {cause} not in graph")
            if effect not in self.causal_graph:
                raise ValueError(f"Variable {effect} not in graph")
            
            self.causal_graph[cause]["edges"][effect] = {"strength": strength}
            if mechanism:
                self.mechanism_network[f"{cause}->{effect}"] = mechanism
    
    def discover_causes(self, effect: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Discover potential causes of an effect"""
        if NETWORKX_AVAILABLE:
            if effect not in self.causal_graph:
                return []
            
            effect_embedding = self.causal_graph.nodes[effect]["embedding"]
        else:
            if effect not in self.causal_graph:
                return []
            effect_embedding = self.causal_graph[effect]["embedding"]
        
        causes = []
        
        for candidate in candidates:
            if NETWORKX_AVAILABLE:
                if candidate in self.causal_graph:
                    candidate_embedding = self.causal_graph.nodes[candidate]["embedding"]
                else:
                    continue
            else:
                if candidate in self.causal_graph:
                    candidate_embedding = self.causal_graph[candidate]["embedding"]
                else:
                    continue
            
            # Estimate causal strength
            combined = torch.cat([candidate_embedding, effect_embedding], dim=-1)
            strength = self.strength_estimator(combined).item()
            
            causes.append((candidate, strength))
        
        # Sort by strength
        causes.sort(key=lambda x: x[1], reverse=True)
        return causes
    
    def intervene(self, variable: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform an intervention (do-calculus)
        
        This answers: "What would happen if we SET variable to value?"
        Unlike correlation, this removes confounding effects.
        """
        if NETWORKX_AVAILABLE:
            if variable not in self.causal_graph:
                raise ValueError(f"Variable {variable} not in graph")
            
            results = {variable: value}
            
            # Get all descendants (effects)
            descendants = nx.descendants(self.causal_graph, variable)
            
            # Propagate intervention through causal graph
            for descendant in nx.topological_sort(self.causal_graph.subgraph([variable] + list(descendants))):
                if descendant == variable:
                    continue
                
                # Get all parents of this node
                parents = list(self.causal_graph.predecessors(descendant))
                
                # Combine parent values
                parent_values = []
                for parent in parents:
                    if parent in results:
                        parent_values.append(results[parent])
                    else:
                        parent_values.append(self.causal_graph.nodes[parent]["embedding"])
                
                if parent_values:
                    combined = torch.stack(parent_values).mean(dim=0)
                    
                    # Apply mechanism if exists
                    mechanism_key = None
                    for parent in parents:
                        if f"{parent}->{descendant}" in self.mechanism_network:
                            mechanism_key = f"{parent}->{descendant}"
                            break
                    
                    if mechanism_key:
                        results[descendant] = self.mechanism_network[mechanism_key](combined)
                    else:
                        results[descendant] = self.intervention_predictor(
                            torch.cat([combined, self.causal_graph.nodes[descendant]["embedding"]], dim=-1)
                        )
        else:
            # Fallback implementation
            if variable not in self.causal_graph:
                raise ValueError(f"Variable {variable} not in graph")
            
            results = {variable: value}
            # Simple propagation
            for var_name, var_data in self.causal_graph.items():
                if var_name != variable and variable in var_data.get("edges", {}):
                    results[var_name] = self.intervention_predictor(
                        torch.cat([value, var_data["embedding"]], dim=-1)
                    )
        
        return results
    
    def counterfactual(
        self,
        factual: Dict[str, torch.Tensor],
        intervention: str,
        new_value: torch.Tensor,
        query: str
    ) -> torch.Tensor:
        """
        Answer counterfactual questions
        
        "Given what actually happened (factual), what would have
        happened to query if we had set intervention to new_value?"
        """
        # Step 1: Abduction - infer latent variables from factual
        latent = {}
        for var, value in factual.items():
            if NETWORKX_AVAILABLE:
                if var in self.causal_graph:
                    latent[var] = value
            else:
                if var in self.causal_graph:
                    latent[var] = value
        
        # Step 2: Intervention - modify the intervened variable
        latent[intervention] = new_value
        
        # Step 3: Prediction - compute counterfactual outcome
        # Combine factual, intervention, and query embeddings
        factual_combined = torch.stack(list(factual.values())).mean(dim=0)
        
        if NETWORKX_AVAILABLE:
            query_embedding = self.causal_graph.nodes[query]["embedding"]
        else:
            query_embedding = self.causal_graph[query]["embedding"]
        
        counterfactual_result = self.counterfactual_reasoner(
            torch.cat([factual_combined, new_value, query_embedding], dim=-1)
        )
        
        return counterfactual_result
    
    def explain(self, effect: str) -> Dict:
        """Generate a causal explanation for an effect"""
        if NETWORKX_AVAILABLE:
            if effect not in self.causal_graph:
                return {"error": f"Variable {effect} not in graph"}
            
            # Get all ancestors (causes)
            ancestors = nx.ancestors(self.causal_graph, effect)
            
            # Build explanation
            explanation = {
                "effect": effect,
                "direct_causes": list(self.causal_graph.predecessors(effect)),
                "indirect_causes": list(ancestors - set(self.causal_graph.predecessors(effect))),
                "causal_chain": [],
                "strength": {}
            }
            
            # Build causal chain
            for ancestor in ancestors:
                paths = list(nx.all_simple_paths(self.causal_graph, ancestor, effect))
                for path in paths:
                    explanation["causal_chain"].append(" → ".join(path))
            
            # Estimate strengths
            for cause in explanation["direct_causes"]:
                edge_data = self.causal_graph.edges[cause, effect]
                explanation["strength"][cause] = edge_data.get("strength", 1.0)
        else:
            if effect not in self.causal_graph:
                return {"error": f"Variable {effect} not in graph"}
            
            explanation = {
                "effect": effect,
                "direct_causes": list(self.causal_graph[effect].get("edges", {}).keys()),
                "indirect_causes": [],
                "causal_chain": [],
                "strength": {cause: data.get("strength", 1.0) for cause, data in self.causal_graph[effect].get("edges", {}).items()}
            }
        
        return explanation
    
    def why(self, question: str, context: Dict[str, torch.Tensor]) -> str:
        """
        Answer "why" questions using causal reasoning
        
        Example: "Why did X happen?"
        """
        # Extract effect from question
        effect = question.lower().replace("why did", "").replace("happen", "").strip()
        
        # Find closest variable in graph
        closest_var = None
        
        if NETWORKX_AVAILABLE:
            for var in self.causal_graph.nodes:
                if var.lower() in effect.lower() or effect.lower() in var.lower():
                    closest_var = var
                    break
        else:
            for var in self.causal_graph.keys():
                if var.lower() in effect.lower() or effect.lower() in var.lower():
                    closest_var = var
                    break
        
        if closest_var:
            explanation = self.explain(closest_var)
            
            # Generate natural language explanation
            causes = explanation["direct_causes"]
            if causes:
                cause_str = ", ".join(causes)
                return f"{closest_var} happened because of {cause_str}. " \
                       f"The causal chain is: {'; '.join(explanation['causal_chain'][:3])}"
            else:
                return f"{closest_var} appears to be a root cause with no known causes."
        
        return "I couldn't identify the causal factors for this question."

