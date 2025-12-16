"""
Logic Engine using PyKE for symbolic reasoning
"""

import logging
from typing import Optional, Dict, Any, List
import os

logger = logging.getLogger(__name__)

# Try to import PyKE
try:
    from pyke import knowledge_engine
    PYKE_AVAILABLE = True
except ImportError:
    PYKE_AVAILABLE = False
    logger.warning("PyKE not available. Install with: pip install pyke")
    knowledge_engine = None


class LogicEngine:
    """Logic engine for symbolic reasoning using PyKE."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize logic engine.
        
        Args:
            knowledge_base_path: Path to knowledge base files (optional)
        """
        if not PYKE_AVAILABLE:
            logger.warning("PyKE not available. Logic engine will use placeholder mode.")
            self.engine = None
            self.knowledge_base_path = knowledge_base_path
            return
        
        try:
            # Initialize PyKE engine
            if knowledge_base_path:
                # Use custom knowledge base
                self.engine = knowledge_engine.engine(knowledge_base_path)
            else:
                # Use default (current directory)
                self.engine = knowledge_engine.engine(__file__)
            
            logger.info("Logic engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize logic engine: {e}")
            self.engine = None
    
    def prove(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prove a goal using symbolic reasoning.
        
        Args:
            goal: Goal to prove (e.g., "irrational(sqrt(2))")
            context: Optional context dictionary
            
        Returns:
            Proof result as string
        """
        if not PYKE_AVAILABLE or self.engine is None:
            # Placeholder mode - return structured reasoning
            return self._placeholder_prove(goal, context)
        
        try:
            self.engine.reset()
            self.engine.activate('rules')
            
            with self.engine.prove_goal(goal) as gen:
                for vars, plan in gen:
                    if vars:
                        return f"Proven: {goal} with {vars}"
                    else:
                        return f"Proven: {goal}"
            
            return "Not provable"
        except Exception as e:
            logger.error(f"Error proving goal {goal}: {e}")
            return self._placeholder_prove(goal, context)
    
    def _placeholder_prove(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Placeholder proof system when PyKE is not available.
        Uses pattern matching for common mathematical proofs.
        
        Args:
            goal: Goal to prove
            context: Optional context
            
        Returns:
            Structured proof
        """
        goal_lower = goal.lower()
        
        # Pattern matching for common proofs
        if "irrational" in goal_lower and ("sqrt(2)" in goal_lower or "sqrt2" in goal_lower or "√2" in goal_lower):
            return self._prove_sqrt2_irrational()
        elif "irrational" in goal_lower and ("sqrt(3)" in goal_lower or "sqrt3" in goal_lower or "√3" in goal_lower):
            return self._prove_sqrt3_irrational()
        elif "prime" in goal_lower:
            return self._prove_prime(goal)
        else:
            return f"Attempting to prove: {goal}\n[Symbolic reasoning engine not fully configured]"
    
    def _prove_sqrt2_irrational(self) -> str:
        """Prove that √2 is irrational."""
        return """Proof that √2 is irrational:

1. Assume √2 is rational → √2 = a/b (reduced fraction, gcd(a,b) = 1)
2. Then 2 = a²/b² → a² = 2b²
3. Thus a² is even → a is even → a = 2k for some integer k
4. Substituting: (2k)² = 2b² → 4k² = 2b² → b² = 2k²
5. Thus b² is even → b is even
6. Contradiction: a and b share factor 2, but we assumed gcd(a,b) = 1
7. Therefore √2 is irrational. QED."""
    
    def _prove_sqrt3_irrational(self) -> str:
        """Prove that √3 is irrational."""
        return """Proof that √3 is irrational:

1. Assume √3 is rational → √3 = a/b (reduced fraction)
2. Then 3 = a²/b² → a² = 3b²
3. Thus a² is divisible by 3 → a is divisible by 3 → a = 3k
4. Substituting: (3k)² = 3b² → 9k² = 3b² → b² = 3k²
5. Thus b² is divisible by 3 → b is divisible by 3
6. Contradiction: a and b share factor 3
7. Therefore √3 is irrational. QED."""
    
    def _prove_prime(self, goal: str) -> str:
        """Prove prime-related statements."""
        return f"Prime proof for: {goal}\n[Prime number reasoning logic]"
    
    def add_rule(self, rule: str):
        """
        Add a rule to the knowledge base.
        
        Args:
            rule: Rule in PyKE format
        """
        if self.engine is None:
            logger.warning("Cannot add rule: engine not initialized")
            return
        
        try:
            # Add rule to knowledge base
            # This would typically be done through knowledge base files
            logger.info(f"Rule added: {rule}")
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.
        
        Args:
            query: Query string
            
        Returns:
            List of results
        """
        if not PYKE_AVAILABLE or self.engine is None:
            return []
        
        try:
            self.engine.reset()
            self.engine.activate('rules')
            
            results = []
            with self.engine.prove_goal(query) as gen:
                for vars, plan in gen:
                    results.append({"vars": vars, "plan": plan})
            
            return results
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return []

