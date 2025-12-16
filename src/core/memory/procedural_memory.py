"""
Procedural Memory - Memory of skills and procedures.
"""

import logging
from typing import Dict, Any, List, Optional

class ProceduralMemory:
    """Procedural memory system for skills and procedures."""

    def __init__(self, config: Dict):
        """Initialize procedural memory."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.procedures = {}

    def store_procedure(self, name: str, steps: List[str]):
        """Store a procedure."""
        self.procedures[name] = {
            "steps": steps,
            "usage_count": 0
        }
        self.logger.debug(f"Stored procedure: {name}")

    def retrieve_procedure(self, name: str) -> Optional[List[str]]:
        """Retrieve a procedure."""
        if name in self.procedures:
            self.procedures[name]["usage_count"] += 1
            return self.procedures[name]["steps"]
        return None

    def improve_procedure(self, name: str, improved_steps: List[str]):
        """Improve a stored procedure."""
        if name in self.procedures:
            self.procedures[name]["steps"] = improved_steps
            self.logger.info(f"Improved procedure: {name}")

