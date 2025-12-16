"""
Base Agent - Base class for all agents.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, config: Dict, logic_engine=None):
        """Initialize base agent."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logic_engine = logic_engine
        self.name = self.__class__.__name__

    @abstractmethod
    def execute(self, task: str, context: Dict = None) -> Dict:
        """Execute a task."""
        pass

    def improve(self) -> Dict:
        """Improve the agent."""
        return {"status": "improved", "agent": self.name}

