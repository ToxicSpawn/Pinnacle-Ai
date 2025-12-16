"""
Researcher Agent - Information gathering and synthesis.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models.llm_manager import LLMManager
from src.tools.web_search import WebSearch

class ResearcherAgent(BaseAgent):
    """Agent for research and information gathering."""

    def __init__(self, llm_manager: LLMManager, config: Dict, logic_engine=None):
        """Initialize researcher agent."""
        super().__init__(config, logic_engine)
        self.llm_manager = llm_manager
        self.web_search = WebSearch()

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Research a topic."""
        context = context or {}
        self.logger.info(f"Researching: {task[:50]}...")
        
        # Search for information
        search_results = self.web_search.search(task)
        
        # Synthesize information
        synthesis = self._synthesize(search_results, task)
        
        return {
            "agent": "researcher",
            "task": task,
            "sources": len(search_results),
            "result": {"synthesis": synthesis, "sources": search_results}
        }

    def _synthesize(self, sources: list, query: str) -> str:
        """Synthesize information from sources."""
        prompt = f"Synthesize information about: {query}"
        return self.llm_manager.generate(prompt)

