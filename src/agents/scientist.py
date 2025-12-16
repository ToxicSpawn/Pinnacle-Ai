"""
Scientist Agent - Scientific research and analysis.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models.llm_manager import LLMManager

class ScientistAgent(BaseAgent):
    """Agent for scientific research and analysis."""

    def __init__(self, llm_manager: LLMManager, config: Dict, logic_engine=None):
        """Initialize scientist agent."""
        super().__init__(config, logic_engine)
        self.llm_manager = llm_manager

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Perform scientific research."""
        context = context or {}
        self.logger.info(f"Scientific task: {task[:50]}...")
        
        # Generate hypothesis
        hypothesis_prompt = f"Formulate a hypothesis for: {task}"
        hypothesis = self.llm_manager.generate(hypothesis_prompt)
        
        # Design experiment
        experiment_prompt = f"Design an experiment to test: {hypothesis}"
        experiment = self.llm_manager.generate(experiment_prompt)
        
        return {
            "agent": "scientist",
            "task": task,
            "hypothesis": hypothesis,
            "experiment": experiment,
            "result": {"hypothesis": hypothesis, "experiment": experiment}
        }

