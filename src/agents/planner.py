"""
Planner Agent - Strategic task decomposition and planning.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models.llm_manager import LLMManager

class PlannerAgent(BaseAgent):
    """Agent for strategic planning and task decomposition."""

    def __init__(self, llm_manager: LLMManager, config: Dict, logic_engine=None):
        """Initialize planner agent."""
        super().__init__(config, logic_engine)
        self.llm_manager = llm_manager

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Plan and decompose a task."""
        context = context or {}
        self.logger.info(f"Planning task: {task[:50]}...")
        
        # Use LLM to generate a plan
        prompt = f"Create a detailed plan for: {task}"
        plan = self.llm_manager.generate(prompt)
        
        # Decompose into subtasks
        subtasks = self._decompose_task(task, plan)
        
        return {
            "agent": "planner",
            "task": task,
            "plan": plan,
            "subtasks": subtasks,
            "result": {"plan": plan, "subtasks": subtasks}
        }

    def _decompose_task(self, task: str, plan: str) -> list:
        """Decompose task into subtasks."""
        # Placeholder: would use logic engine or LLM for actual decomposition
        return [f"Subtask 1 for {task}", f"Subtask 2 for {task}"]

