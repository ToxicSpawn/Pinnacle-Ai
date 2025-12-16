"""
Coder Agent - Code generation and execution.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models.llm_manager import LLMManager
from src.tools.code_executor import CodeExecutor

class CoderAgent(BaseAgent):
    """Agent for code generation and execution."""

    def __init__(self, llm_manager: LLMManager, config: Dict, logic_engine=None):
        """Initialize coder agent."""
        super().__init__(config, logic_engine)
        self.llm_manager = llm_manager
        self.code_executor = CodeExecutor()

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Generate and execute code."""
        context = context or {}
        self.logger.info(f"Coding task: {task[:50]}...")
        
        # Generate code
        prompt = f"Write Python code to: {task}"
        code = self.llm_manager.generate(prompt)
        
        # Execute code (optional)
        execution_result = None
        if context.get("execute", False):
            execution_result = self.code_executor.execute(code)
        
        return {
            "agent": "coder",
            "task": task,
            "code": code,
            "execution": execution_result,
            "result": {"code": code, "execution": execution_result}
        }

