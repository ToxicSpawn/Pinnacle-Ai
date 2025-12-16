"""
Meta-Agent - Coordination of all other agents.
"""

import logging
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent

class MetaAgent(BaseAgent):
    """Meta-agent that coordinates all other agents."""

    def __init__(self, orchestrator, logic_engine, meta_learner, memory):
        """Initialize meta-agent."""
        super().__init__({}, logic_engine)
        self.orchestrator = orchestrator
        self.meta_learner = meta_learner
        self.memory = memory
        self.logger = logging.getLogger(__name__)

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Execute a task by coordinating appropriate agents."""
        context = context or {}
        self.logger.info(f"Meta-agent executing: {task[:50]}...")
        
        # Store task in memory
        self.memory.store(f"task_{len(self.memory.memory)}", {"task": task, "context": context})
        
        # Select appropriate agents
        selected_agents = self._select_agents(task, context)
        
        # Execute with selected agents
        execution_results = []
        for agent_name in selected_agents:
            agent = self.orchestrator.agents.get(agent_name)
            if agent:
                try:
                    result = agent.execute(task, context)
                    execution_results.append(result)
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    execution_results.append({
                        "agent": agent_name,
                        "error": str(e)
                    })
        
        # Synthesize results
        final_result = self._synthesize_results(execution_results, task)
        
        # Evaluate
        evaluation = self._evaluate(final_result, task)
        
        # Learn
        learning = self._learn(task, execution_results, evaluation)
        
        return {
            "task": task,
            "execution": {"execution": execution_results},
            "result": final_result,
            "evaluation": evaluation,
            "learning": learning
        }

    def _select_agents(self, task: str, context: Dict) -> List[str]:
        """Select appropriate agents for a task."""
        # Simple keyword-based selection (would be more sophisticated in production)
        task_lower = task.lower()
        selected = []
        
        if any(word in task_lower for word in ["plan", "strategy", "organize"]):
            selected.append("planner")
        if any(word in task_lower for word in ["research", "find", "search", "information"]):
            selected.append("researcher")
        if any(word in task_lower for word in ["code", "program", "script", "function"]):
            selected.append("coder")
        if any(word in task_lower for word in ["create", "art", "story", "music", "creative"]):
            selected.append("creative")
        if any(word in task_lower for word in ["robot", "robotic", "physical", "move"]):
            selected.append("robotic")
        if any(word in task_lower for word in ["science", "experiment", "hypothesis", "research"]):
            selected.append("scientist")
        if any(word in task_lower for word in ["philosophy", "meaning", "ethics", "abstract"]):
            selected.append("philosopher")
        
        # Default to planner and researcher if no specific match
        if not selected:
            selected = ["planner", "researcher"]
        
        return selected

    def _synthesize_results(self, results: List[Dict], task: str) -> Dict:
        """Synthesize results from multiple agents."""
        synthesized = {
            "task": task,
            "contributions": [r.get("result", {}) for r in results],
            "agents_used": [r.get("agent", "unknown") for r in results]
        }
        return synthesized

    def _evaluate(self, result: Dict, task: str) -> Dict:
        """Evaluate the quality of the result."""
        # Placeholder evaluation
        return {
            "success": True,
            "quality": 0.85,
            "efficiency": 0.80
        }

    def _learn(self, task: str, execution_results: List[Dict], evaluation: Dict) -> Dict:
        """Learn from the execution."""
        # Store learning outcomes
        learning_outcomes = {
            "task_type": self._classify_task(task),
            "agents_used": [r.get("agent") for r in execution_results],
            "performance": evaluation
        }
        
        # Update meta-learner
        if evaluation.get("success"):
            self.meta_learner.update_performance("meta_agent", evaluation.get("quality", 0.0))
        
        return {"learning_outcomes": learning_outcomes}

    def _classify_task(self, task: str) -> str:
        """Classify the type of task."""
        task_lower = task.lower()
        if "code" in task_lower or "program" in task_lower:
            return "coding"
        elif "research" in task_lower:
            return "research"
        elif "create" in task_lower or "art" in task_lower:
            return "creative"
        else:
            return "general"

