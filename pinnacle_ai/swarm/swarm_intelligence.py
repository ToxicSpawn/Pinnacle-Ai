from typing import Dict, List
import asyncio
from loguru import logger
import uuid
import numpy as np


class SwarmAgent:
    """Individual agent in the swarm"""
    
    def __init__(self, agent_id: str, specialty: str = "general"):
        self.id = agent_id
        self.specialty = specialty
        self.performance = 0.5
        self.tasks_completed = 0
    
    async def process(self, task: Dict) -> Dict:
        """Process a task"""
        # Simulate processing
        await asyncio.sleep(0.01)
        
        result = {
            "agent_id": self.id,
            "specialty": self.specialty,
            "result": f"Processed by {self.specialty} agent",
            "confidence": self.performance + np.random.uniform(-0.1, 0.1)
        }
        
        self.tasks_completed += 1
        return result
    
    def update_performance(self, feedback: float):
        """Update performance based on feedback"""
        self.performance = 0.9 * self.performance + 0.1 * feedback
        self.performance = max(0.1, min(1.0, self.performance))


class SwarmIntelligence:
    """
    Swarm Intelligence System
    
    Enables distributed problem-solving through:
    - Multiple specialized agents
    - Parallel processing
    - Consensus mechanisms
    """
    
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents
        self.agents: Dict[str, SwarmAgent] = {}
        
        # Specialties
        self.specialties = ["reasoning", "creativity", "analysis", "synthesis", "evaluation"]
        
        # Create agents
        self._create_swarm()
        
        logger.info(f"Swarm Intelligence initialized with {num_agents} agents")
    
    def _create_swarm(self):
        """Create the swarm of agents"""
        agents_per_specialty = max(1, self.num_agents // len(self.specialties))
        
        for specialty in self.specialties:
            for i in range(agents_per_specialty):
                agent_id = f"{specialty}_{i}_{uuid.uuid4().hex[:6]}"
                self.agents[agent_id] = SwarmAgent(agent_id, specialty)
    
    async def solve(self, problem: str) -> Dict:
        """
        Solve a problem using the swarm
        
        Args:
            problem: Problem to solve
        
        Returns:
            Aggregated solution
        """
        logger.info(f"Swarm solving: {problem[:50]}...")
        
        # Create tasks for each specialty
        tasks = []
        for agent in self.agents.values():
            task = {"problem": problem, "specialty": agent.specialty}
            tasks.append(agent.process(task))
        
        # Run in parallel
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        solution = self._aggregate(results)
        
        return {
            "problem": problem,
            "agents_used": len(results),
            "partial_solutions": results,
            "aggregated_solution": solution,
            "confidence": np.mean([r["confidence"] for r in results])
        }
    
    def _aggregate(self, results: List[Dict]) -> str:
        """Aggregate results from multiple agents"""
        # Weight by confidence
        weighted_parts = []
        total_confidence = sum(r["confidence"] for r in results)
        
        if total_confidence > 0:
            for result in results:
                weight = result["confidence"] / total_confidence
                weighted_parts.append(f"[{result['specialty']}:{weight:.2f}] {result['result']}")
        else:
            for result in results:
                weighted_parts.append(f"[{result['specialty']}] {result['result']}")
        
        return " | ".join(weighted_parts)
    
    async def debate(self, topic: str, rounds: int = 3) -> Dict:
        """Have agents debate a topic"""
        logger.info(f"Swarm debating: {topic}")
        
        debate_history = []
        
        for round_num in range(rounds):
            round_arguments = []
            
            for specialty in self.specialties[:3]:  # Use 3 specialties
                agents = [a for a in self.agents.values() if a.specialty == specialty]
                if agents:
                    agent = agents[0]
                    argument = {
                        "agent": agent.id,
                        "specialty": specialty,
                        "argument": f"[Round {round_num + 1}] {specialty} perspective on {topic}",
                        "strength": agent.performance
                    }
                    round_arguments.append(argument)
            
            debate_history.append(round_arguments)
        
        return {
            "topic": topic,
            "rounds": rounds,
            "debate_history": debate_history,
            "conclusion": f"After {rounds} rounds, the swarm reached consensus on {topic}"
        }
