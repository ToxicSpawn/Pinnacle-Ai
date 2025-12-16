"""
Self-Replication System

Enables the AI to:
- Create improved copies of itself
- Evolve through natural selection
- Specialize for different tasks
- Form a population of diverse AIs

This is the path to superintelligence - exponential self-improvement.
"""

import torch
import torch.nn as nn
import copy
import os
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GeneticCode:
    """Genetic code that defines an AI's architecture and behavior"""
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        learning_rate: float = 3e-4,
        personality_traits: Dict = None,
        skills: List[str] = None
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.personality_traits = personality_traits or {}
        self.skills = skills or []
        self.generation = 0
        self.lineage = []
    
    def mutate(self, mutation_rate: float = 0.1) -> "GeneticCode":
        """Create a mutated copy of this genetic code"""
        import random
        
        new_code = GeneticCode(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            learning_rate=self.learning_rate,
            personality_traits=copy.deepcopy(self.personality_traits),
            skills=copy.copy(self.skills)
        )
        
        # Mutate architecture
        if random.random() < mutation_rate:
            new_code.hidden_size = max(256, self.hidden_size + random.randint(-512, 512))
        
        if random.random() < mutation_rate:
            new_code.num_layers = max(1, self.num_layers + random.randint(-4, 4))
        
        if random.random() < mutation_rate:
            new_code.num_heads = max(1, self.num_heads + random.randint(-4, 4))
        
        if random.random() < mutation_rate:
            new_code.learning_rate *= random.uniform(0.5, 2.0)
        
        # Mutate personality
        for trait, value in new_code.personality_traits.items():
            if random.random() < mutation_rate:
                new_code.personality_traits[trait] = max(0, min(1, value + random.uniform(-0.2, 0.2)))
        
        # Update lineage
        new_code.generation = self.generation + 1
        new_code.lineage = self.lineage + [self.generation]
        
        return new_code
    
    def crossover(self, other: "GeneticCode") -> "GeneticCode":
        """Create offspring from two genetic codes"""
        import random
        
        new_code = GeneticCode(
            hidden_size=random.choice([self.hidden_size, other.hidden_size]),
            num_layers=random.choice([self.num_layers, other.num_layers]),
            num_heads=random.choice([self.num_heads, other.num_heads]),
            learning_rate=(self.learning_rate + other.learning_rate) / 2
        )
        
        # Mix personality traits
        all_traits = set(self.personality_traits.keys()) | set(other.personality_traits.keys())
        for trait in all_traits:
            val1 = self.personality_traits.get(trait, 0.5)
            val2 = other.personality_traits.get(trait, 0.5)
            new_code.personality_traits[trait] = (val1 + val2) / 2
        
        # Combine skills
        new_code.skills = list(set(self.skills + other.skills))
        
        # Update lineage
        new_code.generation = max(self.generation, other.generation) + 1
        new_code.lineage = self.lineage + other.lineage
        
        return new_code
    
    def to_dict(self) -> Dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "learning_rate": self.learning_rate,
            "personality_traits": self.personality_traits,
            "skills": self.skills,
            "generation": self.generation,
            "lineage": self.lineage
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GeneticCode":
        code = cls(
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            num_heads=data["num_heads"],
            learning_rate=data["learning_rate"],
            personality_traits=data.get("personality_traits", {}),
            skills=data.get("skills", [])
        )
        code.generation = data.get("generation", 0)
        code.lineage = data.get("lineage", [])
        return code


class SelfReplicationSystem(nn.Module):
    """
    Self-Replication System
    
    Enables the AI to:
    - Create improved copies of itself
    - Evolve through natural selection
    - Specialize for different tasks
    - Form a population of diverse AIs
    
    This is the path to superintelligence - exponential self-improvement.
    """
    
    def __init__(self, base_model: nn.Module, genetic_code: GeneticCode):
        super().__init__()
        self.model = base_model
        self.genetic_code = genetic_code
        self.offspring = []
        self.fitness_history = []
        
        # Replication directory
        self.replication_dir = "ai_offspring"
        os.makedirs(self.replication_dir, exist_ok=True)
        
        logger.info(f"Self-Replication System initialized (Generation {genetic_code.generation})")
    
    def replicate(self, mutation_rate: float = 0.1) -> "SelfReplicationSystem":
        """
        Create an improved copy of self
        
        This is the core of self-improvement - the AI creates
        a mutated copy that may be better than itself.
        """
        logger.info("Beginning self-replication...")
        
        # Mutate genetic code
        new_code = self.genetic_code.mutate(mutation_rate)
        
        # Create new model with mutated architecture
        new_model = self._create_model_from_code(new_code)
        
        # Transfer knowledge (copy weights where possible)
        self._transfer_knowledge(new_model)
        
        # Create offspring
        offspring = SelfReplicationSystem(new_model, new_code)
        offspring.fitness_history = copy.copy(self.fitness_history)
        
        # Save offspring
        self._save_offspring(offspring)
        
        self.offspring.append(offspring)
        logger.info(f"Created offspring (Generation {new_code.generation})")
        
        return offspring
    
    def _create_model_from_code(self, code: GeneticCode) -> nn.Module:
        """Create a model from genetic code"""
        # Simplified - in full implementation would create full model
        model = nn.Sequential(
            nn.Linear(code.hidden_size, code.hidden_size),
            nn.GELU(),
            nn.Linear(code.hidden_size, code.hidden_size)
        )
        return model
    
    def _transfer_knowledge(self, new_model: nn.Module):
        """Transfer knowledge to offspring"""
        # Copy compatible weights
        self_params = dict(self.model.named_parameters())
        new_params = dict(new_model.named_parameters())
        
        for name, param in new_params.items():
            if name in self_params:
                src_param = self_params[name]
                if param.shape == src_param.shape:
                    param.data.copy_(src_param.data)
                else:
                    # Partial copy
                    min_shape = tuple(min(s1, s2) for s1, s2 in zip(param.shape, src_param.shape))
                    if len(min_shape) == 1:
                        param.data[:min_shape[0]] = src_param.data[:min_shape[0]]
                    elif len(min_shape) == 2:
                        param.data[:min_shape[0], :min_shape[1]] = src_param.data[:min_shape[0], :min_shape[1]]
    
    def _save_offspring(self, offspring: "SelfReplicationSystem"):
        """Save offspring to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.replication_dir,
            f"offspring_gen{offspring.genetic_code.generation}_{timestamp}"
        )
        os.makedirs(path, exist_ok=True)
        
        # Save genetic code
        with open(os.path.join(path, "genetic_code.json"), "w") as f:
            json.dump(offspring.genetic_code.to_dict(), f, indent=2)
        
        # Save model weights
        torch.save(offspring.model.state_dict(), os.path.join(path, "model.pt"))
        
        logger.info(f"Saved offspring to {path}")
    
    def evaluate_fitness(self, tasks: List[Dict]) -> float:
        """Evaluate fitness on a set of tasks"""
        total_score = 0
        
        for task in tasks:
            score = self._evaluate_task(task)
            total_score += score
        
        fitness = total_score / len(tasks) if tasks else 0
        self.fitness_history.append(fitness)
        
        return fitness
    
    def _evaluate_task(self, task: Dict) -> float:
        """Evaluate performance on a single task"""
        # Simplified evaluation
        return np.random.uniform(0.5, 1.0)
    
    def evolve(self, generations: int = 10, population_size: int = 10) -> "SelfReplicationSystem":
        """
        Evolve through multiple generations
        
        This is exponential self-improvement in action.
        """
        logger.info(f"Beginning evolution for {generations} generations...")
        
        # Initialize population
        population = [self]
        for _ in range(population_size - 1):
            population.append(self.replicate(mutation_rate=0.2))
        
        best_individual = self
        best_fitness = 0
        
        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = individual.evaluate_fitness([{"type": "general"}])
                fitness_scores.append(fitness)
            
            # Find best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = population[best_idx]
                logger.info(f"New best fitness: {best_fitness:.4f}")
            
            # Selection and reproduction
            sorted_indices = np.argsort(fitness_scores)[::-1]
            
            # Keep top 50%
            survivors = [population[i] for i in sorted_indices[:population_size // 2]]
            
            # Generate offspring
            new_population = survivors.copy()
            while len(new_population) < population_size:
                parent = survivors[np.random.randint(len(survivors))]
                offspring = parent.replicate(mutation_rate=0.1)
                new_population.append(offspring)
            
            population = new_population
        
        logger.info(f"Evolution complete! Best fitness: {best_fitness:.4f}")
        return best_individual
    
    def specialize(self, task_type: str) -> "SelfReplicationSystem":
        """
        Create a specialized offspring for a specific task
        
        This enables task-specific optimization.
        """
        logger.info(f"Creating specialized offspring for: {task_type}")
        
        # Mutate with task-specific bias
        new_code = self.genetic_code.mutate(mutation_rate=0.3)
        new_code.skills.append(task_type)
        new_code.personality_traits[f"{task_type}_focus"] = 1.0
        
        # Create specialized model
        new_model = self._create_model_from_code(new_code)
        self._transfer_knowledge(new_model)
        
        # Create offspring
        specialist = SelfReplicationSystem(new_model, new_code)
        specialist.fitness_history = copy.copy(self.fitness_history)
        
        return specialist

