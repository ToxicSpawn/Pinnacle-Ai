from typing import Dict, List, Optional
from loguru import logger
import numpy as np
import copy


class GeneticCode:
    """Genetic code representing AI configuration"""
    
    def __init__(self):
        self.genes = {
            "temperature": 0.7,
            "creativity": 0.5,
            "reasoning_depth": 0.7,
            "memory_weight": 0.5,
            "emotional_sensitivity": 0.5
        }
        self.generation = 0
        self.fitness = 0.0
    
    def mutate(self, rate: float = 0.1) -> "GeneticCode":
        """Create mutated copy"""
        child = GeneticCode()
        child.genes = copy.copy(self.genes)
        child.generation = self.generation + 1
        
        for gene in child.genes:
            if np.random.random() < rate:
                mutation = np.random.normal(0, 0.1)
                child.genes[gene] = np.clip(child.genes[gene] + mutation, 0, 1)
        
        return child
    
    def crossover(self, other: "GeneticCode") -> "GeneticCode":
        """Crossover with another genetic code"""
        child = GeneticCode()
        child.generation = max(self.generation, other.generation) + 1
        
        for gene in self.genes:
            if np.random.random() < 0.5:
                child.genes[gene] = self.genes[gene]
            else:
                child.genes[gene] = other.genes[gene]
        
        return child


class SelfEvolution:
    """
    Self-Evolution System
    
    Enables the AI to improve itself through:
    - Genetic algorithms
    - Performance-based selection
    - Continuous optimization
    """
    
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[GeneticCode] = []
        self.best_individual: Optional[GeneticCode] = None
        self.evolution_history = []
        
        # Initialize population
        self._init_population()
        
        logger.info(f"Self-Evolution initialized (population: {population_size})")
    
    def _init_population(self):
        """Initialize random population"""
        self.population = [GeneticCode() for _ in range(self.population_size)]
    
    def _evaluate_fitness(self, individual: GeneticCode) -> float:
        """Evaluate fitness of an individual"""
        # Simple fitness function based on gene balance
        fitness = 0.0
        
        # Reward balanced genes
        for gene, value in individual.genes.items():
            if 0.3 <= value <= 0.7:
                fitness += 0.2
            else:
                fitness += 0.1
        
        # Add some randomness for diversity
        fitness += np.random.uniform(0, 0.3)
        
        individual.fitness = fitness
        return fitness
    
    def evolve(self, generations: int = 10) -> Dict:
        """
        Run evolution for multiple generations
        
        Args:
            generations: Number of generations
        
        Returns:
            Evolution results
        """
        logger.info(f"Starting evolution for {generations} generations...")
        
        initial_best = 0.0
        
        for gen in range(generations):
            # Evaluate fitness
            for individual in self.population:
                self._evaluate_fitness(individual)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = copy.copy(self.population[0])
            
            if gen == 0:
                initial_best = self.population[0].fitness
            
            logger.info(f"Generation {gen + 1}: Best fitness = {self.population[0].fitness:.4f}")
            
            # Selection and reproduction
            survivors = self.population[:self.population_size // 2]
            
            # Generate offspring
            offspring = []
            while len(offspring) < self.population_size - len(survivors):
                parent1, parent2 = np.random.choice(survivors, 2, replace=False)
                child = parent1.crossover(parent2)
                child = child.mutate(self.mutation_rate)
                offspring.append(child)
            
            self.population = survivors + offspring
            
            # Record history
            self.evolution_history.append({
                "generation": gen + 1,
                "best_fitness": self.population[0].fitness,
                "avg_fitness": np.mean([i.fitness for i in self.population])
            })
        
        improvement = (self.best_individual.fitness - initial_best) / max(initial_best, 0.01) * 100
        
        return {
            "generations": generations,
            "initial_fitness": initial_best,
            "final_fitness": self.best_individual.fitness,
            "improvement": f"{improvement:.2f}%",
            "best_genes": self.best_individual.genes,
            "history": self.evolution_history
        }
    
    def get_best(self) -> Optional[GeneticCode]:
        """Get best individual"""
        return self.best_individual

