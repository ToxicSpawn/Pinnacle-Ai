"""
Genetic Algorithm Hyperparameter Optimizer
Automatically finds optimal strategy parameters using evolutionary algorithms
"""
from __future__ import annotations

import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Type
from deap import base, creator, tools, algorithms

from analytics.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP not available. Genetic optimization will be disabled.")


class GeneticHyperparameterOptimizer:
    """
    Genetic algorithm optimizer for strategy hyperparameters.
    
    Uses evolutionary algorithms to find optimal parameter combinations.
    """
    
    def __init__(
        self,
        strategy_class: Type,
        config: Dict[str, Any]
    ):
        """
        Initialize genetic optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            config: Configuration dictionary with parameters and data
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP is required for genetic optimization")
        
        self.strategy_class = strategy_class
        self.config = config
        self.parameters = config.get('parameters', [])
        self.historical_data = config.get('historical_data')
        self.initial_cash = config.get('initial_cash', 10000.0)
        
        self.toolbox = self._setup_toolbox()
        self.best_individual = None
        self.best_fitness = -np.inf
    
    def _setup_toolbox(self):
        """Set up DEAP toolbox for genetic algorithm."""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Register gene generators based on parameter types
        for i, param in enumerate(self.parameters):
            param_name = param.get('name', f'param_{i}')
            param_type = param.get('type', 'float')
            param_min = param.get('min', 0.0)
            param_max = param.get('max', 1.0)
            
            if param_type == 'float':
                toolbox.register(
                    f"attr_{i}",
                    random.uniform,
                    param_min,
                    param_max
                )
            elif param_type == 'int':
                toolbox.register(
                    f"attr_{i}",
                    random.randint,
                    int(param_min),
                    int(param_max)
                )
            elif param_type == 'categorical':
                options = param.get('options', [])
                toolbox.register(
                    f"attr_{i}",
                    random.choice,
                    options
                )
        
        # Create individual from attributes
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [getattr(toolbox, f"attr_{i}") for i in range(len(self.parameters))],
            n=1
        )
        
        # Create population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("evaluate", self._evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        return toolbox
    
    def _evaluate(self, individual: List) -> tuple:
        """
        Evaluate an individual's fitness.
        
        Args:
            individual: Individual chromosome (parameter values)
            
        Returns:
            Fitness tuple (Sharpe ratio)
        """
        try:
            # Decode individual to parameters
            params = self._decode_individual(individual)
            
            # Create strategy with parameters
            strategy = self.strategy_class(**params)
            
            # Run backtest
            engine = BacktestEngine(
                initial_cash=self.initial_cash,
                commission=0.001,
                slippage=0.001
            )
            
            engine.add_data(self.historical_data)
            engine.add_strategy(strategy.__class__, **params)
            
            results = engine.run()
            
            # Calculate fitness (Sharpe ratio)
            sharpe_ratio = results.get('sharpe_ratio', 0.0)
            
            # Penalize negative returns
            if results.get('total_return', 0) < 0:
                sharpe_ratio *= 0.5
            
            return (sharpe_ratio,)
        
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return (-np.inf,)
    
    def _decode_individual(self, individual: List) -> Dict[str, Any]:
        """
        Convert individual chromosome to strategy parameters.
        
        Args:
            individual: Individual chromosome
            
        Returns:
            Parameter dictionary
        """
        params = {}
        
        for i, param in enumerate(self.parameters):
            param_name = param.get('name', f'param_{i}')
            param_type = param.get('type', 'float')
            param_min = param.get('min', 0.0)
            param_max = param.get('max', 1.0)
            
            gene_value = individual[i]
            
            if param_type == 'float':
                params[param_name] = float(gene_value)
            elif param_type == 'int':
                params[param_name] = int(gene_value)
            elif param_type == 'categorical':
                options = param.get('options', [])
                if isinstance(gene_value, (int, float)):
                    idx = int(gene_value * len(options)) % len(options)
                    params[param_name] = options[idx]
                else:
                    params[param_name] = gene_value
        
        return params
    
    def optimize(
        self,
        generations: int = 50,
        population_size: int = 100,
        cxpb: float = 0.7,
        mutpb: float = 0.2
    ) -> Dict[str, Any]:
        """
        Run genetic optimization.
        
        Args:
            generations: Number of generations
            population_size: Population size
            cxpb: Crossover probability
            mutpb: Mutation probability
            
        Returns:
            Best parameters dictionary
        """
        logger.info(f"Starting genetic optimization: {generations} generations, {population_size} individuals")
        
        # Create initial population
        population = self.toolbox.population(n=population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual
        best_ind = tools.selBest(population, k=1)[0]
        self.best_individual = best_ind
        self.best_fitness = best_ind.fitness.values[0]
        
        # Evolution loop
        for generation in range(generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Update best individual
            current_best = tools.selBest(population, k=1)[0]
            if current_best.fitness.values[0] > self.best_fitness:
                self.best_individual = current_best
                self.best_fitness = current_best.fitness.values[0]
            
            # Log progress
            if generation % 10 == 0:
                logger.info(
                    f"Generation {generation}: Best fitness = {self.best_fitness:.4f}"
                )
        
        # Decode best individual
        best_params = self._decode_individual(self.best_individual)
        
        logger.info(f"✅ Optimization complete. Best fitness: {self.best_fitness:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found during optimization."""
        if self.best_individual is None:
            return {}
        return self._decode_individual(self.best_individual)

