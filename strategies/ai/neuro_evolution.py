"""
Neuro-Evolutionary Trading Engine
Evolves neural network architectures and weights using genetic algorithms
"""
from __future__ import annotations

import logging
import numpy as np
import random
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neuro-evolution disabled.")

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP not available. Neuro-evolution disabled.")


class NeuroEvolutionaryTrader:
    """
    Neuro-evolutionary trader that evolves neural network architectures.
    
    Features:
    - Genetic algorithm for architecture search
    - Neural network evolution
    - Fitness evaluation based on trading performance
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_size: int,
        population_size: int = 50
    ):
        """
        Initialize neuro-evolutionary trader.
        
        Args:
            input_shape: Input shape for neural network
            output_size: Output size (number of actions)
            population_size: Population size for genetic algorithm
        """
        if not TENSORFLOW_AVAILABLE or not DEAP_AVAILABLE:
            raise ImportError("TensorFlow and DEAP are required for neuro-evolution")
        
        self.input_shape = input_shape
        self.output_size = output_size
        self.population_size = population_size
        self.best_model = None
        self.best_fitness = -np.inf
        
        # Setup genetic algorithm toolbox
        self.toolbox = self._setup_toolbox()
        
        # Training data (set during evolve)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.test_returns = None
    
    def _setup_toolbox(self):
        """Set up genetic algorithm toolbox."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Register genetic operators
        toolbox.register("attr_float", random.random)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=100  # 100 genes per individual
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self._evaluate_individual)
        
        return toolbox
    
    def _build_model(self, individual: List[float]) -> Optional[Sequential]:
        """Build neural network from genetic individual."""
        try:
            model = Sequential()
            
            # Decode individual into layer parameters
            idx = 0
            num_layers = int(individual[idx] * 5) + 1  # 1-5 layers
            idx += 1
            
            first_layer = True
            for _ in range(num_layers):
                if idx >= len(individual):
                    break
                
                # Layer type
                layer_type = int(individual[idx] * 3) if idx < len(individual) else 0
                idx += 1
                
                if idx >= len(individual):
                    break
                
                # Layer size
                layer_size = int(individual[idx] * 128) + 32 if idx < len(individual) else 64
                idx += 1
                
                # Add layer
                if layer_type == 0:  # Dense
                    if first_layer:
                        model.add(Dense(
                            layer_size,
                            activation='relu',
                            input_shape=self.input_shape
                        ))
                        first_layer = False
                    else:
                        model.add(Dense(layer_size, activation='relu'))
                elif layer_type == 1:  # LSTM
                    if isinstance(self.input_shape, tuple) and len(self.input_shape) == 2:
                        if first_layer:
                            model.add(LSTM(
                                layer_size,
                                return_sequences=(_ < num_layers - 1),
                                input_shape=self.input_shape
                            ))
                            first_layer = False
                        else:
                            model.add(LSTM(
                                layer_size,
                                return_sequences=(_ < num_layers - 1)
                            ))
                    else:
                        # Fallback to Dense if LSTM not applicable
                        if first_layer:
                            model.add(Dense(
                                layer_size,
                                activation='relu',
                                input_shape=self.input_shape
                            ))
                            first_layer = False
                        else:
                            model.add(Dense(layer_size, activation='relu'))
                elif layer_type == 2:  # Dropout
                    if idx < len(individual):
                        dropout_rate = individual[idx]
                        model.add(Dropout(min(max(dropout_rate, 0.0), 0.5)))
                        idx += 1
                    continue
                
                # Batch normalization
                if idx < len(individual) and individual[idx] > 0.5:
                    model.add(BatchNormalization())
                idx += 1
            
            # Output layer
            model.add(Dense(self.output_size, activation='softmax'))
            
            # Compile model
            if idx < len(individual):
                learning_rate = individual[idx] * 0.01
            else:
                learning_rate = 0.001
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to build model: {e}")
            return None
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate an individual's fitness."""
        try:
            model = self._build_model(individual)
            if model is None:
                return (-np.inf,)
            
            if self.X_train is None or self.y_train is None:
                return (-np.inf,)
            
            # Train model (limited epochs for speed)
            try:
                history = model.fit(
                    self.X_train,
                    self.y_train,
                    validation_data=(
                        (self.X_val, self.y_val) if self.X_val is not None else None
                    ),
                    epochs=5,
                    batch_size=32,
                    verbose=0
                )
                
                # Calculate fitness (validation accuracy + Sharpe ratio)
                val_accuracy = (
                    history.history['val_accuracy'][-1]
                    if 'val_accuracy' in history.history
                    else history.history['accuracy'][-1]
                )
                sharpe_ratio = self._calculate_sharpe_ratio(model)
                
                return (val_accuracy + sharpe_ratio,)
                
            except Exception as e:
                logger.warning(f"Model training failed: {e}")
                return (-np.inf,)
                
        except Exception as e:
            logger.warning(f"Individual evaluation failed: {e}")
            return (-np.inf,)
    
    def _calculate_sharpe_ratio(self, model) -> float:
        """Calculate Sharpe ratio for model predictions."""
        if self.X_test is None or self.test_returns is None:
            return 0.0
        
        try:
            # Get model predictions
            predictions = model.predict(self.X_test, verbose=0)
            predicted_actions = np.argmax(predictions, axis=1)
            
            # Calculate returns
            returns = []
            for i, action in enumerate(predicted_actions):
                if i < len(self.test_returns):
                    if action == 0:  # Buy
                        returns.append(self.test_returns[i])
                    elif action == 1:  # Sell
                        returns.append(-self.test_returns[i])
                    else:  # Hold
                        returns.append(0.0)
            
            # Calculate Sharpe ratio
            returns = np.array(returns)
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            
            return float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            
        except Exception as e:
            logger.warning(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def evolve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        test_returns: Optional[np.ndarray] = None,
        generations: int = 20
    ) -> Optional[Sequential]:
        """
        Run neuro-evolutionary optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            test_returns: Test returns for Sharpe calculation
            generations: Number of generations
            
        Returns:
            Best evolved model
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.test_returns = test_returns
        
        # Create population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Run evolution
        logger.info(f"Starting neuro-evolution with {generations} generations...")
        algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=0.7,
            mutpb=0.2,
            ngen=generations,
            verbose=True
        )
        
        # Get best individual
        best_ind = tools.selBest(population, k=1)[0]
        self.best_model = self._build_model(best_ind)
        self.best_fitness = best_ind.fitness.values[0]
        
        logger.info(f"Neuro-evolution completed. Best fitness: {self.best_fitness:.4f}")
        
        return self.best_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the best model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call evolve() first.")
        
        return self.best_model.predict(X, verbose=0)

