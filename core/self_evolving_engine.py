"""
Self-Evolving Strategy Engine
Automatically evolves and optimizes trading strategies
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import strategy components
try:
    from strategies.ai.regime_detector import MarketRegimeDetector
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    logger.warning("Regime detector not available")

try:
    from optimization.genetic_optimizer import GeneticHyperparameterOptimizer
    GENETIC_OPTIMIZER_AVAILABLE = True
except ImportError:
    GENETIC_OPTIMIZER_AVAILABLE = False
    logger.warning("Genetic optimizer not available")

try:
    from validation.backtest_validator import BacktestValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    logger.warning("Backtest validator not available")


class SelfEvolvingStrategyEngine:
    """
    Self-evolving strategy engine that automatically evolves strategies.
    
    Features:
    - Strategy initialization
    - Performance tracking
    - Automatic evolution
    - Strategy validation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize self-evolving strategy engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.strategies: Dict[str, Any] = {}
        self.current_strategy: Optional[str] = None
        self.performance_history: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.market_regime = "neutral"
        
        # Initialize components
        if REGIME_DETECTOR_AVAILABLE:
            try:
                self.regime_detector = MarketRegimeDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize regime detector: {e}")
                self.regime_detector = None
        else:
            self.regime_detector = None
        
        if VALIDATOR_AVAILABLE:
            try:
                self.validator = BacktestValidator(config.get('validation', {}))
            except Exception as e:
                logger.warning(f"Failed to initialize validator: {e}")
                self.validator = None
        else:
            self.validator = None
        
        self.last_optimization = 0
        self.last_validation = 0
    
    def initialize(self):
        """Initialize the strategy engine."""
        logger.info("Initializing self-evolving strategy engine...")
        
        # Load initial strategies
        self._load_initial_strategies()
        
        # Select initial strategy
        self.current_strategy = self._select_initial_strategy()
        
        logger.info(f"✅ Strategy engine initialized. Current strategy: {self.current_strategy}")
    
    def _load_initial_strategies(self):
        """Load initial set of strategies."""
        # Market making
        if 'market_making' in self.config:
            try:
                from strategies.market_making import MarketMakingStrategy
                self.strategies["market_making"] = MarketMakingStrategy(
                    self.config['market_making']
                )
            except Exception as e:
                logger.warning(f"Failed to load market making strategy: {e}")
        
        # Arbitrage
        if 'arbitrage' in self.config:
            try:
                from strategies.enhanced_arbitrage import EnhancedArbitrageStrategy
                self.strategies["arbitrage"] = EnhancedArbitrageStrategy(
                    None,  # Exchange will be set later
                    **self.config['arbitrage']
                )
            except Exception as e:
                logger.warning(f"Failed to load arbitrage strategy: {e}")
        
        # Trend following
        if 'trend_following' in self.config:
            try:
                from strategies.trend_vol_strategy import TrendVolStrategy
                self.strategies["trend_following"] = TrendVolStrategy(
                    self.config['trend_following']
                )
            except Exception as e:
                logger.warning(f"Failed to load trend following strategy: {e}")
        
        # Mean reversion
        if 'mean_reversion' in self.config:
            try:
                from strategies.rsi_ml_strategy import RSIMLStrategy
                self.strategies["mean_reversion"] = RSIMLStrategy(
                    self.config['mean_reversion']
                )
            except Exception as e:
                logger.warning(f"Failed to load mean reversion strategy: {e}")
        
        # AI trader
        if 'ai_trader' in self.config:
            try:
                from strategies.ai.neuro_evolution import NeuroEvolutionaryTrader
                self.strategies["ai_trader"] = NeuroEvolutionaryTrader(
                    input_shape=(30, 5),
                    output_size=3
                )
            except Exception as e:
                logger.warning(f"Failed to load AI trader: {e}")
        
        logger.info(f"Loaded {len(self.strategies)} strategies")
    
    def _select_initial_strategy(self) -> str:
        """Select initial strategy based on market conditions."""
        if self.regime_detector is None:
            # Default to first available strategy
            return list(self.strategies.keys())[0] if self.strategies else None
        
        try:
            # Get market data (placeholder)
            market_data = self._get_market_data()
            self.market_regime = self.regime_detector.detect_regime(market_data)
            
            # Select strategy based on regime
            if self.market_regime == "bull":
                return "trend_following" if "trend_following" in self.strategies else list(self.strategies.keys())[0]
            elif self.market_regime == "bear":
                return "mean_reversion" if "mean_reversion" in self.strategies else list(self.strategies.keys())[0]
            elif self.market_regime == "sideways":
                return "market_making" if "market_making" in self.strategies else list(self.strategies.keys())[0]
            else:
                return "ai_trader" if "ai_trader" in self.strategies else list(self.strategies.keys())[0]
        except Exception as e:
            logger.warning(f"Failed to select initial strategy: {e}")
            return list(self.strategies.keys())[0] if self.strategies else None
    
    def step(self, market_data: Dict) -> Dict:
        """
        Execute one trading step.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signals
        """
        # Update market regime
        if self.regime_detector:
            try:
                self.market_regime = self.regime_detector.detect_regime(market_data)
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")
        
        # Select best strategy for current regime
        best_strategy = self._select_strategy()
        
        # Execute strategy
        if best_strategy and best_strategy in self.strategies:
            strategy = self.strategies[best_strategy]
            try:
                if hasattr(strategy, 'generate_signals'):
                    signals = strategy.generate_signals(market_data)
                elif hasattr(strategy, 'predict'):
                    # For ML models
                    signals = self._ml_strategy_to_signals(strategy, market_data)
                else:
                    signals = {}
                
                # Record performance
                self._record_performance(best_strategy, market_data)
                
                return signals
            except Exception as e:
                logger.error(f"Strategy execution failed: {e}")
                return {}
        
        return {}
    
    def _select_strategy(self) -> Optional[str]:
        """Select best strategy for current market regime."""
        if self.market_regime not in self.performance_history:
            self.performance_history[self.market_regime] = {
                strategy: [] for strategy in self.strategies
            }
        
        # Select strategy with best performance in this regime
        best_strategy = None
        best_performance = -np.inf
        
        for strategy_name in self.strategies.keys():
            if self.performance_history[self.market_regime][strategy_name]:
                avg_performance = np.mean(
                    self.performance_history[self.market_regime][strategy_name]
                )
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_strategy = strategy_name
        
        # Default to current strategy if no data
        if best_strategy is None:
            best_strategy = self.current_strategy or list(self.strategies.keys())[0]
        
        # Switch strategy if needed
        if best_strategy != self.current_strategy:
            self._switch_strategy(best_strategy)
        
        return best_strategy
    
    def _switch_strategy(self, new_strategy: str):
        """Switch to a new strategy."""
        if self.current_strategy:
            old_strategy = self.strategies.get(self.current_strategy)
            if old_strategy and hasattr(old_strategy, 'stop'):
                try:
                    old_strategy.stop()
                except Exception as e:
                    logger.warning(f"Error stopping strategy {self.current_strategy}: {e}")
        
        self.current_strategy = new_strategy
        new_strategy_obj = self.strategies.get(new_strategy)
        
        if new_strategy_obj and hasattr(new_strategy_obj, 'start'):
            try:
                new_strategy_obj.start()
            except Exception as e:
                logger.warning(f"Error starting strategy {new_strategy}: {e}")
        
        logger.info(f"Switched to strategy: {new_strategy}")
    
    def _record_performance(self, strategy: str, market_data: Dict):
        """Record strategy performance."""
        # Calculate performance metric (Sharpe ratio placeholder)
        performance = self._calculate_performance(strategy, market_data)
        
        # Record performance
        self.performance_history[self.market_regime][strategy].append(performance)
        
        # Keep only recent performance
        if len(self.performance_history[self.market_regime][strategy]) > 100:
            self.performance_history[self.market_regime][strategy].pop(0)
    
    def _calculate_performance(self, strategy: str, market_data: Dict) -> float:
        """Calculate strategy performance metric."""
        # Placeholder: would calculate actual Sharpe ratio from trades
        return 0.0
    
    def _ml_strategy_to_signals(self, model: Any, market_data: Dict) -> Dict:
        """Convert ML model predictions to trading signals."""
        # Placeholder: would convert model predictions to signals
        return {}
    
    def _get_market_data(self) -> Dict:
        """Get current market data."""
        # Placeholder: would fetch actual market data
        return {}
    
    def evolve_strategies(self):
        """Evolve strategies using genetic algorithms."""
        if not GENETIC_OPTIMIZER_AVAILABLE:
            logger.warning("Genetic optimizer not available")
            return
        
        logger.info("Starting strategy evolution...")
        
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'get_parameters'):
                    # Optimize strategy parameters
                    params = strategy.get_parameters()
                    
                    optimizer = GeneticHyperparameterOptimizer(
                        strategy.__class__,
                        {
                            'historical_data': self._get_historical_data(),
                            'initial_cash': self.config.get('initial_cash', 10000),
                            'parameters': params
                        }
                    )
                    
                    best_params = optimizer.optimize(
                        generations=self.config.get('optimization', {}).get('generations', 20),
                        population_size=self.config.get('optimization', {}).get('population_size', 50)
                    )
                    
                    # Update strategy with best parameters
                    if hasattr(strategy, 'update_parameters'):
                        strategy.update_parameters(best_params)
                    
                    logger.info(f"Optimized strategy {name} with parameters: {best_params}")
                
                if hasattr(strategy, 'evolve'):
                    # Evolve AI strategy
                    training_data = self._get_training_data()
                    if training_data:
                        strategy.evolve(
                            *training_data,
                            generations=self.config.get('optimization', {}).get('generations', 20)
                        )
                        logger.info(f"Evolved AI strategy {name}")
            except Exception as e:
                logger.error(f"Failed to evolve strategy {name}: {e}")
        
        self.last_optimization = time.time()
    
    def validate_strategies(self) -> Dict:
        """Validate all strategies."""
        if not self.validator:
            logger.warning("Validator not available")
            return {}
        
        results = {}
        
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'get_parameters'):
                    params = strategy.get_parameters()
                else:
                    params = {}
                
                # Validate strategy
                results[name] = self.validator.validate(strategy.__class__, params)
            except Exception as e:
                logger.error(f"Failed to validate strategy {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.last_validation = time.time()
        return results
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical market data."""
        # Placeholder: would query database
        return pd.DataFrame()
    
    def _get_training_data(self) -> Optional[tuple]:
        """Get training data for AI strategies."""
        # Placeholder: would query database
        return None

