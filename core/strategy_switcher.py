"""
Dynamic Strategy Switcher
Automatically switches strategies based on market regime
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from strategies.ai.regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)


class DynamicStrategySwitcher:
    """
    Dynamic strategy switcher based on market regime.
    
    Automatically selects the best strategy for current market conditions.
    """
    
    def __init__(
        self,
        strategies: List,
        regime_detector: MarketRegimeDetector,
        lookback_periods: int = 100
    ):
        """
        Initialize strategy switcher.
        
        Args:
            strategies: List of strategy instances
            regime_detector: Market regime detector
            lookback_periods: Number of periods to track performance
        """
        self.strategies = {strategy.name if hasattr(strategy, 'name') else str(i): strategy 
                          for i, strategy in enumerate(strategies)}
        self.regime_detector = regime_detector
        self.lookback_periods = lookback_periods
        
        self.current_strategy: Optional[str] = None
        self.performance_history: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def select_strategy(self, market_data) -> str:
        """
        Select best strategy based on market regime.
        
        Args:
            market_data: Current market data
            
        Returns:
            Selected strategy name
        """
        # Detect current market regime
        try:
            regime, confidence = self.regime_detector.detect_regime(market_data)
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Using default strategy.")
            regime = 'sideways'
            confidence = 0.5
        
        # Get performance by regime
        if regime not in self.performance_history:
            self.performance_history[regime] = {
                strategy_name: [] for strategy_name in self.strategies.keys()
            }
        
        # Select strategy with best performance in this regime
        best_strategy = None
        best_performance = -np.inf
        
        for strategy_name in self.strategies.keys():
            perf_history = self.performance_history[regime][strategy_name]
            
            if perf_history:
                # Use average performance
                avg_performance = np.mean(perf_history)
                
                # Weight by confidence
                weighted_performance = avg_performance * confidence
                
                if weighted_performance > best_performance:
                    best_performance = weighted_performance
                    best_strategy = strategy_name
        
        # Default to first strategy if no data
        if best_strategy is None:
            best_strategy = list(self.strategies.keys())[0]
            logger.info(f"No performance data. Using default strategy: {best_strategy}")
        
        # Switch strategy if needed
        if best_strategy != self.current_strategy:
            self._switch_strategy(best_strategy)
        
        return best_strategy
    
    def _switch_strategy(self, new_strategy: str) -> None:
        """
        Switch to new strategy.
        
        Args:
            new_strategy: Name of new strategy
        """
        if self.current_strategy:
            old_strategy = self.strategies[self.current_strategy]
            if hasattr(old_strategy, 'stop'):
                try:
                    old_strategy.stop()
                except Exception as e:
                    logger.warning(f"Error stopping strategy {self.current_strategy}: {e}")
        
        self.current_strategy = new_strategy
        new_strategy_obj = self.strategies[new_strategy]
        
        if hasattr(new_strategy_obj, 'start'):
            try:
                new_strategy_obj.start()
            except Exception as e:
                logger.warning(f"Error starting strategy {new_strategy}: {e}")
        
        logger.info(f"✅ Switched to strategy: {new_strategy}")
    
    def update_performance(
        self,
        strategy_name: str,
        performance: float,
        market_data
    ) -> None:
        """
        Update strategy performance history.
        
        Args:
            strategy_name: Name of strategy
            performance: Performance metric (e.g., return, Sharpe ratio)
            market_data: Market data for regime detection
        """
        try:
            regime, _ = self.regime_detector.detect_regime(market_data)
        except Exception:
            regime = 'sideways'
        
        self.performance_history[regime][strategy_name].append(performance)
        
        # Keep only recent performance
        if len(self.performance_history[regime][strategy_name]) > self.lookback_periods:
            self.performance_history[regime][strategy_name].pop(0)
    
    def get_strategy_performance_summary(self) -> Dict:
        """
        Get summary of strategy performance by regime.
        
        Returns:
            Performance summary dictionary
        """
        summary = {}
        
        for regime, strategies in self.performance_history.items():
            summary[regime] = {}
            for strategy_name, perf_history in strategies.items():
                if perf_history:
                    summary[regime][strategy_name] = {
                        'mean': np.mean(perf_history),
                        'std': np.std(perf_history),
                        'count': len(perf_history)
                    }
        
        return summary
    
    def get_current_strategy(self) -> Optional[str]:
        """Get current strategy name."""
        return self.current_strategy

