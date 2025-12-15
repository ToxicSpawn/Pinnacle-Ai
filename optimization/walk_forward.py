"""
Walk-Forward Optimization
Robust optimization technique that tests on out-of-sample data
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Type
from datetime import datetime, timedelta

from optimization.genetic_optimizer import GeneticHyperparameterOptimizer
from analytics.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward optimizer for robust strategy validation.
    
    Splits data into training and testing periods, optimizing on training
    and validating on out-of-sample data.
    """
    
    def __init__(
        self,
        strategy_class: Type,
        config: Dict[str, Any]
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            config: Configuration dictionary
        """
        self.strategy_class = strategy_class
        self.config = config
        self.historical_data = config.get('historical_data')
        self.initial_cash = config.get('initial_cash', 10000.0)
        self.train_periods = config.get('train_periods', 6)  # months
        self.test_periods = config.get('test_periods', 1)  # months
        self.step_size = config.get('step_size', 1)  # months
        
        self.optimizer = GeneticHyperparameterOptimizer(
            strategy_class,
            {
                'parameters': config.get('parameters', []),
                'historical_data': None,  # Will be set per iteration
                'initial_cash': self.initial_cash
            }
        )
        
        self.results: List[Dict] = []
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        Returns:
            Optimization results dictionary
        """
        logger.info("Starting walk-forward optimization")
        
        # Get split points
        split_points = self._get_split_points()
        
        logger.info(f"Split into {len(split_points) - 1} periods")
        
        # Run optimization for each period
        for i in range(len(split_points) - 2):  # -2 to ensure we have test data
            train_start = split_points[i]
            train_end = split_points[i + 1]
            test_start = split_points[i + 1]
            test_end = split_points[i + 2] if i + 2 < len(split_points) else split_points[-1]
            
            logger.info(
                f"Period {i + 1}: Train {train_start} to {train_end}, "
                f"Test {test_start} to {test_end}"
            )
            
            # Get training and testing data
            train_data = self._get_data_slice(train_start, train_end)
            test_data = self._get_data_slice(test_start, test_end)
            
            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning(f"Skipping period {i + 1}: insufficient data")
                continue
            
            # Optimize on training data
            self.optimizer.config['historical_data'] = train_data
            best_params = self.optimizer.optimize(
                generations=self.config.get('generations', 20),
                population_size=self.config.get('population_size', 50)
            )
            
            # Test on out-of-sample data
            strategy = self.strategy_class(**best_params)
            engine = BacktestEngine(
                initial_cash=self.initial_cash,
                commission=0.001,
                slippage=0.001
            )
            
            engine.add_data(test_data)
            engine.add_strategy(strategy.__class__, **best_params)
            
            test_results = engine.run()
            
            # Store results
            self.results.append({
                'period': i + 1,
                'params': best_params,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'train_performance': self.optimizer.best_fitness,
                'test_performance': test_results,
                'params_stability': self._calculate_param_stability(best_params)
            })
        
        # Analyze results
        analysis = self._analyze_results()
        
        logger.info("✅ Walk-forward optimization complete")
        
        return analysis
    
    def _get_split_points(self) -> List[datetime]:
        """Generate split points for walk-forward optimization."""
        if not isinstance(self.historical_data.index, pd.DatetimeIndex):
            # If not datetime index, use numeric splits
            total_len = len(self.historical_data)
            train_len = int(total_len * 0.7)  # 70% for training
            test_len = int(total_len * 0.15)  # 15% for testing
            
            split_points = []
            current = 0
            while current < total_len:
                split_points.append(current)
                current += train_len
            
            if split_points[-1] < total_len:
                split_points.append(total_len)
            
            return split_points
        
        # Datetime-based splits
        split_points = []
        current_date = self.historical_data.index[0]
        end_date = self.historical_data.index[-1]
        
        while current_date < end_date:
            split_points.append(current_date)
            # Move forward by step size
            if isinstance(self.historical_data.index, pd.DatetimeIndex):
                if self.step_size == 1:
                    current_date += pd.DateOffset(months=1)
                else:
                    current_date += pd.DateOffset(months=self.step_size)
            else:
                current_date += self.step_size
        
        split_points.append(end_date)
        
        return split_points
    
    def _get_data_slice(self, start, end) -> pd.DataFrame:
        """Get data slice between start and end."""
        if isinstance(start, datetime) and isinstance(end, datetime):
            mask = (self.historical_data.index >= start) & (self.historical_data.index < end)
            return self.historical_data[mask].copy()
        else:
            # Numeric index
            return self.historical_data.iloc[int(start):int(end)].copy()
    
    def _calculate_param_stability(self, params: Dict) -> Dict[str, float]:
        """Calculate parameter stability across periods."""
        if not self.results:
            return {}
        
        # Get all parameter values
        param_values = {name: [] for name in params.keys()}
        
        for result in self.results:
            for name, value in result['params'].items():
                if name in param_values:
                    param_values[name].append(value)
        
        # Calculate stability (coefficient of variation)
        stability = {}
        for name, values in param_values.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val != 0:
                    stability[name] = std_val / abs(mean_val)
                else:
                    stability[name] = 0.0
            else:
                stability[name] = 0.0
        
        return stability
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze walk-forward optimization results."""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Calculate average performance
        test_performances = [r['test_performance'] for r in self.results]
        
        avg_return = np.mean([r.get('total_return', 0) for r in test_performances])
        avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in test_performances])
        avg_max_dd = np.mean([r.get('max_drawdown', 0) for r in test_performances])
        
        # Calculate parameter stability
        all_params = [r['params'] for r in self.results]
        param_stability = {}
        
        for param_name in all_params[0].keys():
            values = [p[param_name] for p in all_params if param_name in p]
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val != 0:
                    param_stability[param_name] = std_val / abs(mean_val)
                else:
                    param_stability[param_name] = 0.0
        
        # Get best parameters (most stable with good performance)
        best_result = max(
            self.results,
            key=lambda x: x['test_performance'].get('sharpe_ratio', 0) * (1 - sum(x['params_stability'].values()) / len(x['params_stability']))
        )
        
        return {
            'avg_performance': {
                'return': avg_return,
                'sharpe_ratio': avg_sharpe,
                'max_drawdown': avg_max_dd
            },
            'param_stability': param_stability,
            'best_params': best_result['params'],
            'best_performance': best_result['test_performance'],
            'results': self.results,
            'consistency_score': self._calculate_consistency_score()
        }
    
    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score across all periods."""
        if not self.results:
            return 0.0
        
        # Get Sharpe ratios
        sharpe_ratios = [
            r['test_performance'].get('sharpe_ratio', 0)
            for r in self.results
        ]
        
        if len(sharpe_ratios) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower is better)
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        if mean_sharpe == 0:
            return 0.0
        
        cv = std_sharpe / abs(mean_sharpe)
        
        # Convert to consistency score (0-1, higher is better)
        consistency = 1.0 / (1.0 + cv)
        
        return consistency

