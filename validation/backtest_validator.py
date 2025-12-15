"""
Backtest Validator
Comprehensive validation through multiple testing methods
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Type
import random

from analytics.backtest_engine import BacktestEngine
from optimization.walk_forward import WalkForwardOptimizer

logger = logging.getLogger(__name__)


class BacktestValidator:
    """
    Comprehensive backtest validator.
    
    Validates strategies through:
    - Standard backtest
    - Walk-forward analysis
    - Monte Carlo simulation
    - Stress testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.historical_data = config.get('historical_data')
        self.initial_cash = config.get('initial_cash', 10000.0)
    
    def validate(
        self,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate strategy through multiple backtests.
        
        Args:
            strategy_class: Strategy class
            params: Strategy parameters
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating strategy {strategy_class.__name__}")
        
        results = {}
        
        # 1. Standard backtest
        logger.info("Running standard backtest...")
        results['standard'] = self._run_standard_backtest(strategy_class, params)
        
        # 2. Walk-forward analysis
        logger.info("Running walk-forward analysis...")
        results['walk_forward'] = self._run_walk_forward(strategy_class, params)
        
        # 3. Monte Carlo simulation
        logger.info("Running Monte Carlo simulation...")
        results['monte_carlo'] = self._run_monte_carlo(strategy_class, params)
        
        # 4. Stress test
        logger.info("Running stress tests...")
        results['stress_test'] = self._run_stress_test(strategy_class, params)
        
        # Analyze all results
        analysis = self._analyze_results(results)
        
        logger.info(f"✅ Validation complete. Overall score: {analysis.get('score', 0):.2f}")
        
        return analysis
    
    def _run_standard_backtest(
        self,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run standard backtest."""
        try:
            strategy = strategy_class(**params)
            engine = BacktestEngine(
                initial_cash=self.initial_cash,
                commission=0.001,
                slippage=0.001
            )
            
            engine.add_data(self.historical_data)
            engine.add_strategy(strategy.__class__, **params)
            
            return engine.run()
        except Exception as e:
            logger.error(f"Standard backtest failed: {e}")
            return {'error': str(e)}
    
    def _run_walk_forward(
        self,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run walk-forward analysis."""
        try:
            optimizer = WalkForwardOptimizer(
                strategy_class,
                {
                    'historical_data': self.historical_data,
                    'initial_cash': self.initial_cash,
                    'parameters': self._params_to_optimizer_format(params),
                    'generations': 20,
                    'population_size': 50
                }
            )
            
            return optimizer.optimize()
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_monte_carlo(
        self,
        strategy_class: Type,
        params: Dict[str, Any],
        simulations: int = 1000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        try:
            # First, run standard backtest to get returns
            standard_result = self._run_standard_backtest(strategy_class, params)
            
            if 'error' in standard_result:
                return standard_result
            
            # Get returns from backtest (simplified - would need actual returns)
            # For now, simulate based on performance
            returns = self._simulate_returns(standard_result)
            
            # Run Monte Carlo simulation
            final_values = []
            for _ in range(simulations):
                # Bootstrap sample returns
                sample_returns = np.random.choice(
                    returns,
                    size=min(len(returns), 252),  # 1 year
                    replace=True
                )
                
                # Calculate final value
                final_value = self.initial_cash * np.prod(1 + sample_returns)
                final_values.append(final_value)
            
            final_values = np.array(final_values)
            
            return {
                'simulations': simulations,
                'avg_final_value': np.mean(final_values),
                'std_final_value': np.std(final_values),
                'min_final_value': np.min(final_values),
                'max_final_value': np.max(final_values),
                'p5_final_value': np.percentile(final_values, 5),
                'p95_final_value': np.percentile(final_values, 95),
                'probability_of_profit': np.mean(final_values > self.initial_cash),
                'expected_return': (np.mean(final_values) - self.initial_cash) / self.initial_cash
            }
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return {'error': str(e)}
    
    def _run_stress_test(
        self,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run stress test with extreme market conditions."""
        scenarios = self._create_stress_scenarios()
        
        results = {}
        for name, scenario_data in scenarios.items():
            try:
                strategy = strategy_class(**params)
                engine = BacktestEngine(
                    initial_cash=self.initial_cash,
                    commission=0.001,
                    slippage=0.001
                )
                
                engine.add_data(scenario_data)
                engine.add_strategy(strategy.__class__, **params)
                
                results[name] = engine.run()
            except Exception as e:
                logger.warning(f"Stress test {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _create_stress_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Create stress test scenarios."""
        scenarios = {}
        
        # 1. Market crash (20% drop in 1 day)
        crash_data = self.historical_data.copy()
        if 'close' in crash_data.columns:
            crash_data['close'] = crash_data['close'] * 0.8
            crash_data['high'] = crash_data['high'] * 0.8
            crash_data['low'] = crash_data['low'] * 0.8
        scenarios['market_crash'] = crash_data
        
        # 2. High volatility (2x standard deviation)
        high_vol_data = self.historical_data.copy()
        if 'close' in high_vol_data.columns:
            returns = high_vol_data['close'].pct_change()
            high_vol_returns = returns * 2
            high_vol_data['close'] = high_vol_data['close'].shift(1) * (1 + high_vol_returns)
            high_vol_data['close'] = high_vol_data['close'].fillna(method='bfill')
        scenarios['high_volatility'] = high_vol_data
        
        # 3. Low liquidity (random gaps in data)
        low_liq_data = self.historical_data.copy()
        mask = np.random.rand(len(low_liq_data)) < 0.1  # 10% missing data
        if 'close' in low_liq_data.columns:
            low_liq_data.loc[mask, 'close'] = np.nan
            low_liq_data['close'] = low_liq_data['close'].fillna(method='ffill').fillna(method='bfill')
        scenarios['low_liquidity'] = low_liq_data
        
        # 4. Flash crash (50% drop then recovery)
        flash_crash_data = self.historical_data.copy()
        if 'close' in flash_crash_data.columns and len(flash_crash_data) > 10:
            mid_point = len(flash_crash_data) // 2
            flash_crash_data.iloc[mid_point:mid_point+5, flash_crash_data.columns.get_loc('close')] *= 0.5
            flash_crash_data.iloc[mid_point+5:mid_point+10, flash_crash_data.columns.get_loc('close')] *= 1.5
        scenarios['flash_crash'] = flash_crash_data
        
        return scenarios
    
    def _simulate_returns(self, backtest_result: Dict) -> np.ndarray:
        """Simulate returns from backtest result."""
        # Extract return information
        total_return = backtest_result.get('total_return', 0)
        sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
        
        # Generate synthetic returns
        if sharpe_ratio > 0:
            # Use Sharpe ratio to estimate volatility
            volatility = abs(total_return) / (sharpe_ratio * np.sqrt(252)) if sharpe_ratio > 0 else 0.02
        else:
            volatility = 0.02
        
        # Generate returns
        mean_return = total_return / 252  # Daily return
        returns = np.random.normal(mean_return, volatility, 252)
        
        return returns
    
    def _params_to_optimizer_format(self, params: Dict) -> List[Dict]:
        """Convert params to optimizer format."""
        optimizer_params = []
        
        for name, value in params.items():
            if isinstance(value, (int, float)):
                if isinstance(value, int):
                    param_type = 'int'
                    min_val = max(1, int(value * 0.5))
                    max_val = int(value * 2)
                else:
                    param_type = 'float'
                    min_val = value * 0.5
                    max_val = value * 2
                
                optimizer_params.append({
                    'name': name,
                    'type': param_type,
                    'min': min_val,
                    'max': max_val
                })
            elif isinstance(value, str):
                optimizer_params.append({
                    'name': name,
                    'type': 'categorical',
                    'options': [value]  # Single option for fixed param
                })
        
        return optimizer_params
    
    def _analyze_results(self, results: Dict) -> Dict[str, Any]:
        """Analyze validation results."""
        analysis = {}
        
        # Standard backtest analysis
        if 'error' not in results.get('standard', {}):
            standard = results['standard']
            analysis['standard'] = {
                'score': self._calculate_score(standard),
                'return': standard.get('total_return', 0),
                'sharpe_ratio': standard.get('sharpe_ratio', 0),
                'max_drawdown': standard.get('max_drawdown', 0)
            }
        else:
            analysis['standard'] = {'score': 0, 'error': results['standard'].get('error')}
        
        # Walk-forward analysis
        if 'error' not in results.get('walk_forward', {}):
            wf = results['walk_forward']
            analysis['walk_forward'] = {
                'score': self._calculate_wf_score(wf),
                'avg_performance': wf.get('avg_performance', {}),
                'consistency': wf.get('consistency_score', 0)
            }
        else:
            analysis['walk_forward'] = {'score': 0, 'error': results['walk_forward'].get('error')}
        
        # Monte Carlo analysis
        if 'error' not in results.get('monte_carlo', {}):
            mc = results['monte_carlo']
            analysis['monte_carlo'] = {
                'score': self._calculate_mc_score(mc),
                'probability_of_profit': mc.get('probability_of_profit', 0),
                'expected_return': mc.get('expected_return', 0)
            }
        else:
            analysis['monte_carlo'] = {'score': 0, 'error': results['monte_carlo'].get('error')}
        
        # Stress test analysis
        stress = results.get('stress_test', {})
        analysis['stress_test'] = {
            'score': self._calculate_stress_score(stress),
            'scenarios': stress
        }
        
        # Overall score
        weights = {
            'standard': 0.3,
            'walk_forward': 0.3,
            'monte_carlo': 0.2,
            'stress_test': 0.2
        }
        
        overall_score = sum(
            analysis.get(test, {}).get('score', 0) * weight
            for test, weight in weights.items()
        )
        
        analysis['score'] = overall_score
        analysis['grade'] = self._get_grade(overall_score)
        
        return analysis
    
    def _calculate_score(self, result: Dict) -> float:
        """Calculate score for standard backtest."""
        return_score = min(result.get('total_return', 0) * 2, 1.0)  # Cap at 1.0
        sharpe_score = min(result.get('sharpe_ratio', 0) / 2.0, 1.0)  # Cap at 1.0
        dd_penalty = max(0, 1.0 - result.get('max_drawdown', 0) * 2)  # Penalize drawdown
        
        return (return_score * 0.4 + sharpe_score * 0.4 + dd_penalty * 0.2)
    
    def _calculate_wf_score(self, result: Dict) -> float:
        """Calculate score for walk-forward analysis."""
        avg_perf = result.get('avg_performance', {})
        consistency = result.get('consistency_score', 0)
        
        sharpe = avg_perf.get('sharpe_ratio', 0)
        sharpe_score = min(sharpe / 2.0, 1.0)
        
        return (sharpe_score * 0.7 + consistency * 0.3)
    
    def _calculate_mc_score(self, result: Dict) -> float:
        """Calculate score for Monte Carlo simulation."""
        prob_profit = result.get('probability_of_profit', 0)
        expected_return = result.get('expected_return', 0)
        
        return (prob_profit * 0.5 + min(expected_return, 1.0) * 0.5)
    
    def _calculate_stress_score(self, scenarios: Dict) -> float:
        """Calculate score for stress tests."""
        if not scenarios:
            return 0.0
        
        scores = []
        for name, result in scenarios.items():
            if 'error' not in result:
                # Penalize negative returns in stress scenarios
                return_val = result.get('total_return', 0)
                if return_val < -0.5:  # More than 50% loss
                    scores.append(0.0)
                elif return_val < 0:
                    scores.append(0.5)
                else:
                    scores.append(1.0)
            else:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def _get_grade(self, score: float) -> str:
        """Get letter grade from score."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.5:
            return 'D'
        else:
            return 'F'

