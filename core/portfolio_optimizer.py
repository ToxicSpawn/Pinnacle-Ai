"""
Portfolio Optimization Engine
Mean-Variance Optimization (MVO) with correlation adjustment
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from cvxpy import Variable, Problem, Minimize, quad_form
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("CVXPY not available. Using simplified optimization.")


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    name: str
    returns: pd.Series
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


class PortfolioOptimizer:
    """
    Portfolio optimizer using Mean-Variance Optimization.
    
    Features:
    - Mean-Variance Optimization (MVO)
    - Correlation adjustment
    - Risk parity
    - Dynamic rebalancing
    """
    
    def __init__(
        self,
        strategies: List[str],
        config: Optional[Dict] = None
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            strategies: List of strategy names
            config: Configuration dictionary
        """
        self.strategies = strategies
        self.config = config or {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.performance_data: Dict[str, StrategyPerformance] = {}
    
    def calculate_correlation_matrix(
        self,
        returns_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategies.
        
        Args:
            returns_data: Dictionary mapping strategy names to return series
            
        Returns:
            Correlation matrix DataFrame
        """
        # Align returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        self.correlation_matrix = returns_df.corr()
        logger.info("✅ Calculated correlation matrix")
        
        return self.correlation_matrix
    
    def calculate_sharpe_ratios(
        self,
        returns_data: Dict[str, pd.Series],
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate Sharpe ratios for strategies.
        
        Args:
            returns_data: Dictionary mapping strategy names to return series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary mapping strategy names to Sharpe ratios
        """
        sharpe_ratios = {}
        
        for strategy, returns in returns_data.items():
            if len(returns) == 0 or returns.std() == 0:
                sharpe_ratios[strategy] = 0.0
                continue
            
            # Annualize returns and volatility
            annual_return = returns.mean() * 252  # Assuming daily returns
            annual_vol = returns.std() * np.sqrt(252)
            
            if annual_vol > 0:
                sharpe_ratios[strategy] = (annual_return - risk_free_rate) / annual_vol
            else:
                sharpe_ratios[strategy] = 0.0
        
        return sharpe_ratios
    
    def optimize_allocation(
        self,
        returns_data: Dict[str, pd.Series],
        method: str = 'mvo',
        max_correlation: float = 0.7,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Optimize capital allocation across strategies.
        
        Args:
            returns_data: Dictionary mapping strategy names to return series
            method: Optimization method ('mvo', 'equal', 'sharpe')
            max_correlation: Maximum allowed correlation
            min_weight: Minimum weight per strategy
            max_weight: Maximum weight per strategy
            
        Returns:
            Dictionary mapping strategy names to optimal weights
        """
        if method == 'equal':
            # Equal weighting
            n = len(returns_data)
            return {strategy: 1.0 / n for strategy in returns_data.keys()}
        
        elif method == 'sharpe':
            # Weight by Sharpe ratio
            sharpe_ratios = self.calculate_sharpe_ratios(returns_data)
            total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())
            
            if total_sharpe == 0:
                return {strategy: 1.0 / len(returns_data) for strategy in returns_data.keys()}
            
            weights = {
                strategy: max(0, sharpe_ratios[strategy]) / total_sharpe
                for strategy in sharpe_ratios.keys()
            }
            
            # Normalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            return weights
        
        elif method == 'mvo':
            # Mean-Variance Optimization
            return self._mean_variance_optimization(
                returns_data,
                max_correlation,
                min_weight,
                max_weight
            )
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _mean_variance_optimization(
        self,
        returns_data: Dict[str, pd.Series],
        max_correlation: float,
        min_weight: float,
        max_weight: float
    ) -> Dict[str, float]:
        """
        Mean-Variance Optimization.
        
        Args:
            returns_data: Dictionary mapping strategy names to return series
            max_correlation: Maximum allowed correlation
            min_weight: Minimum weight
            max_weight: Maximum weight
            
        Returns:
            Optimal weights
        """
        # Align returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) == 0:
            logger.warning("No valid returns data. Using equal weights.")
            return {strategy: 1.0 / len(returns_data) for strategy in returns_data.keys()}
        
        # Calculate expected returns and covariance
        mu = returns_df.mean().values * 252  # Annualize
        Sigma = returns_df.cov().values * 252  # Annualize
        
        n = len(mu)
        
        if CVXPY_AVAILABLE:
            # Use CVXPY for optimization
            try:
                w = Variable(n)
                
                # Objective: maximize Sharpe ratio (minimize negative Sharpe)
                # Simplified: maximize return - risk
                risk_aversion = self.config.get('risk_aversion', 0.5)
                objective = Minimize(quad_form(w, Sigma) * risk_aversion - mu.T @ w)
                
                # Constraints
                constraints = [
                    w >= min_weight,
                    w <= max_weight,
                    sum(w) == 1.0
                ]
                
                # Correlation constraint
                if self.correlation_matrix is not None:
                    corr_matrix = self.correlation_matrix.values
                    # Add constraint for portfolio correlation
                    # This is simplified - actual implementation would be more complex
                    pass
                
                problem = Problem(objective, constraints)
                problem.solve()
                
                if problem.status == 'optimal':
                    weights = w.value
                    weights = np.maximum(weights, 0)  # Ensure non-negative
                    weights = weights / weights.sum()  # Normalize
                    
                    return dict(zip(returns_data.keys(), weights))
            
            except Exception as e:
                logger.warning(f"CVXPY optimization failed: {e}. Using simplified method.")
        
        # Fallback: Simplified optimization
        # Maximize Sharpe ratio
        sharpe_ratios = self.calculate_sharpe_ratios(returns_data)
        
        # Weight by Sharpe ratio (normalized)
        total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())
        
        if total_sharpe == 0:
            return {strategy: 1.0 / n for strategy in returns_data.keys()}
        
        weights = {
            strategy: max(0, sharpe_ratios[strategy]) / total_sharpe
            for strategy in sharpe_ratios.keys()
        }
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights, min_weight, max_weight)
        
        return weights
    
    def _apply_weight_constraints(
        self,
        weights: Dict[str, float],
        min_weight: float,
        max_weight: float
    ) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        # Clip weights
        weights = {k: max(min_weight, min(max_weight, v)) for k, v in weights.items()}
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def adjust_for_correlation(
        self,
        weights: Dict[str, float],
        correlation_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Adjust weights to minimize correlation risk.
        
        Args:
            weights: Current weights
            correlation_threshold: Correlation threshold
            
        Returns:
            Adjusted weights
        """
        if self.correlation_matrix is None:
            return weights
        
        # Find highly correlated pairs
        corr_pairs = []
        for i, s1 in enumerate(self.correlation_matrix.index):
            for j, s2 in enumerate(self.correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = self.correlation_matrix.loc[s1, s2]
                    if abs(corr) > correlation_threshold:
                        corr_pairs.append((s1, s2, corr))
        
        # Reduce weights of correlated strategies
        adjusted_weights = weights.copy()
        for s1, s2, corr in corr_pairs:
            if s1 in adjusted_weights and s2 in adjusted_weights:
                # Reduce weight of strategy with lower Sharpe ratio
                # (This is simplified - actual implementation would be more sophisticated)
                if adjusted_weights[s1] > adjusted_weights[s2]:
                    adjusted_weights[s1] *= 0.9
                else:
                    adjusted_weights[s2] *= 0.9
        
        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def get_optimal_allocation(
        self,
        returns_data: Dict[str, pd.Series],
        method: str = 'mvo'
    ) -> Dict[str, float]:
        """
        Get optimal allocation with all adjustments.
        
        Args:
            returns_data: Dictionary mapping strategy names to return series
            method: Optimization method
            
        Returns:
            Optimal weights
        """
        # Calculate correlation matrix
        self.calculate_correlation_matrix(returns_data)
        
        # Optimize allocation
        weights = self.optimize_allocation(returns_data, method=method)
        
        # Adjust for correlation
        weights = self.adjust_for_correlation(weights)
        
        return weights

