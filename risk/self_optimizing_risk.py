"""
Self-Optimizing Risk Management
Dynamic risk management that adjusts parameters based on market conditions
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from optimization.genetic_optimizer import GeneticHyperparameterOptimizer
    GENETIC_OPTIMIZER_AVAILABLE = True
except ImportError:
    GENETIC_OPTIMIZER_AVAILABLE = False
    logger.warning("Genetic optimizer not available")


class SelfOptimizingRiskManager:
    """
    Self-optimizing risk management system.
    
    Features:
    - Dynamic position sizing
    - Adaptive risk limits
    - Correlation-based risk adjustment
    - Liquidity checks
    - Automatic parameter optimization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize self-optimizing risk manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.2)
        self.max_daily_loss = config.get('max_daily_loss', 0.1)
        self.max_drawdown = config.get('max_drawdown', 0.2)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.volatility_multiplier = config.get('volatility_multiplier', 1.0)
        
        self.performance_history: List[float] = []
        self.daily_pnl = 0.0
        self.drawdown = 0.0
        self.current_positions: Dict[str, float] = {}
        
        self.risk_metrics = {
            'volatility': [],
            'drawdown': [],
            'correlation': [],
            'liquidity': []
        }
        
        if GENETIC_OPTIMIZER_AVAILABLE:
            self.optimizer = GeneticHyperparameterOptimizer(
                self.__class__,
                {
                    'historical_data': None,  # Will be set later
                    'initial_cash': config.get('initial_capital', 10000),
                    'parameters': self._get_default_parameters()
                }
            )
        else:
            self.optimizer = None
        
        self.last_optimization = 0
    
    def _get_default_parameters(self) -> List[Dict]:
        """Get default risk parameters."""
        return [
            {'name': 'max_position_size', 'type': 'float', 'min': 0.01, 'max': 0.5},
            {'name': 'max_daily_loss', 'type': 'float', 'min': 0.01, 'max': 0.2},
            {'name': 'max_drawdown', 'type': 'float', 'min': 0.05, 'max': 0.5},
            {'name': 'correlation_threshold', 'type': 'float', 'min': 0.5, 'max': 0.9},
            {'name': 'volatility_multiplier', 'type': 'float', 'min': 0.5, 'max': 2.0}
        ]
    
    def check_order(
        self,
        symbol: str,
        amount: float,
        price: float,
        market_data: Optional[Dict] = None
    ) -> bool:
        """
        Check if order meets all risk criteria.
        
        Args:
            symbol: Trading symbol
            amount: Order amount
            price: Order price
            market_data: Current market data
            
        Returns:
            True if order is allowed
        """
        if market_data is None:
            market_data = {}
        
        # Update risk metrics
        self._update_risk_metrics(market_data)
        
        # Check position size
        max_position = self.max_position_size * self._get_position_size_multiplier()
        position_value = abs(amount * price)
        total_capital = self._get_total_capital()
        
        if position_value > max_position * total_capital:
            logger.warning(
                f"Position size {position_value:.2f} exceeds maximum "
                f"{max_position * total_capital:.2f}"
            )
            return False
        
        # Check daily loss limit
        daily_loss_multiplier = self._get_daily_loss_multiplier()
        if self.daily_pnl < -self.max_daily_loss * daily_loss_multiplier * total_capital:
            logger.warning("Daily loss limit exceeded")
            return False
        
        # Check drawdown
        drawdown_multiplier = self._get_drawdown_multiplier()
        if self.drawdown > self.max_drawdown * drawdown_multiplier:
            logger.warning("Drawdown limit exceeded")
            return False
        
        # Check correlation
        if self._is_correlated(symbol, self.current_positions):
            logger.warning(f"Position in {symbol} would exceed correlation limits")
            return False
        
        # Check liquidity
        if not self._check_liquidity(symbol, amount, market_data):
            logger.warning(f"Insufficient liquidity for {symbol} order")
            return False
        
        return True
    
    def _get_position_size_multiplier(self) -> float:
        """Get dynamic position size multiplier based on market conditions."""
        volatility = self._get_market_volatility()
        return self.volatility_multiplier / (1 + volatility)
    
    def _get_daily_loss_multiplier(self) -> float:
        """Get dynamic daily loss multiplier."""
        # Reduce risk after consecutive losses
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            if all(p < 0 for p in recent_performance):
                return 0.5  # Reduce risk by 50%
        return 1.0
    
    def _get_drawdown_multiplier(self) -> float:
        """Get dynamic drawdown multiplier."""
        # Reduce risk after large drawdowns
        if self.drawdown > self.max_drawdown * 0.8:
            return 0.7  # Reduce risk by 30%
        return 1.0
    
    def _update_risk_metrics(self, market_data: Dict):
        """Update risk metrics based on current market conditions."""
        # Update volatility
        volatility = market_data.get('volatility', 0.02)
        self.risk_metrics['volatility'].append(volatility)
        if len(self.risk_metrics['volatility']) > 100:
            self.risk_metrics['volatility'].pop(0)
        
        # Update drawdown
        self.risk_metrics['drawdown'].append(self.drawdown)
        if len(self.risk_metrics['drawdown']) > 100:
            self.risk_metrics['drawdown'].pop(0)
        
        # Update correlation
        portfolio_corr = self._calculate_portfolio_correlation()
        self.risk_metrics['correlation'].append(portfolio_corr)
        if len(self.risk_metrics['correlation']) > 100:
            self.risk_metrics['correlation'].pop(0)
        
        # Update liquidity
        liquidity = market_data.get('liquidity', 1000000.0)
        self.risk_metrics['liquidity'].append(liquidity)
        if len(self.risk_metrics['liquidity']) > 100:
            self.risk_metrics['liquidity'].pop(0)
    
    def _get_market_volatility(self) -> float:
        """Get current market volatility."""
        if not self.risk_metrics['volatility']:
            return 0.02  # Default 2% volatility
        return float(np.mean(self.risk_metrics['volatility']))
    
    def _calculate_portfolio_correlation(self) -> float:
        """Calculate current portfolio correlation."""
        if len(self.current_positions) < 2:
            return 0.0
        
        # Get correlation matrix (placeholder)
        correlation_matrix = self._get_correlation_matrix()
        
        # Calculate portfolio correlation
        symbols = list(self.current_positions.keys())
        weights = np.array([self.current_positions[s] for s in symbols])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        # Calculate weighted correlation
        portfolio_corr = 0.0
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    corr = correlation_matrix.get(symbol1, {}).get(symbol2, 0.0)
                    portfolio_corr += weights[i] * weights[j] * corr
        
        return float(portfolio_corr)
    
    def _get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get current correlation matrix."""
        # Placeholder: would query database or calculate from historical data
        symbols = list(self.current_positions.keys())
        matrix = {}
        for symbol1 in symbols:
            matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = 1.0
                else:
                    # Placeholder correlation
                    matrix[symbol1][symbol2] = 0.3
        return matrix
    
    def _check_liquidity(
        self,
        symbol: str,
        amount: float,
        market_data: Dict
    ) -> bool:
        """Check if there's sufficient liquidity for the order."""
        order_book = market_data.get('order_book', {})
        
        if not order_book:
            return True  # Assume sufficient liquidity if no data
        
        # Calculate available liquidity
        if amount > 0:  # Buy order
            available = sum(
                volume for price, volume in order_book.get('asks', [])[:5]
            )
        else:  # Sell order
            available = sum(
                volume for price, volume in order_book.get('bids', [])[:5]
            )
        
        # Only use 50% of available liquidity
        return abs(amount) <= available * 0.5
    
    def _is_correlated(self, symbol: str, positions: Dict[str, float]) -> bool:
        """Check if symbol is correlated with existing positions."""
        if not positions:
            return False
        
        correlation_matrix = self._get_correlation_matrix()
        
        for pos_symbol in positions.keys():
            corr = correlation_matrix.get(symbol, {}).get(pos_symbol, 0.0)
            if corr > self.correlation_threshold:
                return True
        
        return False
    
    def _get_total_capital(self) -> float:
        """Get total capital."""
        # Placeholder: would query actual capital
        return self.config.get('initial_capital', 10000.0)
    
    def update_performance(self, pnl: float):
        """Update performance metrics."""
        self.performance_history.append(pnl)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Update daily PnL
        self.daily_pnl += pnl
        
        # Update drawdown
        if len(self.performance_history) > 0:
            peak = max(self.performance_history)
            current = sum(self.performance_history)
            if peak > 0:
                self.drawdown = (peak - current) / peak
            else:
                self.drawdown = 0.0
    
    def optimize_parameters(self):
        """Optimize risk parameters using genetic algorithm."""
        if not self.optimizer:
            logger.warning("Genetic optimizer not available")
            return
        
        # Get historical data
        historical_data = self._get_historical_data()
        
        if historical_data is None or historical_data.empty:
            logger.warning("No historical data available for optimization")
            return
        
        # Set up optimizer
        self.optimizer.config['historical_data'] = historical_data
        
        # Run optimization
        try:
            best_params = self.optimizer.optimize(
                generations=self.config.get('optimization', {}).get('generations', 30),
                population_size=self.config.get('optimization', {}).get('population_size', 100)
            )
            
            # Update configuration
            self.max_position_size = best_params.get('max_position_size', self.max_position_size)
            self.max_daily_loss = best_params.get('max_daily_loss', self.max_daily_loss)
            self.max_drawdown = best_params.get('max_drawdown', self.max_drawdown)
            self.correlation_threshold = best_params.get('correlation_threshold', self.correlation_threshold)
            self.volatility_multiplier = best_params.get('volatility_multiplier', self.volatility_multiplier)
            
            logger.info(f"Optimized risk parameters: {best_params}")
            self.last_optimization = time.time()
        except Exception as e:
            logger.error(f"Risk parameter optimization failed: {e}")
    
    def _get_historical_data(self) -> Optional[pd.DataFrame]:
        """Get historical market data."""
        # Placeholder: would query database
        return pd.DataFrame()

