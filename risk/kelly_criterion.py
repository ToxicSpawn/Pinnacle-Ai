"""
Kelly Criterion for Optimal Position Sizing
Maximizes long-term growth while managing risk
"""
from __future__ import annotations

import logging
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeStats:
    """Statistics for Kelly Criterion calculation."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def avg_win(self) -> float:
        """Calculate average win amount."""
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit / self.winning_trades
    
    @property
    def avg_loss(self) -> float:
        """Calculate average loss amount."""
        if self.losing_trades == 0:
            return 0.0
        return abs(self.total_loss) / self.losing_trades
    
    @property
    def win_loss_ratio(self) -> float:
        """Calculate win/loss ratio."""
        if self.avg_loss == 0:
            return 0.0
        return self.avg_win / self.avg_loss


class KellyCriterion:
    """
    Kelly Criterion calculator for optimal position sizing.
    
    Formula: f = (p * b - q) / b
    where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = odds (win/loss ratio)
    """
    
    @staticmethod
    def calculate(
        win_rate: float,
        win_loss_ratio: float,
        fractional: bool = True
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
            fractional: If True, use fractional Kelly (recommended, divide by 2-4)
            
        Returns:
            Optimal fraction of capital to risk (0-1)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        if win_loss_ratio <= 0:
            return 0.0
        
        # Kelly formula
        q = 1 - win_rate
        kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Ensure non-negative
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Fractional Kelly (safer, recommended)
        if fractional:
            kelly_fraction = kelly_fraction / 2  # Half Kelly
        
        # Cap at reasonable maximum (e.g., 25% of capital)
        kelly_fraction = min(kelly_fraction, 0.25)
        
        return kelly_fraction
    
    @staticmethod
    def calculate_from_stats(stats: TradeStats, fractional: bool = True) -> float:
        """
        Calculate Kelly fraction from trade statistics.
        
        Args:
            stats: TradeStats object with historical data
            fractional: Use fractional Kelly
            
        Returns:
            Optimal fraction of capital to risk
        """
        if stats.total_trades == 0:
            return 0.0
        
        win_rate = stats.win_rate
        win_loss_ratio = stats.win_loss_ratio
        
        return KellyCriterion.calculate(win_rate, win_loss_ratio, fractional)
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        win_rate: float,
        win_loss_ratio: float,
        fractional: bool = True
    ) -> float:
        """
        Calculate optimal position size in base currency.
        
        Args:
            account_balance: Total account balance
            win_rate: Probability of winning
            win_loss_ratio: Win/loss ratio
            fractional: Use fractional Kelly
            
        Returns:
            Position size in base currency
        """
        kelly_fraction = KellyCriterion.calculate(win_rate, win_loss_ratio, fractional)
        return account_balance * kelly_fraction
    
    @staticmethod
    def is_valid_kelly(kelly_fraction: float) -> bool:
        """
        Check if Kelly fraction is valid for trading.
        
        Args:
            kelly_fraction: Calculated Kelly fraction
            
        Returns:
            True if valid, False otherwise
        """
        # Kelly should be positive and reasonable
        return 0.0 < kelly_fraction <= 0.25


class KellyPositionSizer:
    """
    Position sizer using Kelly Criterion with dynamic updates.
    """
    
    def __init__(self, fractional: bool = True, max_position_pct: float = 0.25):
        """
        Initialize position sizer.
        
        Args:
            fractional: Use fractional Kelly (safer)
            max_position_pct: Maximum position size as % of capital
        """
        self.fractional = fractional
        self.max_position_pct = max_position_pct
        self.stats = TradeStats()
    
    def update_trade(self, profit: float) -> None:
        """
        Update statistics with trade outcome.
        
        Args:
            profit: Trade profit (positive) or loss (negative)
        """
        self.stats.total_trades += 1
        
        if profit > 0:
            self.stats.winning_trades += 1
            self.stats.total_profit += profit
        else:
            self.stats.losing_trades += 1
            self.stats.total_loss += profit
    
    def get_position_size(self, account_balance: float) -> float:
        """
        Get optimal position size based on current statistics.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Position size in base currency
        """
        kelly_fraction = KellyCriterion.calculate_from_stats(self.stats, self.fractional)
        
        # Apply maximum cap
        kelly_fraction = min(kelly_fraction, self.max_position_pct)
        
        return account_balance * kelly_fraction
    
    def reset(self) -> None:
        """Reset statistics."""
        self.stats = TradeStats()
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'total_trades': self.stats.total_trades,
            'win_rate': self.stats.win_rate,
            'win_loss_ratio': self.stats.win_loss_ratio,
            'avg_win': self.stats.avg_win,
            'avg_loss': self.stats.avg_loss,
            'kelly_fraction': KellyCriterion.calculate_from_stats(self.stats, self.fractional),
        }

