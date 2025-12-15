"""
Maximum Drawdown Control
Monitors and enforces drawdown limits to protect capital
"""
from __future__ import annotations

import logging
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DrawdownSnapshot:
    """Snapshot of drawdown state."""
    timestamp: datetime
    balance: float
    peak_balance: float
    drawdown: float
    drawdown_pct: float


class DrawdownControl:
    """
    Maximum drawdown controller.
    
    Monitors account balance and enforces maximum drawdown limits.
    Automatically halts trading when drawdown exceeds threshold.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.20,
        initial_balance: Optional[float] = None,
        recovery_threshold: float = 0.10
    ):
        """
        Initialize drawdown controller.
        
        Args:
            max_drawdown: Maximum allowed drawdown (0.20 = 20%)
            initial_balance: Starting balance (None = auto-detect from first update)
            recovery_threshold: Drawdown must recover to this level before resuming (0.10 = 10%)
        """
        self.max_drawdown = max_drawdown
        self.recovery_threshold = recovery_threshold
        self.initial_balance = initial_balance
        self.peak_balance: Optional[float] = None
        self.current_balance: Optional[float] = None
        self.max_drawdown_seen: float = 0.0
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.snapshots: list[DrawdownSnapshot] = []
    
    def update(self, current_balance: float, timestamp: Optional[datetime] = None) -> bool:
        """
        Update drawdown tracking with current balance.
        
        Args:
            current_balance: Current account balance
            timestamp: Optional timestamp for snapshot
            
        Returns:
            True if trading allowed, False if halted
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_balance = current_balance
        
        # Initialize on first update
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.peak_balance = current_balance
            logger.info(f"DrawdownControl: Initialized with balance {current_balance:.2f}")
            return True
        
        # Update peak balance
        if self.peak_balance is None or current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown from peak
        if self.peak_balance > 0:
            drawdown = self.peak_balance - current_balance
            drawdown_pct = drawdown / self.peak_balance
            
            # Track maximum drawdown seen
            if drawdown_pct > self.max_drawdown_seen:
                self.max_drawdown_seen = drawdown_pct
            
            # Create snapshot
            snapshot = DrawdownSnapshot(
                timestamp=timestamp,
                balance=current_balance,
                peak_balance=self.peak_balance,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct
            )
            self.snapshots.append(snapshot)
            
            # Check if drawdown exceeds limit
            if drawdown_pct > self.max_drawdown:
                if not self.trading_halted:
                    self.trading_halted = True
                    self.halt_reason = f"Drawdown {drawdown_pct:.2%} exceeds maximum {self.max_drawdown:.2%}"
                    logger.warning(f"⚠️ {self.halt_reason}")
                return False
            
            # Check if recovered enough to resume
            if self.trading_halted:
                if drawdown_pct <= self.recovery_threshold:
                    self.trading_halted = False
                    self.halt_reason = None
                    logger.info(f"✅ Drawdown recovered to {drawdown_pct:.2%}. Trading resumed.")
                    return True
                else:
                    return False
        
        return True
    
    def get_status(self) -> Dict:
        """
        Get current drawdown status.
        
        Returns:
            Dictionary with status information
        """
        if self.current_balance is None or self.peak_balance is None:
            return {
                'initialized': False,
                'trading_halted': False,
                'drawdown_pct': 0.0,
                'max_drawdown_seen': 0.0,
            }
        
        drawdown = self.peak_balance - self.current_balance
        drawdown_pct = drawdown / self.peak_balance if self.peak_balance > 0 else 0.0
        
        return {
            'initialized': True,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct,
            'max_drawdown_limit': self.max_drawdown,
            'max_drawdown_seen': self.max_drawdown_seen,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'recovery_threshold': self.recovery_threshold,
        }
    
    def reset(self, new_initial_balance: Optional[float] = None) -> None:
        """
        Reset drawdown tracking.
        
        Args:
            new_initial_balance: New initial balance (None = use current balance)
        """
        if new_initial_balance is not None:
            self.initial_balance = new_initial_balance
        elif self.current_balance is not None:
            self.initial_balance = self.current_balance
        
        self.peak_balance = self.initial_balance
        self.max_drawdown_seen = 0.0
        self.trading_halted = False
        self.halt_reason = None
        self.snapshots.clear()
        logger.info("DrawdownControl: Reset")
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed.
        
        Returns:
            True if trading allowed, False if halted
        """
        return not self.trading_halted
    
    def get_recent_snapshots(self, count: int = 10) -> list[DrawdownSnapshot]:
        """
        Get recent drawdown snapshots.
        
        Args:
            count: Number of recent snapshots to return
            
        Returns:
            List of recent snapshots
        """
        return self.snapshots[-count:] if self.snapshots else []

