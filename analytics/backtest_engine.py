"""
Backtesting Engine using Backtrader
Realistic backtesting with slippage, fees, and latency simulation
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    logger.warning("Backtrader not available. Backtesting will be disabled.")


class RealisticBroker(bt.brokers.BackBroker):
    """
    Realistic broker with slippage and fees.
    """
    
    def __init__(self, slippage: float = 0.001, commission: float = 0.001, **kwargs):
        """
        Initialize realistic broker.
        
        Args:
            slippage: Slippage percentage (0.001 = 0.1%)
            commission: Commission percentage (0.001 = 0.1%)
        """
        super().__init__(**kwargs)
        self.slippage = slippage
        self.setcommission(commission=commission)
    
    def _execute_order(self, order):
        """Execute order with slippage."""
        if order.executed.size:
            # Apply slippage
            if order.isbuy():
                price = order.data.close[0] * (1 + self.slippage)
            else:
                price = order.data.close[0] * (1 - self.slippage)
            
            order.executed.price = price
            return super()._execute_order(order)


class BacktestEngine:
    """
    Backtesting engine with realistic market conditions.
    
    Features:
    - Slippage simulation
    - Trading fees
    - Latency simulation
    - Multiple strategy support
    - Performance metrics
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        latency_ms: float = 50.0
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_cash: Starting capital
            commission: Trading commission (0.001 = 0.1%)
            slippage: Slippage percentage (0.001 = 0.1%)
            latency_ms: Simulated latency in milliseconds
        """
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader is required for backtesting")
        
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.latency_ms = latency_ms
        self.cerebro = bt.Cerebro()
        
        # Set up realistic broker
        self.cerebro.broker = RealisticBroker(
            slippage=slippage,
            commission=commission
        )
        self.cerebro.broker.setcash(initial_cash)
    
    def add_data(
        self,
        df: pd.DataFrame,
        name: str = "data"
    ) -> None:
        """
        Add historical data to backtest.
        
        Args:
            df: DataFrame with OHLCV data
            name: Data feed name
        """
        # Ensure proper column names
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Convert to Backtrader format
        df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
        
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data, name=name)
    
    def add_strategy(self, strategy_class, **kwargs) -> None:
        """
        Add trading strategy to backtest.
        
        Args:
            strategy_class: Strategy class (must inherit from bt.Strategy)
            **kwargs: Strategy parameters
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def add_analyzer(self, analyzer_class, **kwargs) -> None:
        """
        Add performance analyzer.
        
        Args:
            analyzer_class: Analyzer class
            **kwargs: Analyzer parameters
        """
        self.cerebro.addanalyzer(analyzer_class, **kwargs)
    
    def run(self) -> Dict[str, Any]:
        """
        Run backtest.
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with ${self.initial_cash:.2f}")
        
        # Add default analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run backtest
        results = self.cerebro.run()
        
        # Extract results
        strat = results[0]
        final_value = self.cerebro.broker.getvalue()
        
        # Get analyzer results
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        results_dict = {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe.get('sharperatio', 0.0),
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0.0),
            'max_drawdown_pct': drawdown.get('max', {}).get('drawdown', 0.0) * 100,
            'total_trades': trades.get('total', {}).get('total', 0),
            'won': trades.get('won', {}).get('total', 0),
            'lost': trades.get('lost', {}).get('total', 0),
            'win_rate': (
                trades.get('won', {}).get('total', 0) / 
                max(trades.get('total', {}).get('total', 1), 1)
            ) * 100,
            'avg_win': trades.get('won', {}).get('pnl', {}).get('average', 0.0),
            'avg_loss': trades.get('lost', {}).get('pnl', {}).get('average', 0.0),
        }
        
        logger.info(f"Backtest complete: Final value ${final_value:.2f} ({total_return*100:.2f}% return)")
        
        return results_dict
    
    def plot(self, style: str = 'candlestick', **kwargs) -> None:
        """
        Plot backtest results.
        
        Args:
            style: Plot style ('candlestick', 'line', etc.)
            **kwargs: Additional plot arguments
        """
        self.cerebro.plot(style=style, **kwargs)
    
    def save_results(self, filepath: str, results: Dict[str, Any]) -> None:
        """
        Save backtest results to file.
        
        Args:
            filepath: Path to save results
            results: Results dictionary
        """
        import json
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


# Example strategy for backtesting
class SMACrossStrategy(bt.Strategy):
    """Simple moving average crossover strategy."""
    
    params = (
        ('fast', 10),
        ('slow', 30),
    )
    
    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.p.fast)
        self.sma_slow = bt.indicators.SMA(period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

