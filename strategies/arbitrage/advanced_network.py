"""
Advanced Arbitrage Network
Builds arbitrage graph across exchanges and symbols
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from core.low_latency import LowLatencyEngine
    LOW_LATENCY_AVAILABLE = True
except ImportError:
    LOW_LATENCY_AVAILABLE = False

try:
    from execution.hft_engine import HFTExecutionEngine
    HFT_ENGINE_AVAILABLE = True
except ImportError:
    HFT_ENGINE_AVAILABLE = False


class AdvancedArbitrageNetwork:
    """
    Advanced arbitrage network that builds a graph of opportunities.
    
    Features:
    - Multi-exchange arbitrage graph
    - Latency-based opportunity detection
    - Dynamic position sizing
    - Execution probability calculation
    """
    
    def __init__(self, exchange_cluster: Any, config: Dict):
        """
        Initialize advanced arbitrage network.
        
        Args:
            exchange_cluster: Exchange cluster instance
            config: Configuration dictionary
        """
        self.exchanges = exchange_cluster
        self.config = config
        
        if LOW_LATENCY_AVAILABLE:
            self.latency_engine = LowLatencyEngine(config)
        else:
            self.latency_engine = None
        
        if HFT_ENGINE_AVAILABLE:
            primary_exchange = exchange_cluster.get_exchange(
                config.get('primary_exchange', 'kraken')
            )
            self.execution_engine = HFTExecutionEngine(primary_exchange, config)
        else:
            self.execution_engine = None
        
        self.arbitrage_graph: Dict[str, Dict] = {}
        self.opportunity_cache: Dict[str, Dict] = {}
        
        # Build arbitrage graph
        self._build_arbitrage_graph()
    
    def _build_arbitrage_graph(self):
        """Build arbitrage graph between all exchanges and symbols."""
        # Get all available symbols from all exchanges
        symbols = set()
        for exchange_name, exchange in self.exchanges.exchanges.items():
            if hasattr(exchange, 'symbols'):
                symbols.update(exchange.symbols)
        
        # Build graph
        for symbol in symbols:
            self.arbitrage_graph[symbol] = {}
            exchange_names = list(self.exchanges.exchanges.keys())
            
            for i, name1 in enumerate(exchange_names):
                for j, name2 in enumerate(exchange_names):
                    if i != j:
                        edge_key = (name1, name2)
                        latency = 0.0
                        if self.latency_engine:
                            try:
                                latency = self.latency_engine.get_latency(name1, name2)
                            except:
                                latency = 100.0  # Default latency
                        
                        self.arbitrage_graph[symbol][edge_key] = {
                            'latency': latency,
                            'last_price': None,
                            'last_update': 0
                        }
        
        logger.info(f"Built arbitrage graph with {len(symbols)} symbols")
    
    async def run(self):
        """Run the arbitrage detection loop."""
        logger.info("Starting advanced arbitrage network...")
        
        while True:
            try:
                # Update arbitrage graph
                await self._update_arbitrage_graph()
                
                # Find arbitrage opportunities
                opportunities = self._find_opportunities()
                
                # Execute profitable opportunities
                await self._execute_opportunities(opportunities)
                
                # Sleep for refresh interval
                refresh_interval = self.config.get('refresh_interval', 0.5)
                await asyncio.sleep(refresh_interval)
                
            except Exception as e:
                logger.error(f"Arbitrage error: {e}")
                await asyncio.sleep(5)
    
    async def _update_arbitrage_graph(self):
        """Update prices in the arbitrage graph."""
        tasks = []
        for symbol in self.arbitrage_graph:
            for (name1, name2) in self.arbitrage_graph[symbol]:
                tasks.append(self._update_arbitrage_edge(symbol, name1, name2))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _update_arbitrage_edge(self, symbol: str, name1: str, name2: str):
        """Update a single edge in the arbitrage graph."""
        try:
            exchange1 = self.exchanges.get_exchange(name1)
            exchange2 = self.exchanges.get_exchange(name2)
            
            # Get prices from both exchanges
            price1 = await exchange1.get_price(symbol)
            price2 = await exchange2.get_price(symbol)
            
            # Update graph
            self.arbitrage_graph[symbol][(name1, name2)]['last_price'] = (price1, price2)
            self.arbitrage_graph[symbol][(name1, name2)]['last_update'] = time.time()
        except Exception as e:
            logger.warning(f"Failed to update {symbol} {name1}-{name2}: {e}")
    
    def _find_opportunities(self) -> List[Dict]:
        """Find arbitrage opportunities in the graph."""
        opportunities = []
        min_profit = self.config.get('min_profit', 0.005)
        latency_threshold = self.config.get('latency_threshold', 100.0)
        
        for symbol in self.arbitrage_graph:
            for (name1, name2), edge in self.arbitrage_graph[symbol].items():
                if edge['last_price'] is None:
                    continue
                
                price1, price2 = edge['last_price']
                latency = edge['latency']
                
                if price1 == 0 or price2 == 0:
                    continue
                
                # Calculate potential profit
                profit_pct = (price2 - price1) / price1 - min_profit
                
                if profit_pct > 0:
                    # Calculate execution probability based on latency
                    execution_prob = np.exp(-latency / latency_threshold)
                    
                    # Calculate expected profit
                    expected_profit = profit_pct * execution_prob
                    
                    if expected_profit > 0:
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': name1,
                            'sell_exchange': name2,
                            'buy_price': price1,
                            'sell_price': price2,
                            'profit_pct': profit_pct,
                            'latency': latency,
                            'execution_prob': execution_prob,
                            'expected_profit': expected_profit,
                            'timestamp': time.time()
                        })
        
        # Sort by expected profit
        opportunities.sort(key=lambda x: x['expected_profit'], reverse=True)
        
        return opportunities
    
    async def _execute_opportunities(self, opportunities: List[Dict]):
        """Execute profitable arbitrage opportunities."""
        opportunity_timeout = self.config.get('opportunity_timeout', 5.0)
        
        for opportunity in opportunities[:10]:  # Limit to top 10
            # Check if opportunity is still valid
            if not self._check_opportunity_validity(opportunity, opportunity_timeout):
                continue
            
            # Calculate position size based on expected profit
            position_size = self._calculate_position_size(opportunity)
            
            if position_size <= 0:
                continue
            
            # Execute arbitrage
            try:
                if self.execution_engine:
                    # Execute on slow exchange first (to avoid slippage)
                    buy_order = await self.execution_engine.execute_order(
                        opportunity['symbol'],
                        'buy',
                        position_size,
                        opportunity['buy_price'],
                        'aggressive'
                    )
                    
                    # Execute on fast exchange
                    sell_order = await self.execution_engine.execute_order(
                        opportunity['symbol'],
                        'sell',
                        position_size,
                        opportunity['sell_price'],
                        'aggressive'
                    )
                    
                    # Record trade
                    self._record_trade(opportunity, buy_order, sell_order)
            except Exception as e:
                logger.error(f"Arbitrage execution failed: {e}")
    
    def _check_opportunity_validity(self, opportunity: Dict, timeout: float) -> bool:
        """Check if arbitrage opportunity is still valid."""
        # Check if opportunity is too old
        if time.time() - opportunity['timestamp'] > timeout:
            return False
        
        # Check current prices
        try:
            buy_exchange = self.exchanges.get_exchange(opportunity['buy_exchange'])
            sell_exchange = self.exchanges.get_exchange(opportunity['sell_exchange'])
            
            current_buy_price = buy_exchange.get_price(opportunity['symbol'])
            current_sell_price = sell_exchange.get_price(opportunity['symbol'])
            
            # Check if profit still exists
            min_profit = self.config.get('min_profit', 0.005)
            current_profit = (current_sell_price - current_buy_price) / current_buy_price
            
            if current_profit <= min_profit:
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Failed to check opportunity validity: {e}")
            return False
    
    def _calculate_position_size(self, opportunity: Dict) -> float:
        """Calculate optimal position size for arbitrage opportunity."""
        base_position_size = self.config.get('base_position_size', 0.1)
        max_position_size = self.config.get('max_position_size', 1.0)
        
        # Base position size
        position_size = base_position_size
        
        # Adjust for expected profit
        position_size *= (1 + opportunity['expected_profit'] * 10)
        
        # Adjust for execution probability
        position_size *= opportunity['execution_prob']
        
        # Adjust for latency
        latency_threshold = self.config.get('latency_threshold', 100.0)
        position_size *= np.exp(-opportunity['latency'] / latency_threshold)
        
        # Ensure position size is within limits
        position_size = min(position_size, max_position_size)
        
        return float(position_size)
    
    def _record_trade(self, opportunity: Dict, buy_order: Dict, sell_order: Dict):
        """Record arbitrage trade."""
        if 'error' not in buy_order and 'error' not in sell_order:
            buy_price = buy_order.get('price', opportunity['buy_price'])
            sell_price = sell_order.get('price', opportunity['sell_price'])
            amount = buy_order.get('filled', 0)
            actual_profit = (sell_price - buy_price) * amount
            
            logger.info(
                f"Arbitrage executed: {opportunity['symbol']} "
                f"Buy: {amount}@{buy_price:.2f} on {opportunity['buy_exchange']}, "
                f"Sell: {amount}@{sell_price:.2f} on {opportunity['sell_exchange']}, "
                f"Profit: {actual_profit:.2f} "
                f"({actual_profit/buy_price/amount*100:.2f}%)"
            )
        else:
            logger.warning(
                f"Arbitrage failed: {opportunity['symbol']} - "
                f"Buy error: {buy_order.get('error', 'None')}, "
                f"Sell error: {sell_order.get('error', 'None')}"
            )

