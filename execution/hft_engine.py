"""
High-Frequency Execution Engine
Ultra-low latency order execution with multiple strategies
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
    logger.warning("Low latency engine not available")


class HFTExecutionEngine:
    """
    High-frequency execution engine for ultra-low latency trading.
    
    Features:
    - Aggressive execution
    - Passive execution
    - Iceberg orders
    - Smart execution with market impact calculation
    - Execution quality monitoring
    """
    
    def __init__(self, exchange: Any, config: Dict):
        """
        Initialize HFT execution engine.
        
        Args:
            exchange: Exchange instance
            config: Configuration dictionary
        """
        self.exchange = exchange
        self.config = config
        
        if LOW_LATENCY_AVAILABLE:
            self.latency_engine = LowLatencyEngine(config)
        else:
            self.latency_engine = None
        
        self.order_book_cache: Dict[str, Dict] = {}
        self.active_orders: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.last_execution_time = 0
        
        self.execution_quality = {
            'slippage': [],
            'latency': [],
            'fill_rate': []
        }
    
    async def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        strategy: str = 'aggressive'
    ) -> Dict:
        """
        Execute order with HFT techniques.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (None for market orders)
            strategy: Execution strategy ('aggressive', 'passive', 'iceberg', 'smart')
            
        Returns:
            Execution result
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Get current market conditions
            market_data = await self._get_market_data(symbol)
            
            # Determine execution strategy
            if strategy == 'aggressive':
                execution_plan = self._aggressive_execution(market_data, side, amount, price)
            elif strategy == 'passive':
                execution_plan = self._passive_execution(market_data, side, amount, price)
            elif strategy == 'iceberg':
                execution_plan = self._iceberg_execution(market_data, side, amount, price)
            else:
                execution_plan = self._smart_execution(market_data, side, amount, price)
            
            # Execute order
            result = await self._execute_plan(execution_plan)
            
            # Record execution quality
            self._record_execution_quality(start_time, result, market_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data with minimal latency."""
        # Check cache first
        if symbol in self.order_book_cache:
            cached_data = self.order_book_cache[symbol]
            if time.time() - cached_data['timestamp'] < 0.1:  # 100ms cache
                return cached_data['data']
        
        # Fetch fresh data
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit=10)
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Update cache
            self.order_book_cache[symbol] = {
                'data': {
                    'order_book': order_book,
                    'ticker': ticker,
                    'timestamp': time.time()
                },
                'timestamp': time.time()
            }
            
            return self.order_book_cache[symbol]['data']
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return {'order_book': {'bids': [], 'asks': []}, 'ticker': {}}
    
    def _aggressive_execution(
        self,
        market_data: Dict,
        side: str,
        amount: float,
        price: Optional[float]
    ) -> List[Dict]:
        """Create aggressive execution plan."""
        order_book = market_data.get('order_book', {})
        ticker = market_data.get('ticker', {})
        
        if price is None:
            # Market order
            if side == 'buy':
                price = (
                    order_book['asks'][0][0]
                    if order_book.get('asks') and len(order_book['asks']) > 0
                    else ticker.get('ask', 0)
                )
            else:
                price = (
                    order_book['bids'][0][0]
                    if order_book.get('bids') and len(order_book['bids']) > 0
                    else ticker.get('bid', 0)
                )
            return [{
                'symbol': order_book.get('symbol', ''),
                'side': side,
                'amount': amount,
                'price': price,
                'type': 'market'
            }]
        else:
            # Aggressive limit order
            return [{
                'symbol': order_book.get('symbol', ''),
                'side': side,
                'amount': amount,
                'price': price,
                'type': 'limit',
                'post_only': False
            }]
    
    def _passive_execution(
        self,
        market_data: Dict,
        side: str,
        amount: float,
        price: Optional[float]
    ) -> List[Dict]:
        """Create passive execution plan."""
        order_book = market_data.get('order_book', {})
        ticker = market_data.get('ticker', {})
        
        if price is None:
            # Passive limit order at best bid/ask
            if side == 'buy':
                price = (
                    order_book['bids'][0][0]
                    if order_book.get('bids') and len(order_book['bids']) > 0
                    else ticker.get('bid', 0)
                )
            else:
                price = (
                    order_book['asks'][0][0]
                    if order_book.get('asks') and len(order_book['asks']) > 0
                    else ticker.get('ask', 0)
                )
        else:
            # Ensure price is passive
            if side == 'buy' and order_book.get('asks') and len(order_book['asks']) > 0:
                if price >= order_book['asks'][0][0]:
                    price = order_book['bids'][0][0] if order_book.get('bids') else ticker.get('bid', 0)
            elif side == 'sell' and order_book.get('bids') and len(order_book['bids']) > 0:
                if price <= order_book['bids'][0][0]:
                    price = order_book['asks'][0][0] if order_book.get('asks') else ticker.get('ask', 0)
        
        return [{
            'symbol': order_book.get('symbol', ''),
            'side': side,
            'amount': amount,
            'price': price,
            'type': 'limit',
            'post_only': True
        }]
    
    def _iceberg_execution(
        self,
        market_data: Dict,
        side: str,
        amount: float,
        price: Optional[float]
    ) -> List[Dict]:
        """Create iceberg execution plan."""
        order_book = market_data.get('order_book', {})
        ticker = market_data.get('ticker', {})
        
        if price is None:
            if side == 'buy':
                price = (
                    order_book['asks'][0][0]
                    if order_book.get('asks') and len(order_book['asks']) > 0
                    else ticker.get('ask', 0)
                )
            else:
                price = (
                    order_book['bids'][0][0]
                    if order_book.get('bids') and len(order_book['bids']) > 0
                    else ticker.get('bid', 0)
                )
        
        # Split into smaller orders
        visible_size = min(amount * 0.2, 0.1)  # 20% of order or 0.1 BTC max
        hidden_size = amount - visible_size
        
        return [{
            'symbol': order_book.get('symbol', ''),
            'side': side,
            'amount': visible_size,
            'price': price,
            'type': 'limit',
            'post_only': True,
            'iceberg': True,
            'hidden_size': hidden_size
        }]
    
    def _smart_execution(
        self,
        market_data: Dict,
        side: str,
        amount: float,
        price: Optional[float]
    ) -> List[Dict]:
        """Create smart execution plan based on market conditions."""
        order_book = market_data.get('order_book', {})
        ticker = market_data.get('ticker', {})
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(order_book, side, amount)
        
        if price is None:
            # Market order with slippage control
            if side == 'buy':
                price = (
                    order_book['asks'][0][0] * (1 + market_impact)
                    if order_book.get('asks') and len(order_book['asks']) > 0
                    else ticker.get('ask', 0)
                )
            else:
                price = (
                    order_book['bids'][0][0] * (1 - market_impact)
                    if order_book.get('bids') and len(order_book['bids']) > 0
                    else ticker.get('bid', 0)
                )
            
            return [{
                'symbol': order_book.get('symbol', ''),
                'side': side,
                'amount': amount,
                'price': price,
                'type': 'market',
                'max_slippage': market_impact
            }]
        else:
            # Smart limit order
            if side == 'buy':
                best_ask = (
                    order_book['asks'][0][0]
                    if order_book.get('asks') and len(order_book['asks']) > 0
                    else ticker.get('ask', 0)
                )
                if price > best_ask * (1 + market_impact):
                    # Aggressive limit order
                    return [{
                        'symbol': order_book.get('symbol', ''),
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'type': 'limit',
                        'post_only': False
                    }]
                else:
                    # Passive limit order
                    return [{
                        'symbol': order_book.get('symbol', ''),
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'type': 'limit',
                        'post_only': True
                    }]
            else:
                best_bid = (
                    order_book['bids'][0][0]
                    if order_book.get('bids') and len(order_book['bids']) > 0
                    else ticker.get('bid', 0)
                )
                if price < best_bid * (1 - market_impact):
                    # Aggressive limit order
                    return [{
                        'symbol': order_book.get('symbol', ''),
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'type': 'limit',
                        'post_only': False
                    }]
                else:
                    # Passive limit order
                    return [{
                        'symbol': order_book.get('symbol', ''),
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'type': 'limit',
                        'post_only': True
                    }]
    
    def _calculate_market_impact(self, order_book: Dict, side: str, amount: float) -> float:
        """Calculate expected market impact."""
        if side == 'buy':
            book = order_book.get('asks', [])
        else:
            book = order_book.get('bids', [])
        
        if not book:
            return 0.01  # 1% default impact
        
        total_volume = 0.0
        impact = 0.0
        remaining = amount
        
        for price, volume in book:
            if remaining <= 0:
                break
            
            take_volume = min(volume, remaining)
            if len(book) > 0:
                impact += (
                    (price * take_volume - book[0][0] * take_volume) /
                    (book[0][0] * take_volume)
                )
            remaining -= take_volume
            total_volume += volume
        
        if remaining > 0:
            # If order is larger than book depth, add additional impact
            impact += 0.01 * (remaining / amount)
        
        return min(max(impact, 0.0), 0.1)  # Cap at 10%
    
    async def _execute_plan(self, execution_plan: List[Dict]) -> Dict:
        """Execute the execution plan."""
        results = []
        
        for order in execution_plan:
            try:
                if order['type'] == 'market':
                    result = await self.exchange.create_market_order(
                        order['symbol'],
                        order['side'],
                        order['amount']
                    )
                else:
                    result = await self.exchange.create_limit_order(
                        order['symbol'],
                        order['side'],
                        order['amount'],
                        order['price'],
                        post_only=order.get('post_only', False)
                    )
                
                # Handle iceberg orders
                if order.get('iceberg', False):
                    result = await self._execute_iceberg_order(order, result)
                
                results.append(result)
                if 'id' in result:
                    self.active_orders[result['id']] = result
            except Exception as e:
                results.append({'error': str(e), 'order': order})
        
        # Return combined result
        if len(results) == 1:
            return results[0]
        else:
            return {
                'orders': results,
                'status': 'partial' if any('error' in r for r in results) else 'complete'
            }
    
    async def _execute_iceberg_order(self, order: Dict, initial_result: Dict) -> Dict:
        """Execute iceberg order."""
        remaining = order.get('hidden_size', 0)
        visible_size = order['amount']
        
        while remaining > 0:
            # Wait for the visible portion to fill
            if 'id' in initial_result:
                filled = await self._wait_for_fill(initial_result['id'])
                if not filled:
                    break
            
            # Place next visible order
            next_size = min(visible_size, remaining)
            try:
                result = await self.exchange.create_limit_order(
                    order['symbol'],
                    order['side'],
                    next_size,
                    order['price'],
                    post_only=True
                )
                remaining -= next_size
                if 'filled' in initial_result:
                    initial_result['filled'] += next_size
                else:
                    initial_result['filled'] = next_size
                initial_result['status'] = 'partial'
            except Exception as e:
                initial_result['error'] = str(e)
                break
        
        initial_result['status'] = 'closed' if remaining <= 0 else 'partial'
        return initial_result
    
    async def _wait_for_fill(self, order_id: str, timeout: float = 10.0) -> bool:
        """Wait for order to fill."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order = await self.exchange.fetch_order(order_id)
                if order.get('status') == 'closed':
                    return True
                await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(0.1)
        return False
    
    def _record_execution_quality(
        self,
        start_time: float,
        result: Dict,
        market_data: Dict
    ):
        """Record execution quality metrics."""
        end_time = time.perf_counter_ns()
        latency = (end_time - start_time) / 1_000_000  # ms
        
        if 'error' not in result:
            # Calculate slippage
            order_book = market_data.get('order_book', {})
            ticker = market_data.get('ticker', {})
            
            if result.get('type') == 'market':
                if result.get('side') == 'buy':
                    expected_price = (
                        order_book['asks'][0][0]
                        if order_book.get('asks') and len(order_book['asks']) > 0
                        else ticker.get('ask', 0)
                    )
                else:
                    expected_price = (
                        order_book['bids'][0][0]
                        if order_book.get('bids') and len(order_book['bids']) > 0
                        else ticker.get('bid', 0)
                    )
            else:
                expected_price = result.get('price', 0)
            
            actual_price = result.get('price', expected_price)
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
            else:
                slippage = 0.0
            
            # Calculate fill rate
            filled = result.get('filled', 0)
            amount = result.get('amount', 1)
            fill_rate = filled / amount if amount > 0 else 0.0
            
            # Record metrics
            self.execution_quality['slippage'].append(slippage)
            self.execution_quality['latency'].append(latency)
            self.execution_quality['fill_rate'].append(fill_rate)
            
            # Keep only recent metrics
            for metric in self.execution_quality:
                if len(self.execution_quality[metric]) > 100:
                    self.execution_quality[metric].pop(0)
    
    def get_execution_stats(self) -> Dict:
        """Get execution quality statistics."""
        stats = {}
        for metric, values in self.execution_quality.items():
            if values:
                stats[metric] = {
                    'avg': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                stats[metric] = {
                    'avg': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
        return stats

