"""
Latency Arbitrage Strategy
Exploits latency differences between exchanges
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

from exchange.ccxt_adapter import CCXTAdapter

logger = logging.getLogger(__name__)


@dataclass
class LatencyOpportunity:
    """Latency arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    latency_diff_ms: float


class LatencyArbitrage:
    """
    Latency arbitrage strategy.
    
    Exploits differences in latency between exchanges to capture
    price discrepancies before they disappear.
    """
    
    def __init__(
        self,
        adapter: CCXTAdapter,
        config: Optional[Dict] = None
    ):
        """
        Initialize latency arbitrage.
        
        Args:
            adapter: CCXT adapter
            config: Configuration dictionary
        """
        self.adapter = adapter
        self.config = config or {}
        
        self.min_profit = self.config.get('min_profit', 0.01)  # 1%
        self.max_position = self.config.get('max_position', 1000.0)
        self.latency_threshold = self.config.get('latency_threshold', 50.0)  # ms
        
        self.order_book_cache: Dict[str, Dict] = {}
    
    def check_opportunity(
        self,
        symbol: str,
        exchanges: List[str]
    ) -> Optional[LatencyOpportunity]:
        """
        Check for latency arbitrage opportunities.
        
        Args:
            symbol: Trading symbol
            exchanges: List of exchange names to check
            
        Returns:
            Latency opportunity or None
        """
        order_books = {}
        timestamps = {}
        
        # Fetch order books and measure latency
        for exchange_name in exchanges:
            try:
                start = time.perf_counter_ns()
                order_book = self.adapter.exchange_manager.get_exchange(
                    exchange_name
                ).fetch_order_book(symbol, limit=5)
                end = time.perf_counter_ns()
                
                order_books[exchange_name] = order_book
                timestamps[exchange_name] = (end - start) / 1_000_000  # Convert to ms
                
            except Exception as e:
                logger.warning(f"Failed to fetch order book from {exchange_name}: {e}")
                continue
        
        if len(order_books) < 2:
            return None
        
        # Find exchanges with significant latency differences
        fast_ex = min(timestamps, key=timestamps.get)
        slow_ex = max(timestamps, key=timestamps.get)
        
        latency_diff = timestamps[slow_ex] - timestamps[fast_ex]
        
        if latency_diff > self.latency_threshold:
            # Get best prices
            fast_bid = order_books[fast_ex]['bids'][0][0] if order_books[fast_ex]['bids'] else None
            slow_ask = order_books[slow_ex]['asks'][0][0] if order_books[slow_ex]['asks'] else None
            
            if fast_bid and slow_ask:
                profit_pct = (fast_bid - slow_ask) / slow_ask
                
                if profit_pct > self.min_profit:
                    return LatencyOpportunity(
                        symbol=symbol,
                        buy_exchange=slow_ex,
                        sell_exchange=fast_ex,
                        buy_price=slow_ask,
                        sell_price=fast_bid,
                        profit_pct=profit_pct,
                        latency_diff_ms=latency_diff
                    )
        
        return None
    
    async def execute(
        self,
        opportunity: LatencyOpportunity,
        amount: Optional[float] = None
    ) -> Dict:
        """
        Execute latency arbitrage trade.
        
        Args:
            opportunity: Latency opportunity
            amount: Trade amount (auto-calculated if None)
            
        Returns:
            Execution result dictionary
        """
        if amount is None:
            # Calculate position size based on latency difference
            amount = self.max_position * (
                opportunity.latency_diff_ms / self.latency_threshold
            )
            amount = min(amount, self.max_position)
        
        try:
            # Execute on slow exchange first (to avoid slippage)
            buy_order = await self.adapter.create_order(
                exchange_name=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                side='buy',
                amount=amount,
                order_type='market'
            )
            
            if not buy_order:
                return {'success': False, 'error': 'Buy order failed'}
            
            # Execute on fast exchange
            sell_order = await self.adapter.create_order(
                exchange_name=opportunity.sell_exchange,
                symbol=opportunity.symbol,
                side='sell',
                amount=amount,
                order_type='market'
            )
            
            if not sell_order:
                return {'success': False, 'error': 'Sell order failed'}
            
            logger.info(
                f"✅ Latency arbitrage executed: {opportunity.symbol} "
                f"Buy {opportunity.buy_exchange} @ {opportunity.buy_price:.2f}, "
                f"Sell {opportunity.sell_exchange} @ {opportunity.sell_price:.2f}, "
                f"Latency diff: {opportunity.latency_diff_ms:.2f}ms"
            )
            
            return {
                'success': True,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'profit_pct': opportunity.profit_pct,
                'latency_diff_ms': opportunity.latency_diff_ms
            }
        
        except Exception as e:
            logger.error(f"Error executing latency arbitrage: {e}")
            return {'success': False, 'error': str(e)}

