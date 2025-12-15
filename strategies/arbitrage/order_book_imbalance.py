"""
Order Book Imbalance Arbitrage
Trades based on order book imbalance signals
"""
from __future__ import annotations

import logging
from typing import Dict, Optional
from dataclasses import dataclass

from exchange.ccxt_adapter import CCXTAdapter

logger = logging.getLogger(__name__)


@dataclass
class ImbalanceOpportunity:
    """Order book imbalance opportunity."""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    imbalance: float
    confidence: float


class OrderBookImbalanceArbitrage:
    """
    Order book imbalance arbitrage strategy.
    
    Trades based on order book imbalances, which can indicate
    short-term price movements.
    """
    
    def __init__(
        self,
        adapter: CCXTAdapter,
        config: Optional[Dict] = None
    ):
        """
        Initialize order book imbalance strategy.
        
        Args:
            adapter: CCXT adapter
            config: Configuration dictionary
        """
        self.adapter = adapter
        self.config = config or {}
        
        self.min_imbalance = self.config.get('min_imbalance', 0.2)  # 20%
        self.max_position = self.config.get('max_position', 1000.0)
        self.order_book_depth = self.config.get('order_book_depth', 20)
    
    def check_opportunity(
        self,
        exchange_name: str,
        symbol: str
    ) -> Optional[ImbalanceOpportunity]:
        """
        Check for order book imbalance opportunities.
        
        Args:
            exchange_name: Exchange name
            symbol: Trading symbol
            
        Returns:
            Imbalance opportunity or None
        """
        try:
            exchange = self.adapter.exchange_manager.get_exchange(exchange_name)
            order_book = exchange.fetch_order_book(symbol, limit=self.order_book_depth)
            
            # Calculate bid-ask imbalance
            total_bid = sum([x[1] for x in order_book['bids']])
            total_ask = sum([x[1] for x in order_book['asks']])
            
            if total_bid + total_ask == 0:
                return None
            
            imbalance = (total_bid - total_ask) / (total_bid + total_ask)
            
            if abs(imbalance) > self.min_imbalance:
                # Calculate fair price (mid price)
                mid_price = (
                    order_book['bids'][0][0] + order_book['asks'][0][0]
                ) / 2
                
                # Determine trade direction
                if imbalance > 0:
                    # More bids than asks - price likely to rise
                    return ImbalanceOpportunity(
                        symbol=symbol,
                        side='buy',
                        price=order_book['asks'][0][0],  # Buy at ask
                        imbalance=imbalance,
                        confidence=min(abs(imbalance) / 0.5, 1.0)  # Normalize to 0-1
                    )
                else:
                    # More asks than bids - price likely to fall
                    return ImbalanceOpportunity(
                        symbol=symbol,
                        side='sell',
                        price=order_book['bids'][0][0],  # Sell at bid
                        imbalance=imbalance,
                        confidence=min(abs(imbalance) / 0.5, 1.0)
                    )
        
        except Exception as e:
            logger.warning(f"Error checking order book imbalance: {e}")
        
        return None
    
    async def execute(
        self,
        opportunity: ImbalanceOpportunity,
        exchange_name: str,
        amount: Optional[float] = None
    ) -> Dict:
        """
        Execute order book imbalance trade.
        
        Args:
            opportunity: Imbalance opportunity
            exchange_name: Exchange name
            amount: Trade amount (auto-calculated if None)
            
        Returns:
            Execution result dictionary
        """
        if amount is None:
            # Calculate position size based on imbalance
            amount = self.max_position * (abs(opportunity.imbalance) / 0.5)
            amount = min(amount, self.max_position)
        
        try:
            order = await self.adapter.create_order(
                exchange_name=exchange_name,
                symbol=opportunity.symbol,
                side=opportunity.side,
                amount=amount,
                order_type='limit',
                price=opportunity.price
            )
            
            if order:
                logger.info(
                    f"✅ Order book imbalance trade executed: {opportunity.symbol} "
                    f"{opportunity.side} @ {opportunity.price:.2f}, "
                    f"imbalance: {opportunity.imbalance:.2%}"
                )
            
            return {
                'success': order is not None,
                'order': order,
                'imbalance': opportunity.imbalance,
                'confidence': opportunity.confidence
            }
        
        except Exception as e:
            logger.error(f"Error executing order book imbalance trade: {e}")
            return {'success': False, 'error': str(e)}

