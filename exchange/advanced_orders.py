"""
Advanced Order Types Support
Stop-Loss, OCO, Trailing Stops, and other advanced order types
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Any
import ccxt

logger = logging.getLogger(__name__)


class AdvancedOrderManager:
    """
    Manager for advanced order types across multiple exchanges.
    
    Supported order types:
    - Stop-Loss
    - Take-Profit
    - OCO (One-Cancels-the-Other)
    - Trailing Stop
    - Stop-Limit
    """
    
    def __init__(self, exchange_manager):
        """
        Initialize advanced order manager.
        
        Args:
            exchange_manager: UnifiedExchangeManager instance
        """
        self.exchange_manager = exchange_manager
    
    async def create_stop_loss(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        limit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a stop-loss order.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order amount
            stop_price: Price that triggers the stop order
            limit_price: Optional limit price (for stop-limit orders)
            
        Returns:
            Order response or None on error
        """
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            logger.error(f"Exchange {exchange_name} not found")
            return None
        
        try:
            params = {'stopPrice': stop_price}
            
            if limit_price:
                # Stop-limit order
                order_type = 'stop_limit'
                params['stopLimitPrice'] = limit_price
            else:
                # Stop-market order
                order_type = 'stop'
            
            order = exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=stop_price,
                params=params
            )
            
            logger.info(f"✅ Stop-loss order placed on {exchange_name}: {side} {amount} {symbol} @ {stop_price}")
            return order
        
        except ccxt.NotSupported as e:
            logger.error(f"Stop-loss not supported on {exchange_name}: {e}")
            # Fallback: Use conditional order if available
            return await self._create_conditional_order(
                exchange_name, symbol, side, amount, stop_price, 'stop'
            )
        except Exception as e:
            logger.error(f"Error creating stop-loss on {exchange_name}: {e}")
            return None
    
    async def create_take_profit(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        take_profit_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Create a take-profit order.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order amount
            take_profit_price: Price at which to take profit
            
        Returns:
            Order response or None on error
        """
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            return None
        
        try:
            # Take-profit is typically a limit order
            order = exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=take_profit_price
            )
            
            logger.info(f"✅ Take-profit order placed on {exchange_name}: {side} {amount} {symbol} @ {take_profit_price}")
            return order
        
        except Exception as e:
            logger.error(f"Error creating take-profit on {exchange_name}: {e}")
            return None
    
    async def create_oco_order(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create an OCO (One-Cancels-the-Other) order.
        
        OCO orders place two orders simultaneously:
        - A limit order (take-profit)
        - A stop-loss order
        
        When one executes, the other is cancelled.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit order price (take-profit)
            stop_price: Stop order trigger price
            stop_limit_price: Optional stop-limit price
            
        Returns:
            Order response or None on error
        """
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            return None
        
        try:
            params = {
                'stopPrice': stop_price,
                'price': price,
            }
            
            if stop_limit_price:
                params['stopLimitPrice'] = stop_limit_price
            
            order = exchange.create_order(
                symbol=symbol,
                type='oco',
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            logger.info(f"✅ OCO order placed on {exchange_name}: {side} {amount} {symbol}")
            return order
        
        except ccxt.NotSupported as e:
            logger.error(f"OCO not supported on {exchange_name}: {e}")
            # Fallback: Place both orders separately (not true OCO, but functional)
            logger.warning("Placing stop-loss and take-profit separately as fallback")
            stop_order = await self.create_stop_loss(exchange_name, symbol, side, amount, stop_price, stop_limit_price)
            tp_order = await self.create_take_profit(exchange_name, symbol, side, amount, price)
            return {'stop_order': stop_order, 'take_profit_order': tp_order}
        except Exception as e:
            logger.error(f"Error creating OCO order on {exchange_name}: {e}")
            return None
    
    async def create_trailing_stop(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        trailing_percent: Optional[float] = None,
        trailing_distance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a trailing stop order.
        
        Trailing stops automatically adjust the stop price as the market moves favorably.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order amount
            trailing_percent: Trailing stop percentage (e.g., 5.0 for 5%)
            trailing_distance: Trailing stop distance in price units
            
        Returns:
            Order response or None on error
        """
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            return None
        
        try:
            params = {}
            
            if trailing_percent:
                params['trailingPercent'] = trailing_percent
            elif trailing_distance:
                params['trailingDistance'] = trailing_distance
            else:
                logger.error("Either trailing_percent or trailing_distance must be provided")
                return None
            
            order = exchange.create_order(
                symbol=symbol,
                type='trailing_stop',
                side=side,
                amount=amount,
                params=params
            )
            
            logger.info(f"✅ Trailing stop placed on {exchange_name}: {side} {amount} {symbol} ({trailing_percent}%)")
            return order
        
        except ccxt.NotSupported as e:
            logger.error(f"Trailing stop not supported on {exchange_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating trailing stop on {exchange_name}: {e}")
            return None
    
    async def _create_conditional_order(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        trigger_price: float,
        order_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback method for creating conditional orders on exchanges
        that don't support advanced order types natively.
        """
        logger.warning(f"Using fallback conditional order method for {exchange_name}")
        # This would require implementing a monitoring system that watches
        # the price and places orders when conditions are met
        # For now, return None to indicate it's not implemented
        return None
    
    def get_supported_order_types(self, exchange_name: str) -> Dict[str, bool]:
        """
        Get list of supported order types for an exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            Dictionary mapping order type names to support status
        """
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            return {}
        
        return {
            'market': True,  # All exchanges support market orders
            'limit': True,  # All exchanges support limit orders
            'stop': self.exchange_manager.check_exchange_support(exchange_name, 'stopLoss'),
            'stop_limit': self.exchange_manager.check_exchange_support(exchange_name, 'stopLoss'),
            'trailing_stop': self.exchange_manager.check_exchange_support(exchange_name, 'trailingStop'),
            'oco': self.exchange_manager.check_exchange_support(exchange_name, 'oco'),
        }

