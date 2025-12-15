"""
Autonomous Market Making Engine
Dynamic market making with adaptive spreads and inventory management
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from execution.hft_engine import HFTExecutionEngine
    HFT_ENGINE_AVAILABLE = True
except ImportError:
    HFT_ENGINE_AVAILABLE = False
    logger.warning("HFT execution engine not available")

try:
    from core.low_latency import LowLatencyEngine
    LOW_LATENCY_AVAILABLE = True
except ImportError:
    LOW_LATENCY_AVAILABLE = False


class AutonomousMarketMaker:
    """
    Autonomous market making engine.
    
    Features:
    - Dynamic spread adjustment
    - Inventory management
    - Volatility-based pricing
    - HFT execution
    """
    
    def __init__(
        self,
        exchange: Any,
        symbols: List[str],
        config: Dict
    ):
        """
        Initialize autonomous market maker.
        
        Args:
            exchange: Exchange instance
            symbols: List of symbols to market make
            config: Configuration dictionary
        """
        self.exchange = exchange
        self.symbols = symbols
        self.config = config
        
        if HFT_ENGINE_AVAILABLE:
            self.hft_engine = HFTExecutionEngine(exchange, config)
        else:
            self.hft_engine = None
        
        if LOW_LATENCY_AVAILABLE:
            self.latency_engine = LowLatencyEngine(config)
        else:
            self.latency_engine = None
        
        self.order_books: Dict[str, Dict] = {
            symbol: {'bids': [], 'asks': []} for symbol in symbols
        }
        self.active_orders: Dict[str, Dict] = {
            symbol: {'bids': [], 'asks': []} for symbol in symbols
        }
        self.inventory: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.pnl: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        
        # Initialize models
        self.spread_model = self._initialize_spread_model()
        self.inventory_model = self._initialize_inventory_model()
        self.mid_price_model = self._initialize_mid_price_model()
        
        self.running = False
    
    def _initialize_spread_model(self):
        """Initialize spread determination model."""
        base_spread = self.config.get('base_spread', 0.001)
        return lambda symbol, volatility: base_spread * (1 + volatility)
    
    def _initialize_inventory_model(self):
        """Initialize inventory management model."""
        inventory_threshold = self.config.get('inventory_threshold', 1.0)
        return lambda symbol, inventory: np.tanh(inventory / inventory_threshold)
    
    def _initialize_mid_price_model(self):
        """Initialize mid price prediction model."""
        return lambda symbol, order_book: (
            (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            if order_book.get('bids') and order_book.get('asks')
            and len(order_book['bids']) > 0 and len(order_book['asks']) > 0
            else 0.0
        )
    
    async def run(self):
        """Run the market making loop."""
        self.running = True
        logger.info("Starting autonomous market maker...")
        
        while self.running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Adjust spreads and inventory
                await self._adjust_parameters()
                
                # Place orders
                await self._place_orders()
                
                # Manage existing orders
                await self._manage_orders()
                
                # Sleep for refresh interval
                refresh_interval = self.config.get('refresh_interval', 1.0)
                await asyncio.sleep(refresh_interval)
                
            except Exception as e:
                logger.error(f"Market making error: {e}")
                await asyncio.sleep(5)
    
    async def _update_market_data(self):
        """Update market data for all symbols."""
        for symbol in self.symbols:
            try:
                # Get order book
                order_book = await self.exchange.fetch_order_book(symbol, limit=10)
                self.order_books[symbol] = order_book
                
                # Get recent trades
                trades = await self.exchange.fetch_trades(symbol, limit=100)
                self._update_inventory(symbol, trades)
                
                # Get ticker
                ticker = await self.exchange.fetch_ticker(symbol)
                
                # Update PnL
                self._update_pnl(symbol, ticker)
            except Exception as e:
                logger.warning(f"Failed to update market data for {symbol}: {e}")
    
    def _update_inventory(self, symbol: str, trades: List[Dict]):
        """Update inventory based on recent trades."""
        for trade in trades:
            if trade.get('side') == 'buy':
                self.inventory[symbol] += trade.get('amount', 0.0)
            else:
                self.inventory[symbol] -= trade.get('amount', 0.0)
    
    def _update_pnl(self, symbol: str, ticker: Dict):
        """Update PnL for a symbol."""
        last_price = ticker.get('last', 0.0)
        bid = ticker.get('bid', last_price)
        ask = ticker.get('ask', last_price)
        
        if self.inventory[symbol] > 0:
            self.pnl[symbol] = self.inventory[symbol] * (last_price - bid)
        elif self.inventory[symbol] < 0:
            self.pnl[symbol] = self.inventory[symbol] * (last_price - ask)
        else:
            self.pnl[symbol] = 0.0
    
    async def _adjust_parameters(self):
        """Adjust market making parameters based on current conditions."""
        for symbol in self.symbols:
            try:
                order_book = self.order_books[symbol]
                volatility = self._calculate_volatility(symbol)
                inventory = self.inventory[symbol]
                
                # Adjust spread
                if 'spreads' not in self.config:
                    self.config['spreads'] = {}
                self.config['spreads'][symbol] = self.spread_model(symbol, volatility)
                
                # Adjust inventory target
                if 'inventory_targets' not in self.config:
                    self.config['inventory_targets'] = {}
                inventory_adjustment = self.inventory_model(symbol, inventory)
                self.config['inventory_targets'][symbol] = inventory_adjustment
            except Exception as e:
                logger.warning(f"Failed to adjust parameters for {symbol}: {e}")
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility for a symbol."""
        try:
            trades = self.exchange.fetch_trades(symbol, limit=100)
            
            if len(trades) < 2:
                return 0.02  # Default 2% volatility
            
            # Calculate returns
            prices = [trade.get('price', 0.0) for trade in trades]
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate volatility
            return float(np.std(returns) * np.sqrt(252))  # Annualized
        except Exception:
            return 0.02
    
    async def _place_orders(self):
        """Place new orders for all symbols."""
        for symbol in self.symbols:
            try:
                # Cancel existing orders
                await self._cancel_orders(symbol)
                
                # Get current market conditions
                order_book = self.order_books[symbol]
                mid_price = self.mid_price_model(symbol, order_book)
                
                if mid_price == 0:
                    continue
                
                spread = self.config.get('spreads', {}).get(symbol, 0.001)
                inventory = self.inventory[symbol]
                inventory_target = self.config.get('inventory_targets', {}).get(symbol, 0.0)
                
                # Calculate order sizes
                base_order_size = self.config.get('order_size', 0.1)
                bid_size = base_order_size * (1 - inventory_target)
                ask_size = base_order_size * (1 + inventory_target)
                
                # Place bid order
                if bid_size > 0 and self.hft_engine:
                    bid_price = mid_price * (1 - spread / 2)
                    await self.hft_engine.execute_order(
                        symbol, 'buy', bid_size, bid_price, 'passive'
                    )
                
                # Place ask order
                if ask_size > 0 and self.hft_engine:
                    ask_price = mid_price * (1 + spread / 2)
                    await self.hft_engine.execute_order(
                        symbol, 'sell', ask_size, ask_price, 'passive'
                    )
            except Exception as e:
                logger.warning(f"Failed to place orders for {symbol}: {e}")
    
    async def _cancel_orders(self, symbol: str):
        """Cancel all active orders for a symbol."""
        for order in (
            self.active_orders[symbol]['bids'] + self.active_orders[symbol]['asks']
        ):
            try:
                await self.exchange.cancel_order(order.get('id'), symbol)
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.get('id')}: {e}")
        
        self.active_orders[symbol] = {'bids': [], 'asks': []}
    
    async def _manage_orders(self):
        """Manage existing orders."""
        for symbol in self.symbols:
            try:
                order_book = self.order_books[symbol]
                
                if not order_book.get('bids') or not order_book.get('asks'):
                    continue
                
                # Check bid orders
                for order in self.active_orders[symbol]['bids']:
                    if order.get('price', 0) < order_book['bids'][0][0] * 0.99:
                        # Order is too far from best bid, cancel and replace
                        try:
                            await self.exchange.cancel_order(order.get('id'), symbol)
                            self.active_orders[symbol]['bids'].remove(order)
                        except Exception as e:
                            logger.warning(f"Failed to cancel bid order: {e}")
                
                # Check ask orders
                for order in self.active_orders[symbol]['asks']:
                    if order.get('price', 0) > order_book['asks'][0][0] * 1.01:
                        # Order is too far from best ask, cancel and replace
                        try:
                            await self.exchange.cancel_order(order.get('id'), symbol)
                            self.active_orders[symbol]['asks'].remove(order)
                        except Exception as e:
                            logger.warning(f"Failed to cancel ask order: {e}")
            except Exception as e:
                logger.warning(f"Failed to manage orders for {symbol}: {e}")
    
    def stop(self):
        """Stop the market maker."""
        self.running = False
        logger.info("Stopping autonomous market maker...")

