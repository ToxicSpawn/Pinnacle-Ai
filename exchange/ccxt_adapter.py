"""
CCXT Adapter for Backward Compatibility
Provides compatibility layer between new unified system and existing code
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from exchange.unified_exchange_manager import UnifiedExchangeManager
from exchange.websocket_manager import WebSocketManager
from exchange.advanced_orders import AdvancedOrderManager
from exchange.encryption import APIKeyEncryption

logger = logging.getLogger(__name__)


class CCXTAdapter:
    """
    Adapter class that provides backward-compatible interface
    while using the new unified exchange manager.
    """
    
    def __init__(self, config_path: str = "config/exchanges.yaml"):
        """
        Initialize the CCXT adapter.
        
        Args:
            config_path: Path to exchange configuration file
        """
        self.encryption = APIKeyEncryption()
        self.exchange_manager = UnifiedExchangeManager(config_path)
        self.websocket_manager = WebSocketManager()
        self.advanced_orders = AdvancedOrderManager(self.exchange_manager)
        
        # Backward compatibility: expose individual clients
        self.kraken = self._create_client_wrapper('kraken')
        self.binance = self._create_client_wrapper('binance')
        self.coinbase = self._create_client_wrapper('coinbase')
    
    def _create_client_wrapper(self, exchange_name: str):
        """Create a wrapper that mimics the old client interface."""
        return ExchangeClientWrapper(self.exchange_manager, exchange_name)
    
    def get_exchange(self, exchange_name: str):
        """Get an exchange instance."""
        return self.exchange_manager.get_exchange(exchange_name)
    
    async def fetch_ticker(self, exchange_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker data."""
        return await self.exchange_manager.fetch_ticker(exchange_name, symbol)
    
    async def fetch_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 200
    ) -> List[List]:
        """Fetch OHLCV data."""
        return await self.exchange_manager.fetch_ohlcv(exchange_name, symbol, timeframe, limit)
    
    async def create_order(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Create an order."""
        return await self.exchange_manager.create_order(
            exchange_name, symbol, side, amount, order_type, price, kwargs.get('params')
        )
    
    async def subscribe_websocket(
        self,
        exchange_name: str,
        symbols: List[str],
        callback
    ) -> bool:
        """Subscribe to WebSocket updates."""
        return await self.websocket_manager.subscribe(exchange_name, symbols, callback)
    
    async def disconnect_websocket(self, exchange_name: str) -> None:
        """Disconnect from WebSocket."""
        await self.websocket_manager.disconnect(exchange_name)
    
    async def disconnect_all_websockets(self) -> None:
        """Disconnect from all WebSocket connections."""
        await self.websocket_manager.disconnect_all()


class ExchangeClientWrapper:
    """
    Wrapper class that provides backward-compatible interface
    for individual exchange clients.
    """
    
    def __init__(self, exchange_manager: UnifiedExchangeManager, exchange_name: str):
        self.exchange_manager = exchange_manager
        self.exchange_name = exchange_name
        self._client = exchange_manager.get_exchange(exchange_name)
    
    @property
    def name(self) -> str:
        """Get exchange name."""
        return self.exchange_name
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker (backward compatible)."""
        return await self.exchange_manager.fetch_ticker(self.exchange_name, symbol)
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> List[List]:
        """Fetch OHLCV (backward compatible)."""
        return await self.exchange_manager.fetch_ohlcv(self.exchange_name, symbol, timeframe, limit)
    
    async def fetch_order_book(self, symbol: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Fetch order book."""
        if not self._client:
            return None
        try:
            return self._client.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    async def create_order(self, symbol: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
        """Create market order (backward compatible)."""
        return await self.exchange_manager.create_order(
            self.exchange_name, symbol, side, amount, 'market'
        )
    
    def create_order_sync(self, symbol: str, type: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
        """Synchronous order creation (for compatibility with some code)."""
        if not self._client:
            return None
        try:
            return self._client.create_order(symbol, type, side, amount)
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

