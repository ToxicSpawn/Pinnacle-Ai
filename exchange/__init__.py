"""
Exchange module with unified CCXT support, WebSocket streaming, and advanced orders.
"""
from exchange.unified_exchange_manager import UnifiedExchangeManager
from exchange.websocket_manager import WebSocketManager
from exchange.advanced_orders import AdvancedOrderManager
from exchange.encryption import APIKeyEncryption
from exchange.ccxt_adapter import CCXTAdapter

__all__ = [
    'UnifiedExchangeManager',
    'WebSocketManager',
    'AdvancedOrderManager',
    'APIKeyEncryption',
    'CCXTAdapter',
]

