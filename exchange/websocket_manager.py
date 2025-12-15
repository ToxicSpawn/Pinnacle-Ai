"""
WebSocket Manager for Real-Time Market Data
Supports multiple exchanges via WebSocket connections
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Callable, Any, List
import json

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not installed. WebSocket features will be disabled.")


class WebSocketManager:
    """
    WebSocket manager for real-time market data streaming.
    
    Supports:
    - Ticker updates
    - Order book updates
    - Trade updates
    - OHLCV updates
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocketClientProtocol] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # exchange -> list of symbols
        self.callbacks: Dict[str, Callable] = {}  # subscription_id -> callback
        self.running = False
        self._tasks: List[asyncio.Task] = []
    
    async def connect_kraken(self, symbols: List[str], callback: Callable) -> bool:
        """
        Connect to Kraken WebSocket.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTC/USD', 'ETH/USD'])
            callback: Callback function for messages
            
        Returns:
            True if connection successful
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available")
            return False
        
        try:
            # Kraken WebSocket URL
            url = "wss://ws.kraken.com"
            
            async def handle_kraken():
                async with websockets.connect(url) as ws:
                    self.connections['kraken'] = ws
                    
                    # Subscribe to ticker for each symbol
                    for symbol in symbols:
                        # Convert CCXT symbol format to Kraken format
                        kraken_pair = symbol.replace('/', '').replace('BTC', 'XBT')
                        subscribe_msg = {
                            "event": "subscribe",
                            "pair": [kraken_pair],
                            "subscription": {"name": "ticker"}
                        }
                        await ws.send(json.dumps(subscribe_msg))
                    
                    logger.info(f"✅ Connected to Kraken WebSocket for {len(symbols)} symbols")
                    
                    # Listen for messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if isinstance(data, list) and len(data) > 0:
                                # Ticker update
                                callback('kraken', data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from Kraken: {message}")
                        except Exception as e:
                            logger.error(f"Error processing Kraken message: {e}")
            
            task = asyncio.create_task(handle_kraken())
            self._tasks.append(task)
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Kraken WebSocket: {e}")
            return False
    
    async def connect_binance(self, symbols: List[str], callback: Callable) -> bool:
        """
        Connect to Binance WebSocket.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
            callback: Callback function for messages
            
        Returns:
            True if connection successful
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available")
            return False
        
        try:
            # Binance WebSocket URL (stream endpoint)
            base_url = "wss://stream.binance.com:9443/ws/"
            
            async def handle_binance():
                # Create individual streams for each symbol
                streams = []
                for symbol in symbols:
                    # Convert CCXT format to Binance format
                    binance_symbol = symbol.replace('/', '').lower()
                    streams.append(f"{binance_symbol}@ticker")
                
                # Use combined stream
                stream_names = "/".join(streams)
                url = f"wss://stream.binance.com:9443/stream?streams={stream_names}"
                
                async with websockets.connect(url) as ws:
                    self.connections['binance'] = ws
                    logger.info(f"✅ Connected to Binance WebSocket for {len(symbols)} symbols")
                    
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if 'data' in data:
                                callback('binance', data['data'])
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from Binance: {message}")
                        except Exception as e:
                            logger.error(f"Error processing Binance message: {e}")
            
            task = asyncio.create_task(handle_binance())
            self._tasks.append(task)
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Binance WebSocket: {e}")
            return False
    
    async def connect_coinbase(self, symbols: List[str], callback: Callable) -> bool:
        """
        Connect to Coinbase WebSocket.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTC-USD', 'ETH-USD'])
            callback: Callback function for messages
            
        Returns:
            True if connection successful
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available")
            return False
        
        try:
            url = "wss://ws-feed.pro.coinbase.com"
            
            async def handle_coinbase():
                async with websockets.connect(url) as ws:
                    self.connections['coinbase'] = ws
                    
                    # Subscribe to ticker for each symbol
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": [s.replace('/', '-') for s in symbols],
                        "channels": ["ticker"]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    logger.info(f"✅ Connected to Coinbase WebSocket for {len(symbols)} symbols")
                    
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if data.get('type') == 'ticker':
                                callback('coinbase', data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from Coinbase: {message}")
                        except Exception as e:
                            logger.error(f"Error processing Coinbase message: {e}")
            
            task = asyncio.create_task(handle_coinbase())
            self._tasks.append(task)
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase WebSocket: {e}")
            return False
    
    async def subscribe(
        self,
        exchange_name: str,
        symbols: List[str],
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> bool:
        """
        Subscribe to real-time updates for symbols on an exchange.
        
        Args:
            exchange_name: Name of the exchange ('kraken', 'binance', 'coinbase')
            symbols: List of trading pairs
            callback: Callback function (exchange_name, data) -> None
            
        Returns:
            True if subscription successful
        """
        exchange_name = exchange_name.lower()
        
        if exchange_name == 'kraken':
            return await self.connect_kraken(symbols, callback)
        elif exchange_name == 'binance':
            return await self.connect_binance(symbols, callback)
        elif exchange_name == 'coinbase':
            return await self.connect_coinbase(symbols, callback)
        else:
            logger.error(f"Unsupported exchange for WebSocket: {exchange_name}")
            return False
    
    async def disconnect(self, exchange_name: str) -> None:
        """Disconnect from an exchange's WebSocket."""
        exchange_name = exchange_name.lower()
        if exchange_name in self.connections:
            await self.connections[exchange_name].close()
            del self.connections[exchange_name]
            logger.info(f"Disconnected from {exchange_name} WebSocket")
    
    async def disconnect_all(self) -> None:
        """Disconnect from all WebSocket connections."""
        for exchange_name in list(self.connections.keys()):
            await self.disconnect(exchange_name)
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        self._tasks.clear()
        logger.info("All WebSocket connections closed")

