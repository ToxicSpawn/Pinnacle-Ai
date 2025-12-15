"""
Unified Exchange Manager using CCXT
Supports multiple exchanges with a single unified API
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional, Any, List
import ccxt
import yaml

from exchange.encryption import APIKeyEncryption

logger = logging.getLogger(__name__)


class UnifiedExchangeManager:
    """
    Unified exchange manager supporting multiple exchanges via CCXT.
    
    Features:
    - Dynamic exchange loading from config
    - Unified API for all exchanges
    - Rate limiting and error handling
    - Support for spot and futures markets
    """
    
    def __init__(self, config_path: str = "config/exchanges.yaml"):
        """
        Initialize the unified exchange manager.
        
        Args:
            config_path: Path to exchange configuration file
        """
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.config_path = config_path
        self.encryption = APIKeyEncryption()
        self._load_exchanges()
    
    def _load_exchanges(self) -> None:
        """Load exchanges from configuration file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Exchange config not found at {self.config_path}, using environment variables")
            self._load_from_env()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            exchanges_config = config.get('exchanges', {})
            
            for exchange_name, exchange_config in exchanges_config.items():
                if not exchange_config.get('enabled', True):
                    logger.info(f"Skipping disabled exchange: {exchange_name}")
                    continue
                
                try:
                    exchange = self._create_exchange(exchange_name, exchange_config)
                    if exchange:
                        self.exchanges[exchange_name] = exchange
                        logger.info(f"✅ Loaded exchange: {exchange_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load exchange {exchange_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading exchange config: {e}")
            self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Fallback: Load exchanges from environment variables."""
        # Kraken
        if os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_API_SECRET"):
            try:
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': os.getenv("KRAKEN_API_KEY"),
                    'secret': os.getenv("KRAKEN_API_SECRET"),
                    'enableRateLimit': True,
                })
                logger.info("✅ Loaded Kraken from environment")
            except Exception as e:
                logger.error(f"Failed to load Kraken: {e}")
        
        # Binance
        if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
            try:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': os.getenv("BINANCE_API_KEY"),
                    'secret': os.getenv("BINANCE_API_SECRET"),
                    'enableRateLimit': True,
                })
                logger.info("✅ Loaded Binance from environment")
            except Exception as e:
                logger.error(f"Failed to load Binance: {e}")
        
        # Coinbase
        if os.getenv("COINBASE_API_KEY") and os.getenv("COINBASE_API_SECRET"):
            try:
                self.exchanges['coinbase'] = ccxt.coinbase({
                    'apiKey': os.getenv("COINBASE_API_KEY"),
                    'secret': os.getenv("COINBASE_API_SECRET"),
                    'password': os.getenv("COINBASE_API_PASSPHRASE", ""),
                    'enableRateLimit': True,
                })
                logger.info("✅ Loaded Coinbase from environment")
            except Exception as e:
                logger.error(f"Failed to load Coinbase: {e}")
    
    def _resolve_config_value(self, value: Any, exchange_name: str = "") -> Optional[str]:
        """
        Resolve configuration value, handling environment variables and encryption.
        
        Args:
            value: Configuration value (may be env var reference or encrypted)
            exchange_name: Exchange name for context
            
        Returns:
            Resolved value or None
        """
        if value is None:
            return None
        
        value_str = str(value)
        
        # Check if it's an environment variable reference (${VAR_NAME})
        env_match = re.match(r'\$\{([^}]+)\}', value_str)
        if env_match:
            env_var = env_match.group(1)
            resolved = os.getenv(env_var)
            if resolved:
                return resolved
            else:
                logger.warning(f"Environment variable {env_var} not set for {exchange_name}")
                return None
        
        # If not an env var reference, return as-is
        return value_str
    
    def _create_exchange(self, exchange_name: str, config: Dict[str, Any]) -> Optional[ccxt.Exchange]:
        """
        Create an exchange instance from config.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'kraken', 'binance')
            config: Exchange configuration dictionary
            
        Returns:
            CCXT exchange instance or None if creation fails
        """
        try:
            # Get exchange class from CCXT
            exchange_class = getattr(ccxt, exchange_name, None)
            if not exchange_class:
                logger.error(f"Exchange {exchange_name} not supported by CCXT")
                return None
            
            # Resolve API key and secret (handle env vars and encryption)
            api_key = self._resolve_config_value(config.get('api_key'), exchange_name)
            secret = self._resolve_config_value(config.get('secret'), exchange_name)
            
            # Decrypt if encrypted
            if config.get('encrypted', False):
                if api_key:
                    api_key = self.encryption.decrypt(api_key)
                if secret:
                    secret = self.encryption.decrypt(secret)
            
            # Fallback to environment variables if not in config
            if not api_key:
                api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
            if not secret:
                secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")
            
            if not api_key or not secret:
                logger.warning(f"API credentials not found for {exchange_name}, skipping")
                return None
            
            # Prepare exchange options
            options = {
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': config.get('enable_rate_limit', True),
                'options': config.get('options', {}),
            }
            
            # Add password for exchanges that need it (e.g., Coinbase)
            if exchange_name == 'coinbase':
                passphrase = self._resolve_config_value(config.get('passphrase'), exchange_name)
                if config.get('encrypted', False) and passphrase:
                    passphrase = self.encryption.decrypt(passphrase)
                options['password'] = passphrase or os.getenv("COINBASE_API_PASSPHRASE", "")
            
            # Create exchange instance
            exchange = exchange_class(options)
            
            # Enable sandbox mode if configured
            if config.get('sandbox', False):
                exchange.set_sandbox_mode(True)
                logger.info(f"Sandbox mode enabled for {exchange_name}")
            
            # Load markets (optional, can be done lazily)
            if config.get('load_markets_on_init', False):
                exchange.load_markets()
            
            return exchange
        
        except Exception as e:
            logger.error(f"Error creating exchange {exchange_name}: {e}")
            return None
    
    def get_exchange(self, exchange_name: str) -> Optional[ccxt.Exchange]:
        """
        Get an exchange instance by name.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            Exchange instance or None if not found
        """
        return self.exchanges.get(exchange_name.lower())
    
    def list_exchanges(self) -> List[str]:
        """List all loaded exchange names."""
        return list(self.exchanges.keys())
    
    async def fetch_ticker(self, exchange_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker data from an exchange.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            
        Returns:
            Ticker data dictionary or None on error
        """
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            logger.error(f"Exchange {exchange_name} not found")
            return None
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.NetworkError as e:
            logger.error(f"Network error on {exchange_name}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error on {exchange_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching ticker from {exchange_name}: {e}")
            return None
    
    async def fetch_ohlcv(self, exchange_name: str, symbol: str, timeframe: str = '1h', limit: int = 200) -> List[List]:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV candles
        """
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            logger.error(f"Exchange {exchange_name} not found")
            return []
        
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV from {exchange_name}: {e}")
            return []
    
    async def create_order(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create an order on an exchange.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order amount
            order_type: 'market', 'limit', 'stop', 'stop_limit', etc.
            price: Price for limit orders
            params: Additional exchange-specific parameters
            
        Returns:
            Order response dictionary or None on error
        """
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            logger.error(f"Exchange {exchange_name} not found")
            return None
        
        try:
            order_params = params or {}
            
            if order_type == 'market':
                order = exchange.create_market_order(symbol, side, amount, None, order_params)
            elif order_type == 'limit':
                if price is None:
                    logger.error("Price required for limit orders")
                    return None
                order = exchange.create_limit_order(symbol, side, amount, price, order_params)
            else:
                # Use generic create_order for advanced order types
                order = exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    params=order_params
                )
            
            logger.info(f"✅ Order placed on {exchange_name}: {side} {amount} {symbol} ({order_type})")
            return order
        
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds on {exchange_name}: {e}")
            return None
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order on {exchange_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing order on {exchange_name}: {e}")
            return None
    
    async def fetch_balance(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch account balance from an exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            Balance dictionary or None on error
        """
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None
        
        try:
            balance = exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance from {exchange_name}: {e}")
            return None
    
    def check_exchange_support(self, exchange_name: str, feature: str) -> bool:
        """
        Check if an exchange supports a specific feature.
        
        Args:
            exchange_name: Name of the exchange
            feature: Feature to check (e.g., 'stopLoss', 'trailingStop', 'oco')
            
        Returns:
            True if feature is supported, False otherwise
        """
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return False
        
        # Check if exchange has the required method or capability
        feature_map = {
            'stopLoss': hasattr(exchange, 'create_stop_loss_order'),
            'trailingStop': hasattr(exchange, 'create_trailing_stop_order'),
            'oco': hasattr(exchange, 'create_oco_order'),
        }
        
        return feature_map.get(feature, False)

