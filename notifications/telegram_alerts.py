"""
Telegram Alerts System
Real-time notifications for trades, errors, and system events
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available. Telegram alerts will be disabled.")


class TelegramAlerts:
    """
    Telegram alert system for trading bot notifications.
    
    Features:
    - Trade notifications
    - Error alerts
    - System status updates
    - Performance summaries
    - Custom messages
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram alerts.
        
        Args:
            token: Telegram bot token (or from TELEGRAM_TOKEN env var)
            chat_id: Chat ID for notifications (or from TELEGRAM_CHAT_ID env var)
            enabled: Enable/disable alerts
        """
        self.enabled = enabled and TELEGRAM_AVAILABLE
        
        if not self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("Telegram library not available")
            return
        
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not set. Alerts disabled.")
            self.enabled = False
            return
        
        try:
            self.bot = Bot(token=self.token)
            logger.info("✅ Telegram alerts initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enabled = False
    
    async def send_message(
        self,
        message: str,
        parse_mode: Optional[str] = None
    ) -> bool:
        """
        Send a text message.
        
        Args:
            message: Message text
            parse_mode: Parse mode ('HTML' or 'Markdown')
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    async def send_trade_alert(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        exchange: str,
        order_id: Optional[str] = None
    ) -> bool:
        """
        Send trade execution alert.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Trade amount
            price: Execution price
            exchange: Exchange name
            order_id: Order ID
            
        Returns:
            True if sent successfully
        """
        emoji = "🟢" if side.lower() == 'buy' else "🔴"
        message = (
            f"{emoji} <b>Trade Executed</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
            f"Amount: {amount:.6f}\n"
            f"Price: ${price:.2f}\n"
            f"Exchange: {exchange}\n"
        )
        
        if order_id:
            message += f"Order ID: {order_id}\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message, parse_mode='HTML')
    
    async def send_error_alert(
        self,
        error: str,
        context: Optional[str] = None
    ) -> bool:
        """
        Send error alert.
        
        Args:
            error: Error message
            context: Additional context
            
        Returns:
            True if sent successfully
        """
        message = (
            f"❌ <b>Error Alert</b>\n\n"
            f"Error: {error}\n"
        )
        
        if context:
            message += f"Context: {context}\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message, parse_mode='HTML')
    
    async def send_arbitrage_alert(
        self,
        opportunity: Dict[str, Any]
    ) -> bool:
        """
        Send arbitrage opportunity alert.
        
        Args:
            opportunity: Arbitrage opportunity details
            
        Returns:
            True if sent successfully
        """
        message = (
            f"💰 <b>Arbitrage Opportunity</b>\n\n"
            f"Symbol: {opportunity.get('symbol', 'N/A')}\n"
            f"Buy Exchange: {opportunity.get('buy_exchange', 'N/A')}\n"
            f"Sell Exchange: {opportunity.get('sell_exchange', 'N/A')}\n"
            f"Spread: {opportunity.get('spread_pct', 0):.2f}%\n"
            f"Profit: ${opportunity.get('profit_after_fees', 0):.2f}\n"
        )
        
        return await self.send_message(message, parse_mode='HTML')
    
    async def send_performance_summary(
        self,
        stats: Dict[str, Any]
    ) -> bool:
        """
        Send performance summary.
        
        Args:
            stats: Performance statistics dictionary
            
        Returns:
            True if sent successfully
        """
        message = (
            f"📊 <b>Performance Summary</b>\n\n"
            f"Total PnL: ${stats.get('total_pnl', 0):.2f}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Max Drawdown: {stats.get('max_drawdown_pct', 0):.1f}%\n"
        )
        
        return await self.send_message(message, parse_mode='HTML')
    
    async def send_system_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send system status update.
        
        Args:
            status: Status message
            details: Additional details
            
        Returns:
            True if sent successfully
        """
        emoji = "✅" if "started" in status.lower() or "running" in status.lower() else "⚠️"
        
        message = (
            f"{emoji} <b>System Status</b>\n\n"
            f"Status: {status}\n"
        )
        
        if details:
            for key, value in details.items():
                message += f"{key}: {value}\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message, parse_mode='HTML')
    
    def test_connection(self) -> bool:
        """
        Test Telegram connection.
        
        Returns:
            True if connection successful
        """
        if not self.enabled:
            return False
        
        import asyncio
        try:
            asyncio.run(self.send_message("🧪 Test message from trading bot"))
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False

