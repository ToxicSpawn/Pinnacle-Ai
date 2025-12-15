"""
Advanced Alerting System
Multi-channel alerting with intelligent routing
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, List
from datetime import datetime

from notifications.telegram_alerts import TelegramAlerts

logger = logging.getLogger(__name__)


class AdvancedAlertSystem:
    """
    Advanced alerting system with multiple channels and intelligent routing.
    
    Features:
    - Multi-channel alerts (Telegram, Email, PagerDuty)
    - Alert cooldown and deduplication
    - Severity-based routing
    - Performance monitoring alerts
    - Risk alerts
    - Strategy alerts
    - Market alerts
    """
    
    def __init__(self, config: Dict):
        """
        Initialize advanced alert system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.telegram = TelegramAlerts(
            token=config.get('telegram', {}).get('token'),
            chat_id=config.get('telegram', {}).get('chat_id')
        )
        
        # Email alerts (placeholder - would need email library)
        self.email_enabled = bool(config.get('email', {}).get('smtp_server'))
        
        # PagerDuty alerts (placeholder - would need PagerDuty library)
        self.pagerduty_enabled = bool(config.get('pagerduty', {}).get('service_key'))
        
        self.last_alerts: Dict[tuple, float] = {}
        self.alert_cooldown = config.get('cooldown', 300)  # 5 minutes
    
    def check_alerts(self, database) -> None:
        """
        Check for alert conditions.
        
        Args:
            database: Trading database instance
        """
        # Performance alerts
        self._check_performance_alerts(database)
        
        # Risk alerts
        self._check_risk_alerts(database)
        
        # Strategy alerts
        self._check_strategy_alerts(database)
        
        # Market alerts
        self._check_market_alerts(database)
    
    def _check_performance_alerts(self, database) -> None:
        """Check performance-related alerts."""
        alerts_config = self.config.get('alerts', {})
        
        # Daily PnL alert
        daily_pnl = database.get_daily_pnl() if hasattr(database, 'get_daily_pnl') else 0
        min_daily_pnl = alerts_config.get('min_daily_pnl', -100)
        
        if daily_pnl < min_daily_pnl:
            self._send_alert(
                f"Daily PnL Alert: ${daily_pnl:,.2f} (below threshold of ${min_daily_pnl:,.2f})",
                "warning"
            )
        
        # Win rate alert
        win_rate = database.get_win_rate(period='day') if hasattr(database, 'get_win_rate') else 0
        min_win_rate = alerts_config.get('min_win_rate', 50)
        
        if win_rate < min_win_rate:
            self._send_alert(
                f"Win Rate Alert: {win_rate:.1f}% (below threshold of {min_win_rate:.1f}%)",
                "warning"
            )
    
    def _check_risk_alerts(self, database) -> None:
        """Check risk-related alerts."""
        alerts_config = self.config.get('alerts', {})
        
        # Drawdown alert
        drawdown = database.get_current_drawdown() if hasattr(database, 'get_current_drawdown') else 0
        max_drawdown = alerts_config.get('max_drawdown', 0.2)
        
        if drawdown > max_drawdown:
            self._send_alert(
                f"Drawdown Alert: {drawdown:.2%} (exceeds threshold of {max_drawdown:.2%})",
                "critical"
            )
        
        # Leverage alert
        leverage = database.get_current_leverage() if hasattr(database, 'get_current_leverage') else 0
        max_leverage = alerts_config.get('max_leverage', 5)
        
        if leverage > max_leverage:
            self._send_alert(
                f"Leverage Alert: {leverage:.1f}x (exceeds threshold of {max_leverage:.1f}x)",
                "warning"
            )
    
    def _check_strategy_alerts(self, database) -> None:
        """Check strategy performance alerts."""
        alerts_config = self.config.get('alerts', {})
        
        strat_perf = database.get_strategy_performance() if hasattr(database, 'get_strategy_performance') else []
        
        if not isinstance(strat_perf, list):
            return
        
        for strategy in strat_perf:
            return_pct = strategy.get('return_pct', 0)
            min_return = alerts_config.get('min_strategy_return', 0.05)
            
            if return_pct < min_return:
                self._send_alert(
                    f"Strategy Performance Alert: {strategy.get('name', 'Unknown')} has "
                    f"{return_pct:.2f}% return (below threshold)",
                    "warning"
                )
            
            profit_factor = strategy.get('profit_factor', 0)
            min_profit_factor = alerts_config.get('min_profit_factor', 1.5)
            
            if profit_factor < min_profit_factor:
                self._send_alert(
                    f"Strategy Profit Factor Alert: {strategy.get('name', 'Unknown')} has "
                    f"profit factor of {profit_factor:.2f} (below threshold)",
                    "warning"
                )
    
    def _check_market_alerts(self, database) -> None:
        """Check market condition alerts."""
        alerts_config = self.config.get('alerts', {})
        
        # Volatility alert
        volatility = database.get_market_volatility() if hasattr(database, 'get_market_volatility') else 0
        max_volatility = alerts_config.get('max_volatility', 0.05)
        
        if volatility > max_volatility:
            self._send_alert(
                f"Market Volatility Alert: {volatility:.2%} (exceeds threshold of {max_volatility:.2%})",
                "warning"
            )
        
        # Liquidity alert
        liquidity = database.get_market_liquidity() if hasattr(database, 'get_market_liquidity') else 0
        min_liquidity = alerts_config.get('min_liquidity', 1000000)
        
        if liquidity < min_liquidity:
            self._send_alert(
                f"Market Liquidity Alert: {liquidity:,.0f} (below threshold of {min_liquidity:,.0f})",
                "warning"
            )
        
        # Sentiment alert
        news_sentiment = database.get_news_sentiment() if hasattr(database, 'get_news_sentiment') else 0
        min_sentiment = alerts_config.get('min_sentiment', -0.5)
        
        if news_sentiment < min_sentiment:
            self._send_alert(
                f"News Sentiment Alert: {news_sentiment:.2f} (below threshold of {min_sentiment:.2f})",
                "warning"
            )
    
    def _send_alert(self, message: str, severity: str = "info") -> None:
        """
        Send alert through appropriate channels.
        
        Args:
            message: Alert message
            severity: Alert severity (info, warning, critical)
        """
        alert_key = (message, severity)
        current_time = time.time()
        
        # Check cooldown
        if alert_key in self.last_alerts:
            last_time = self.last_alerts[alert_key]
            if current_time - last_time < self.alert_cooldown:
                return  # Still in cooldown
        
        # Send based on severity
        if severity == "info":
            self.telegram.send_message(message)
            if self.email_enabled:
                self._send_email(message, severity)
        
        elif severity == "warning":
            self.telegram.send_message(f"⚠️ {message}")
            if self.email_enabled:
                self._send_email(f"WARNING: {message}", severity)
        
        elif severity == "critical":
            self.telegram.send_message(f"🚨 {message}")
            if self.email_enabled:
                self._send_email(f"CRITICAL: {message}", severity)
            if self.pagerduty_enabled:
                self._trigger_pagerduty(message)
        
        # Update last alert time
        self.last_alerts[alert_key] = current_time
        
        # Clean up old alerts
        self.last_alerts = {
            k: v for k, v in self.last_alerts.items()
            if current_time - v < self.alert_cooldown * 2
        }
    
    def _send_email(self, message: str, severity: str):
        """Send email alert (placeholder)."""
        # In production, would use smtplib or email library
        logger.info(f"Email alert ({severity}): {message}")
    
    def _trigger_pagerduty(self, message: str):
        """Trigger PagerDuty alert (placeholder)."""
        # In production, would use PagerDuty API
        logger.info(f"PagerDuty alert: {message}")

