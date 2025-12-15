"""
Main Bot Orchestrator
Unified entry point that integrates all Phase 1+ features
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# Phase 1 imports
from exchange.ccxt_adapter import CCXTAdapter
from exchange.websocket_manager import WebSocketManager
from exchange.advanced_orders import AdvancedOrderManager

# Strategy imports
from strategies.lstm_strategy import LSTMPredictor
from strategies.enhanced_arbitrage import EnhancedArbitrageStrategy

# Risk management
from risk.kelly_criterion import KellyPositionSizer, TradeStats
from risk.drawdown_control import DrawdownControl

# Notifications
from notifications.telegram_alerts import TelegramAlerts

# Tax reporting
from utils.tax_reporting import TaxReporter

# Existing bot components
from core.state import GlobalState
from core.runtime import MultiAgentRuntime
from core.policy import PolicyEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KrackenBotOrchestrator:
    """
    Main bot orchestrator integrating all features.
    
    Features:
    - Multi-exchange trading
    - WebSocket streaming
    - LSTM price prediction
    - Arbitrage strategies
    - Risk management (Kelly Criterion, drawdown control)
    - Telegram alerts
    - Tax reporting
    - Backtesting support
    """
    
    def __init__(self, config_path: str = "config/exchanges.yaml"):
        """Initialize bot orchestrator."""
        self.config_path = config_path
        self.running = False
        self._tasks: List[asyncio.Task] = []
        
        # Initialize components
        logger.info("Initializing Kracken Bot Orchestrator...")
        
        # Phase 1: Exchange infrastructure
        self.adapter = CCXTAdapter(config_path)
        self.websocket_manager = WebSocketManager()
        self.advanced_orders = AdvancedOrderManager(self.adapter.exchange_manager)
        
        # Strategies
        self.lstm_predictors: Dict[str, LSTMPredictor] = {}
        self.arbitrage_strategy = EnhancedArbitrageStrategy(
            self.adapter,
            min_profit_pct=0.5,
            max_position_size=1000.0
        )
        
        # Risk management
        self.kelly_sizer = KellyPositionSizer(fractional=True, max_position_pct=0.25)
        self.drawdown_control = DrawdownControl(max_drawdown=0.20)
        
        # Notifications
        self.telegram = TelegramAlerts()
        
        # Tax reporting
        self.tax_reporter = TaxReporter()
        
        # Existing bot state
        self.state = GlobalState()
        self.policy = PolicyEngine()
        self.runtime: Optional[MultiAgentRuntime] = None
        
        # Trade history for tax reporting
        self.trade_history: List[Dict] = []
        
        logger.info("✅ Bot orchestrator initialized")
    
    async def initialize_lstm_models(self, symbols: List[str]) -> None:
        """Initialize LSTM models for symbols."""
        logger.info("Initializing LSTM models...")
        
        for symbol in symbols:
            try:
                predictor = LSTMPredictor(symbol=symbol)
                self.lstm_predictors[symbol] = predictor
                logger.info(f"✅ LSTM model initialized for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to initialize LSTM for {symbol}: {e}")
    
    async def start_websocket_streams(self, symbols: List[str]) -> None:
        """Start WebSocket streams for real-time data."""
        logger.info("Starting WebSocket streams...")
        
        exchanges = self.adapter.exchange_manager.list_exchanges()
        
        async def handle_ticker(exchange_name: str, data: dict):
            """Handle ticker updates."""
            # Update state with latest prices
            # This is a simplified handler - integrate with your state management
            logger.debug(f"Ticker update from {exchange_name}: {data}")
        
        for exchange_name in exchanges:
            try:
                await self.websocket_manager.subscribe(
                    exchange_name,
                    symbols,
                    handle_ticker
                )
                logger.info(f"✅ WebSocket stream started for {exchange_name}")
            except Exception as e:
                logger.warning(f"Failed to start WebSocket for {exchange_name}: {e}")
    
    async def run_arbitrage_loop(self) -> None:
        """Run arbitrage detection loop."""
        logger.info("Starting arbitrage detection loop...")
        
        while self.running:
            try:
                # Check cross-exchange arbitrage
                exchanges = ['kraken', 'binance', 'coinbase']
                symbols = ['BTC/USD', 'BTC/USDT', 'ETH/USD', 'ETH/USDT']
                
                for symbol in symbols:
                    opportunities = await self.arbitrage_strategy.find_cross_exchange_opportunities(
                        symbol, exchanges
                    )
                    
                    for opp in opportunities:
                        logger.info(
                            f"💰 Arbitrage opportunity: {opp.symbol} "
                            f"{opp.spread_pct:.2f}% spread between "
                            f"{opp.buy_exchange} and {opp.sell_exchange}"
                        )
                        
                        # Send alert
                        await self.telegram.send_arbitrage_alert({
                            'symbol': opp.symbol,
                            'buy_exchange': opp.buy_exchange,
                            'sell_exchange': opp.sell_exchange,
                            'spread_pct': opp.spread_pct,
                            'profit_after_fees': opp.profit_after_fees
                        })
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in arbitrage loop: {e}")
                await asyncio.sleep(60)
    
    async def run_lstm_prediction_loop(self) -> None:
        """Run LSTM prediction loop."""
        logger.info("Starting LSTM prediction loop...")
        
        while self.running:
            try:
                for symbol, predictor in self.lstm_predictors.items():
                    # Fetch historical data
                    ohlcv = await self.adapter.exchange_manager.fetch_ohlcv(
                        'binance', symbol, '1h', limit=200
                    )
                    
                    if len(ohlcv) < 60:
                        continue
                    
                    # Convert to DataFrame
                    import pandas as pd
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Predict
                    try:
                        predicted_price = predictor.predict(df)
                        current_price = df['close'].iloc[-1]
                        
                        signal, confidence = predictor.get_prediction_signal(
                            current_price, predicted_price
                        )
                        
                        if signal != 'HOLD':
                            logger.info(
                                f"📊 LSTM {symbol}: {signal} signal "
                                f"(confidence: {confidence:.2f}, "
                                f"current: ${current_price:.2f}, "
                                f"predicted: ${predicted_price:.2f})"
                            )
                    except Exception as e:
                        logger.warning(f"LSTM prediction failed for {symbol}: {e}")
                
                await asyncio.sleep(300)  # Update every 5 minutes
            
            except Exception as e:
                logger.error(f"Error in LSTM loop: {e}")
                await asyncio.sleep(300)
    
    async def monitor_risk(self) -> None:
        """Monitor risk metrics and enforce limits."""
        while self.running:
            try:
                # Update drawdown control
                # Get current balance from exchanges
                balance = 10000.0  # Placeholder - get from actual account
                
                can_trade = self.drawdown_control.update(balance)
                
                if not can_trade:
                    logger.warning("⚠️ Trading halted due to drawdown limit")
                    await self.telegram.send_system_status(
                        "Trading Halted",
                        {'reason': self.drawdown_control.halt_reason}
                    )
                
                # Get Kelly position size
                kelly_stats = self.kelly_sizer.get_stats()
                if kelly_stats['total_trades'] > 10:
                    logger.info(f"Kelly stats: {kelly_stats}")
                
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)
    
    async def start(self) -> None:
        """Start the bot."""
        if self.running:
            logger.warning("Bot already running")
            return
        
        self.running = True
        logger.info("🚀 Starting Kracken Trading Bot...")
        
        # Send startup notification
        await self.telegram.send_system_status("Bot Started", {
            'mode': os.getenv('BOT_MODE', 'paper'),
            'exchanges': ', '.join(self.adapter.exchange_manager.list_exchanges())
        })
        
        # Start existing runtime if available
        if self.runtime is None:
            self.runtime = MultiAgentRuntime(self.state, self.policy)
            await self.runtime.start()
        
        # Start background tasks
        symbols = ['BTC/USD', 'ETH/USD', 'BTC/USDT', 'ETH/USDT']
        
        # Initialize LSTM models
        await self.initialize_lstm_models(symbols)
        
        # Start WebSocket streams
        await self.start_websocket_streams(symbols)
        
        # Start background loops
        self._tasks.append(asyncio.create_task(self.run_arbitrage_loop()))
        self._tasks.append(asyncio.create_task(self.run_lstm_prediction_loop()))
        self._tasks.append(asyncio.create_task(self.monitor_risk()))
        
        logger.info("✅ Bot started successfully")
    
    async def stop(self) -> None:
        """Stop the bot."""
        if not self.running:
            return
        
        logger.info("Stopping bot...")
        self.running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Disconnect WebSockets
        await self.websocket_manager.disconnect_all()
        
        # Stop runtime
        if self.runtime:
            await self.runtime.stop()
        
        # Export trades for tax reporting
        if self.trade_history:
            self.tax_reporter.export_all_formats(self.trade_history)
        
        # Send shutdown notification
        await self.telegram.send_system_status("Bot Stopped")
        
        logger.info("✅ Bot stopped")
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())


async def main():
    """Main entry point."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize orchestrator
    orchestrator = KrackenBotOrchestrator()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, orchestrator.handle_signal)
    signal.signal(signal.SIGTERM, orchestrator.handle_signal)
    
    try:
        # Start bot
        await orchestrator.start()
        
        # Keep running
        while orchestrator.running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())

