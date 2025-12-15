"""
Multi-Agent Trading System
Autonomous trading agents interacting in a simulated market
"""
from __future__ import annotations

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import mesa
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.datacollection import DataCollector
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False
    logger.warning("Mesa not available. Multi-agent system disabled.")


if MESA_AVAILABLE:
    class TradingAgent(Agent):
        """Autonomous trading agent with specialized behavior."""
        
        def __init__(
            self,
            unique_id: int,
            model: Model,
            agent_type: str,
            initial_capital: float = 1000.0
        ):
            super().__init__(unique_id, model)
            self.type = agent_type
            self.capital = initial_capital
            self.initial_capital = initial_capital
            self.positions: Dict[str, float] = {}
            self.performance = 0.0
            self.memory: List[Dict] = []
            self.strategy = self._initialize_strategy()
        
        def _initialize_strategy(self):
            """Initialize strategy based on agent type."""
            if self.type == "market_maker":
                return MarketMakingStrategy()
            elif self.type == "arbitrageur":
                return ArbitrageStrategy()
            elif self.type == "trend_follower":
                return TrendFollowingStrategy()
            elif self.type == "mean_reversion":
                return MeanReversionStrategy()
            elif self.type == "ai_trader":
                # Would use NeuroEvolutionaryTrader in real implementation
                return RandomTradingStrategy()
            else:
                return RandomTradingStrategy()
        
        def step(self):
            """Execute one trading step."""
            # Get market data
            market_data = self.model.get_market_data()
            
            # Generate trading signals
            signals = self.strategy.generate_signals(market_data)
            
            # Execute trades
            for symbol, signal in signals.items():
                if signal['action'] == 'buy':
                    self._buy(symbol, signal['amount'], signal['price'])
                elif signal['action'] == 'sell':
                    self._sell(symbol, signal['amount'], signal['price'])
            
            # Update performance
            self._update_performance()
            
            # Learn from experience
            self._learn(market_data)
        
        def _buy(self, symbol: str, amount: float, price: float):
            """Execute buy order."""
            cost = amount * price
            if cost > self.capital:
                amount = self.capital / price
                cost = self.capital
            
            if amount > 0:
                self.positions[symbol] = self.positions.get(symbol, 0.0) + amount
                self.capital -= cost
                self.model.record_trade(self, 'buy', symbol, amount, price)
        
        def _sell(self, symbol: str, amount: float, price: float):
            """Execute sell order."""
            if symbol not in self.positions or self.positions[symbol] == 0:
                return
            
            if amount > self.positions[symbol]:
                amount = self.positions[symbol]
            
            if amount > 0:
                self.positions[symbol] -= amount
                self.capital += amount * price
                self.model.record_trade(self, 'sell', symbol, amount, price)
        
        def _update_performance(self):
            """Update agent performance metrics."""
            # Calculate portfolio value
            portfolio_value = self.capital
            for symbol, amount in self.positions.items():
                portfolio_value += amount * self.model.get_price(symbol)
            
            # Update performance
            self.performance = (
                (portfolio_value - self.initial_capital) / self.initial_capital
            )
        
        def _learn(self, market_data: Dict):
            """Learn from recent experience."""
            if hasattr(self.strategy, 'learn'):
                self.strategy.learn(market_data, self.memory)
            
            # Update memory
            self.memory.append(market_data)
            if len(self.memory) > 100:  # Keep last 100 steps
                self.memory.pop(0)
    
    
    class MultiAgentTradingModel(Model):
        """Multi-agent trading system model."""
        
        def __init__(self, num_agents: int = 10, initial_capital: float = 10000.0):
            self.num_agents = num_agents
            self.initial_capital = initial_capital
            self.schedule = RandomActivation(self)
            self.market = MarketEnvironment()
            self.datacollector = DataCollector(
                model_reporters={
                    "Total Value": lambda m: sum(
                        a.capital + sum(
                            a.positions[s] * m.get_price(s)
                            for s in a.positions
                        )
                        for a in m.schedule.agents
                    ),
                    "Market Price": lambda m: m.get_price("BTC/USD")
                },
                agent_reporters={
                    "Performance": "performance",
                    "Capital": "capital",
                    "Positions": lambda a: sum(a.positions.values())
                }
            )
            
            # Create agents
            agent_types = [
                "market_maker", "arbitrageur",
                "trend_follower", "mean_reversion", "ai_trader"
            ]
            
            for i in range(self.num_agents):
                # Randomly assign agent types
                agent_type = np.random.choice(agent_types)
                agent = TradingAgent(
                    i, self, agent_type, initial_capital / num_agents
                )
                self.schedule.add(agent)
        
        def step(self):
            """Advance the model by one step."""
            self.schedule.step()
            self.market.step()
            self.datacollector.collect(self)
        
        def get_market_data(self) -> Dict:
            """Get current market data."""
            return self.market.get_data()
        
        def get_price(self, symbol: str) -> float:
            """Get current price for a symbol."""
            return self.market.get_price(symbol)
        
        def record_trade(
            self,
            agent: TradingAgent,
            action: str,
            symbol: str,
            amount: float,
            price: float
        ):
            """Record a trade in the market."""
            self.market.record_trade(agent, action, symbol, amount, price)
    
    
    class MarketEnvironment:
        """Market environment for multi-agent system."""
        
        def __init__(self):
            self.symbols = ["BTC/USD", "ETH/USD", "BTC/ETH"]
            self.prices = {
                "BTC/USD": 50000.0,
                "ETH/USD": 3000.0,
                "BTC/ETH": 16.67
            }
            self.order_books = {
                symbol: {'bids': [], 'asks': []}
                for symbol in self.symbols
            }
            self.trade_history: List[Dict] = []
            self.volatility = 0.02  # 2% daily volatility
            self.trend = 0.001  # 0.1% daily trend
            self.time = 0
        
        def step(self):
            """Update market state."""
            self.time += 1
            
            # Update prices based on trend and volatility
            for symbol in self.symbols:
                change = np.random.normal(self.trend, self.volatility)
                self.prices[symbol] *= (1 + change)
            
            # Update order books
            self._update_order_books()
        
        def _update_order_books(self):
            """Update order books based on recent trades."""
            for symbol in self.symbols:
                # Clear old orders
                self.order_books[symbol]['bids'] = []
                self.order_books[symbol]['asks'] = []
                
                # Add new orders based on recent trades
                recent_trades = [
                    t for t in self.trade_history
                    if t['symbol'] == symbol and t['timestamp'] > self.time - 10
                ]
                
                if recent_trades:
                    avg_price = np.mean([t['price'] for t in recent_trades])
                    std_price = np.std([t['price'] for t in recent_trades]) or 1.0
                    
                    # Add bids
                    for i in range(5):
                        price = avg_price - (i + 1) * std_price / 2
                        amount = np.random.uniform(0.1, 1.0)
                        self.order_books[symbol]['bids'].append((price, amount))
                    
                    # Add asks
                    for i in range(5):
                        price = avg_price + (i + 1) * std_price / 2
                        amount = np.random.uniform(0.1, 1.0)
                        self.order_books[symbol]['asks'].append((price, amount))
                else:
                    # Default order book
                    mid_price = self.prices[symbol]
                    for i in range(5):
                        self.order_books[symbol]['bids'].append(
                            (mid_price * (1 - 0.001 * (i + 1)), 1.0)
                        )
                        self.order_books[symbol]['asks'].append(
                            (mid_price * (1 + 0.001 * (i + 1)), 1.0)
                        )
        
        def get_data(self) -> Dict:
            """Get current market data."""
            return {
                'prices': self.prices,
                'order_books': self.order_books,
                'volatility': self.volatility,
                'trend': self.trend,
                'time': self.time
            }
        
        def get_price(self, symbol: str) -> float:
            """Get current price for a symbol."""
            return self.prices.get(symbol, 0.0)
        
        def record_trade(
            self,
            agent: Any,
            action: str,
            symbol: str,
            amount: float,
            price: float
        ):
            """Record a trade in the market."""
            self.trade_history.append({
                'agent': agent.unique_id,
                'action': action,
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'timestamp': self.time
            })
            
            # Update price based on trade
            if action == 'buy':
                self.prices[symbol] *= (1 + 0.0001 * amount)
            else:
                self.prices[symbol] *= (1 - 0.0001 * amount)
    
    
    # Strategy classes (simplified implementations)
    class MarketMakingStrategy:
        def generate_signals(self, market_data: Dict) -> Dict:
            return {}
    
    class ArbitrageStrategy:
        def generate_signals(self, market_data: Dict) -> Dict:
            return {}
    
    class TrendFollowingStrategy:
        def generate_signals(self, market_data: Dict) -> Dict:
            return {}
    
    class MeanReversionStrategy:
        def generate_signals(self, market_data: Dict) -> Dict:
            return {}
    
    class RandomTradingStrategy:
        def generate_signals(self, market_data: Dict) -> Dict:
            return {}

else:
    # Placeholder classes when Mesa is not available
    class MultiAgentTradingModel:
        def __init__(self, *args, **kwargs):
            logger.warning("Mesa not available. Multi-agent system disabled.")
        
        def step(self):
            pass

