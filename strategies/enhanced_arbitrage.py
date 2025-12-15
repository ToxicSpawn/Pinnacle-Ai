"""
Enhanced Arbitrage Strategies
Cross-exchange and triangular arbitrage with risk management
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from exchange.ccxt_adapter import CCXTAdapter

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread: float
    spread_pct: float
    profit_after_fees: float
    min_amount: float
    max_amount: float


@dataclass
class TriangularArbitrageOpportunity:
    """Represents a triangular arbitrage opportunity."""
    path: List[str]  # e.g., ['BTC/USDT', 'ETH/USDT', 'BTC/ETH']
    exchanges: List[str]
    implied_rate: float
    market_rate: float
    profit_pct: float
    amount: float


class EnhancedArbitrageStrategy:
    """
    Enhanced arbitrage strategy with risk management and fee calculation.
    
    Features:
    - Cross-exchange arbitrage
    - Triangular arbitrage
    - Fee-aware profit calculation
    - Risk limits and position sizing
    - Execution timing optimization
    """
    
    def __init__(
        self,
        adapter: CCXTAdapter,
        min_profit_pct: float = 0.5,
        max_position_size: float = 1000.0,
        fee_map: Optional[Dict[str, float]] = None
    ):
        """
        Initialize arbitrage strategy.
        
        Args:
            adapter: CCXT adapter for exchange access
            min_profit_pct: Minimum profit percentage to execute (0.5 = 0.5%)
            max_position_size: Maximum position size per trade
            fee_map: Exchange fee map (exchange_name -> fee_bps), defaults to common fees
        """
        self.adapter = adapter
        self.min_profit_pct = min_profit_pct
        self.max_position_size = max_position_size
        
        # Default fee map (basis points)
        self.fee_map = fee_map or {
            'kraken': 16,  # 0.16%
            'binance': 10,  # 0.10%
            'coinbase': 15,  # 0.15%
        }
    
    def _calculate_fee(self, exchange_name: str, amount: float, price: float) -> float:
        """Calculate trading fee for an exchange."""
        fee_bps = self.fee_map.get(exchange_name.lower(), 10)
        notional = amount * price
        return notional * (fee_bps / 10000)
    
    def _calculate_net_profit(
        self,
        buy_exchange: str,
        sell_exchange: str,
        amount: float,
        buy_price: float,
        sell_price: float
    ) -> float:
        """Calculate net profit after fees."""
        buy_fee = self._calculate_fee(buy_exchange, amount, buy_price)
        sell_fee = self._calculate_fee(sell_exchange, amount, sell_price)
        
        gross_profit = (sell_price - buy_price) * amount
        net_profit = gross_profit - buy_fee - sell_fee
        
        return net_profit
    
    async def find_cross_exchange_opportunities(
        self,
        symbol: str,
        exchanges: List[str]
    ) -> List[ArbitrageOpportunity]:
        """
        Find cross-exchange arbitrage opportunities.
        
        Args:
            symbol: Trading pair symbol
            exchanges: List of exchange names to check
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        # Fetch prices from all exchanges
        prices = {}
        for exchange_name in exchanges:
            try:
                ticker = await self.adapter.fetch_ticker(exchange_name, symbol)
                if ticker and 'last' in ticker:
                    prices[exchange_name] = float(ticker['last'])
            except Exception as e:
                logger.warning(f"Failed to fetch price from {exchange_name}: {e}")
                continue
        
        if len(prices) < 2:
            return opportunities
        
        # Find best buy and sell prices
        buy_exchange = min(prices, key=prices.get)
        sell_exchange = max(prices, key=prices.get)
        buy_price = prices[buy_exchange]
        sell_price = prices[sell_exchange]
        
        # Calculate spread
        spread = sell_price - buy_price
        spread_pct = (spread / buy_price) * 100
        
        # Check if profitable after fees
        test_amount = 0.1  # Test with small amount
        net_profit = self._calculate_net_profit(
            buy_exchange, sell_exchange, test_amount, buy_price, sell_price
        )
        profit_pct = (net_profit / (buy_price * test_amount)) * 100
        
        if profit_pct >= self.min_profit_pct:
            opportunity = ArbitrageOpportunity(
                symbol=symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                spread=spread,
                spread_pct=spread_pct,
                profit_after_fees=net_profit,
                min_amount=0.001,  # Minimum trade size
                max_amount=self.max_position_size / buy_price
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    async def execute_cross_exchange_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        amount: float
    ) -> Dict:
        """
        Execute cross-exchange arbitrage trade.
        
        Args:
            opportunity: Arbitrage opportunity
            amount: Amount to trade
            
        Returns:
            Execution result dictionary
        """
        if amount < opportunity.min_amount or amount > opportunity.max_amount:
            return {
                'success': False,
                'error': f'Amount {amount} outside valid range'
            }
        
        try:
            # Execute buy order
            buy_order = await self.adapter.create_order(
                exchange_name=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                side='buy',
                amount=amount,
                order_type='market'
            )
            
            if not buy_order:
                return {'success': False, 'error': 'Buy order failed'}
            
            # Execute sell order
            sell_order = await self.adapter.create_order(
                exchange_name=opportunity.sell_exchange,
                symbol=opportunity.symbol,
                side='sell',
                amount=amount,
                order_type='market'
            )
            
            if not sell_order:
                return {'success': False, 'error': 'Sell order failed'}
            
            # Calculate actual profit
            buy_cost = float(buy_order.get('cost', amount * opportunity.buy_price))
            sell_revenue = float(sell_order.get('cost', amount * opportunity.sell_price))
            actual_profit = sell_revenue - buy_cost
            
            logger.info(
                f"✅ Arbitrage executed: {opportunity.symbol} "
                f"Buy {opportunity.buy_exchange} @ {opportunity.buy_price:.2f}, "
                f"Sell {opportunity.sell_exchange} @ {opportunity.sell_price:.2f}, "
                f"Profit: {actual_profit:.2f}"
            )
            
            return {
                'success': True,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'profit': actual_profit,
                'profit_pct': (actual_profit / buy_cost) * 100
            }
        
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return {'success': False, 'error': str(e)}
    
    async def find_triangular_opportunities(
        self,
        exchange_name: str,
        symbol_paths: List[List[str]]
    ) -> List[TriangularArbitrageOpportunity]:
        """
        Find triangular arbitrage opportunities.
        
        Args:
            exchange_name: Exchange to check
            symbol_paths: List of symbol paths (e.g., [['BTC/USDT', 'ETH/USDT', 'BTC/ETH']])
            
        Returns:
            List of triangular arbitrage opportunities
        """
        opportunities = []
        
        for path in symbol_paths:
            if len(path) != 3:
                continue
            
            try:
                # Fetch prices for all three pairs
                prices = {}
                for symbol in path:
                    ticker = await self.adapter.fetch_ticker(exchange_name, symbol)
                    if ticker and 'last' in ticker:
                        prices[symbol] = float(ticker['last'])
                    else:
                        break
                
                if len(prices) != 3:
                    continue
                
                # Calculate implied rate
                # Example: BTC/USDT -> ETH/USDT -> BTC/ETH
                # implied BTC/USDT = (BTC/ETH) * (ETH/USDT)
                implied_rate = prices[path[0]] * prices[path[1]] / prices[path[2]]
                market_rate = prices[path[0]]
                
                # Check for opportunity
                profit_pct = ((implied_rate - market_rate) / market_rate) * 100
                
                if profit_pct >= self.min_profit_pct:
                    opportunity = TriangularArbitrageOpportunity(
                        path=path,
                        exchanges=[exchange_name],
                        implied_rate=implied_rate,
                        market_rate=market_rate,
                        profit_pct=profit_pct,
                        amount=0.1  # Default amount
                    )
                    opportunities.append(opportunity)
            
            except Exception as e:
                logger.warning(f"Error checking triangular arbitrage: {e}")
                continue
        
        return opportunities
    
    async def execute_triangular_arbitrage(
        self,
        opportunity: TriangularArbitrageOpportunity,
        amount: float
    ) -> Dict:
        """
        Execute triangular arbitrage trade.
        
        Args:
            opportunity: Triangular arbitrage opportunity
            amount: Starting amount
            
        Returns:
            Execution result dictionary
        """
        try:
            exchange_name = opportunity.exchanges[0]
            path = opportunity.path
            
            # Execute trades in sequence
            orders = []
            current_amount = amount
            
            # Trade 1: Buy first pair
            order1 = await self.adapter.create_order(
                exchange_name=exchange_name,
                symbol=path[0],
                side='buy',
                amount=current_amount,
                order_type='market'
            )
            if not order1:
                return {'success': False, 'error': 'First order failed'}
            orders.append(order1)
            
            # Calculate amount for next trade
            # This is simplified - in reality, you'd use the actual filled amount
            current_amount = amount * opportunity.implied_rate / opportunity.market_rate
            
            # Trade 2: Buy second pair
            order2 = await self.adapter.create_order(
                exchange_name=exchange_name,
                symbol=path[1],
                side='buy',
                amount=current_amount,
                order_type='market'
            )
            if not order2:
                return {'success': False, 'error': 'Second order failed'}
            orders.append(order2)
            
            # Trade 3: Sell third pair
            order3 = await self.adapter.create_order(
                exchange_name=exchange_name,
                symbol=path[2],
                side='sell',
                amount=amount,  # Should return to original amount
                order_type='market'
            )
            if not order3:
                return {'success': False, 'error': 'Third order failed'}
            orders.append(order3)
            
            logger.info(
                f"✅ Triangular arbitrage executed: {opportunity.path} "
                f"on {exchange_name}, Profit: {opportunity.profit_pct:.2f}%"
            )
            
            return {
                'success': True,
                'orders': orders,
                'profit_pct': opportunity.profit_pct
            }
        
        except Exception as e:
            logger.error(f"Error executing triangular arbitrage: {e}")
            return {'success': False, 'error': str(e)}

