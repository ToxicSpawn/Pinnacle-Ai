"""
Phase 1: Core Infrastructure Examples
Demonstrates multi-exchange support, WebSocket streaming, and advanced orders
"""
import asyncio
import logging
from exchange.ccxt_adapter import CCXTAdapter
from exchange.advanced_orders import AdvancedOrderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_multi_exchange():
    """Example: Using multiple exchanges with unified API."""
    print("\n=== Example 1: Multi-Exchange Support ===\n")
    
    adapter = CCXTAdapter()
    
    # List all loaded exchanges
    exchanges = adapter.exchange_manager.list_exchanges()
    print(f"Loaded exchanges: {exchanges}")
    
    # Fetch ticker from different exchanges
    symbols = {
        'kraken': 'BTC/USD',
        'binance': 'BTC/USDT',
        'coinbase': 'BTC-USD'
    }
    
    for exchange_name, symbol in symbols.items():
        if exchange_name in exchanges:
            ticker = await adapter.fetch_ticker(exchange_name, symbol)
            if ticker:
                print(f"{exchange_name.upper()} {symbol}: ${ticker['last']:.2f}")
    
    print("\n✅ Multi-exchange support working!")


async def example_websocket_streaming():
    """Example: Real-time WebSocket data streaming."""
    print("\n=== Example 2: WebSocket Streaming ===\n")
    
    adapter = CCXTAdapter()
    
    # Callback for WebSocket messages
    def handle_ticker(exchange_name: str, data: dict):
        """Handle incoming ticker updates."""
        if exchange_name == 'kraken':
            # Kraken format
            if isinstance(data, list) and len(data) > 1:
                ticker_data = data[1]
                if isinstance(ticker_data, dict) and 'c' in ticker_data:
                    price = ticker_data['c'][0]
                    print(f"📊 {exchange_name.upper()} BTC/USD: ${price}")
        elif exchange_name == 'binance':
            # Binance format
            if 'c' in data:
                print(f"📊 {exchange_name.upper()} BTC/USDT: ${data['c']}")
        elif exchange_name == 'coinbase':
            # Coinbase format
            if 'price' in data:
                print(f"📊 {exchange_name.upper()} BTC-USD: ${data['price']}")
    
    # Subscribe to WebSocket updates
    print("Connecting to WebSocket streams...")
    await adapter.subscribe_websocket('kraken', ['BTC/USD'], handle_ticker)
    await adapter.subscribe_websocket('binance', ['BTC/USDT'], handle_ticker)
    
    # Let it run for a few seconds
    print("Receiving real-time updates (press Ctrl+C to stop)...")
    try:
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        pass
    
    # Disconnect
    await adapter.disconnect_all_websockets()
    print("\n✅ WebSocket streaming example complete!")


async def example_advanced_orders():
    """Example: Using advanced order types."""
    print("\n=== Example 3: Advanced Order Types ===\n")
    
    adapter = CCXTAdapter()
    
    # Check supported order types
    exchange_name = 'binance'  # Binance supports most advanced orders
    supported = adapter.advanced_orders.get_supported_order_types(exchange_name)
    print(f"Supported order types on {exchange_name}:")
    for order_type, is_supported in supported.items():
        status = "✅" if is_supported else "❌"
        print(f"  {status} {order_type}")
    
    # Example: Create a stop-loss order (paper trading only!)
    print("\n⚠️  Note: These are examples. Orders will NOT be placed in this demo.")
    print("To actually place orders, set BOT_MODE=live and use real API keys.\n")
    
    # Stop-loss example (commented out for safety)
    # stop_order = await adapter.advanced_orders.create_stop_loss(
    #     exchange_name='binance',
    #     symbol='BTC/USDT',
    #     side='sell',
    #     amount=0.001,
    #     stop_price=30000.0
    # )
    # if stop_order:
    #     print(f"✅ Stop-loss order placed: {stop_order.get('id')}")
    
    # OCO order example
    # oco_order = await adapter.advanced_orders.create_oco_order(
    #     exchange_name='binance',
    #     symbol='BTC/USDT',
    #     side='sell',
    #     amount=0.001,
    #     price=40000.0,  # Take-profit
    #     stop_price=35000.0,  # Stop-loss
    # )
    # if oco_order:
    #     print(f"✅ OCO order placed: {oco_order.get('id')}")
    
    # Trailing stop example
    # trailing_order = await adapter.advanced_orders.create_trailing_stop(
    #     exchange_name='binance',
    #     symbol='BTC/USDT',
    #     side='sell',
    #     amount=0.001,
    #     trailing_percent=5.0  # 5% trailing stop
    # )
    # if trailing_order:
    #     print(f"✅ Trailing stop placed: {trailing_order.get('id')}")
    
    print("\n✅ Advanced orders example complete!")


async def example_arbitrage_check():
    """Example: Cross-exchange arbitrage opportunity detection."""
    print("\n=== Example 4: Arbitrage Detection ===\n")
    
    adapter = CCXTAdapter()
    
    # Check prices across exchanges
    symbol_mapping = {
        'kraken': 'BTC/USD',
        'binance': 'BTC/USDT',
        'coinbase': 'BTC-USD'
    }
    
    prices = {}
    for exchange_name, symbol in symbol_mapping.items():
        ticker = await adapter.fetch_ticker(exchange_name, symbol)
        if ticker:
            prices[exchange_name] = float(ticker['last'])
            print(f"{exchange_name.upper()}: ${prices[exchange_name]:.2f}")
    
    if len(prices) >= 2:
        # Find price difference
        min_price = min(prices.values())
        max_price = max(prices.values())
        spread = max_price - min_price
        spread_pct = (spread / min_price) * 100
        
        print(f"\nPrice spread: ${spread:.2f} ({spread_pct:.2f}%)")
        
        if spread_pct > 1.0:  # 1% threshold
            min_exchange = min(prices, key=prices.get)
            max_exchange = max(prices, key=prices.get)
            print(f"💰 Arbitrage opportunity detected!")
            print(f"   Buy on {min_exchange.upper()} @ ${prices[min_exchange]:.2f}")
            print(f"   Sell on {max_exchange.upper()} @ ${prices[max_exchange]:.2f}")
            print(f"   Potential profit: {spread_pct:.2f}% (before fees)")
        else:
            print("No significant arbitrage opportunity")
    
    print("\n✅ Arbitrage check complete!")


async def main():
    """Run all examples."""
    print("🚀 Phase 1: Core Infrastructure Examples\n")
    print("=" * 50)
    
    try:
        await example_multi_exchange()
        await asyncio.sleep(1)
        
        # Uncomment to test WebSocket (requires active connection)
        # await example_websocket_streaming()
        
        await example_advanced_orders()
        await asyncio.sleep(1)
        
        await example_arbitrage_check()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

