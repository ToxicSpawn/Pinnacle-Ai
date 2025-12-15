# Quick Start: Phase 1 Core Infrastructure

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys (Choose One Method)

**Method A: Environment Variables (Recommended)**
```bash
export KRAKEN_API_KEY="your_key"
export KRAKEN_API_SECRET="your_secret"
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

**Method B: Config File**
Edit `config/exchanges.yaml` and set your API keys directly.

**Method C: Encrypted Keys**
```bash
python scripts/encrypt_api_keys.py
```

### 3. Test It Works
```bash
python examples/phase1_examples.py
```

## 📝 Basic Usage

### Multi-Exchange Trading
```python
from exchange.ccxt_adapter import CCXTAdapter

adapter = CCXTAdapter()

# Get price from any exchange
ticker = await adapter.fetch_ticker('kraken', 'BTC/USD')
print(f"Price: ${ticker['last']}")
```

### WebSocket Streaming
```python
def handle_update(exchange, data):
    print(f"{exchange}: {data}")

await adapter.subscribe_websocket('kraken', ['BTC/USD'], handle_update)
```

### Advanced Orders
```python
# Stop-loss
await adapter.advanced_orders.create_stop_loss(
    'binance', 'BTC/USDT', 'sell', 0.001, stop_price=30000
)
```

## 🔧 Configuration

Edit `config/exchanges.yaml` to:
- Enable/disable exchanges
- Set symbols to trade
- Configure sandbox mode
- Set up WebSocket subscriptions

## ⚠️ Important

- Always test in sandbox mode first
- Never commit `.encryption_key` to git
- Start with small amounts
- Check exchange support for advanced orders

## 📚 Full Documentation

See `PHASE1_IMPLEMENTATION.md` for complete details.

