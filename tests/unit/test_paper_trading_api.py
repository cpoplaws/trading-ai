"""
Test Paper Trading API Endpoints
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from api.main import app
import json

client = TestClient(app)

print("🧪 Testing Paper Trading API Endpoints")
print("=" * 60)

# Test 1: Get overview
print("\n1. GET /api/v1/paper-trading/ (Overview)")
print("-" * 60)
response = client.get("/api/v1/paper-trading/")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"System: {data['system']}")
    print(f"Version: {data['version']}")
    print(f"Exchanges: {', '.join(data['exchanges'])}")
else:
    print(f"Error: {response.text}")

# Test 2: Get portfolio
print("\n2. GET /api/v1/paper-trading/portfolio")
print("-" * 60)
response = client.get("/api/v1/paper-trading/portfolio?eth_price=2000")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Initial Value: ${data['initial_value']:,.2f}")
    print(f"Current Value: ${data['current_value']:,.2f}")
    print(f"Total P&L: ${data['total_pnl']:,.2f}")
    print(f"Balances: {len(data['balances'])} tokens")
    for bal in data['balances']:
        print(f"  {bal['symbol']}: {bal['available']:.4f}")
else:
    print(f"Error: {response.text}")

# Test 3: Execute buy order
print("\n3. POST /api/v1/paper-trading/orders (Buy ETH)")
print("-" * 60)
order_request = {
    "exchange": "coinbase",
    "symbol": "ETH-USD",
    "side": "buy",
    "quantity": 2.0,
    "current_price": 2000.0,
    "order_type": "market"
}
response = client.post("/api/v1/paper-trading/orders", json=order_request)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Order ID: {data['order_id']}")
    print(f"Status: {data['status']}")
    print(f"Filled: {data['filled_quantity']} {data['symbol'].split('-')[0]} @ ${data['avg_fill_price']:.2f}")
    print(f"Fees: ${data['fees']:.2f}")
    print(f"Total Cost: ${data['total_cost']:.2f}")
    order1_id = data['order_id']
else:
    print(f"Error: {response.text}")

# Test 4: Execute sell order
print("\n4. POST /api/v1/paper-trading/orders (Sell ETH)")
print("-" * 60)
order_request = {
    "exchange": "uniswap",
    "symbol": "ETH-USD",
    "side": "sell",
    "quantity": 1.0,
    "current_price": 2100.0,
    "order_type": "market"
}
response = client.post("/api/v1/paper-trading/orders", json=order_request)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Order ID: {data['order_id']}")
    print(f"Filled: {data['filled_quantity']} {data['symbol'].split('-')[0]} @ ${data['avg_fill_price']:.2f}")
    print(f"Fees: ${data['fees']:.2f}, Gas: ${data['gas_cost']:.2f}")
    print(f"Total Received: ${data['total_cost']:.2f}")
else:
    print(f"Error: {response.text}")

# Test 5: Get portfolio after trades
print("\n5. GET /api/v1/paper-trading/portfolio (After trades)")
print("-" * 60)
response = client.get("/api/v1/paper-trading/portfolio?eth_price=2100")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Current Value: ${data['current_value']:,.2f}")
    print(f"Total P&L: ${data['total_pnl']:,.2f} ({data['total_pnl_percent']:.2f}%)")
    print(f"Net P&L: ${data['net_pnl']:,.2f}")
    print("\nBalances:")
    for bal in data['balances']:
        if bal['available'] > 0:
            print(f"  {bal['symbol']}: {bal['available']:.4f}")
else:
    print(f"Error: {response.text}")

# Test 6: Get order history
print("\n6. GET /api/v1/paper-trading/orders")
print("-" * 60)
response = client.get("/api/v1/paper-trading/orders")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Total Orders: {data['total']}")
    for order in data['orders'][:3]:  # Show first 3
        print(f"  {order['order_id']}: {order['side']} {order['quantity']} {order['symbol']} @ ${order['avg_fill_price']:.2f}")
else:
    print(f"Error: {response.text}")

# Test 7: Get trade history
print("\n7. GET /api/v1/paper-trading/trades")
print("-" * 60)
response = client.get("/api/v1/paper-trading/trades")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Total Trades: {data['total']}")
    for trade in data['trades']:
        if trade['pnl'] is not None:
            pnl_sign = "+" if trade['pnl'] > 0 else ""
            print(f"  {trade['trade_id']}: {trade['side']} {trade['quantity']} {trade['symbol'].split('-')[0]} | "
                  f"P&L: {pnl_sign}${trade['pnl']:.2f} ({pnl_sign}{trade['pnl_percent']:.2f}%)")
else:
    print(f"Error: {response.text}")

# Test 8: Get analytics
print("\n8. GET /api/v1/paper-trading/analytics")
print("-" * 60)
response = client.get("/api/v1/paper-trading/analytics")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Total Trades: {data['total_trades']}")
    print(f"Win Rate: {data['win_rate']:.1f}%")
    print(f"Total P&L: ${data['total_pnl']:.2f}")
    profit_factor = data['profit_factor']
    if profit_factor is None:
        print(f"Profit Factor: N/A (no losses)")
    else:
        print(f"Profit Factor: {profit_factor:.2f}")
else:
    print(f"Error: {response.text}")

# Test 9: Run backtest
print("\n9. POST /api/v1/paper-trading/backtest (SMA Strategy)")
print("-" * 60)

# Generate sample candles
from datetime import datetime, timedelta
candles = []
base_price = 2000.0
for i in range(50):
    if i < 20:
        price = base_price + (i * 10)
    elif i < 35:
        price = base_price + (20 * 10) - ((i - 20) * 15)
    else:
        price = base_price + (35 - 20) * (-15) + ((i - 35) * 20)

    candles.append({
        "timestamp": (datetime.now() - timedelta(hours=50-i)).isoformat(),
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": 1000000.0
    })

backtest_request = {
    "strategy": "sma",
    "candles": candles,
    "initial_capital": 10000.0,
    "exchange": "coinbase",
    "symbol": "ETH-USD",
    "short_window": 10,
    "long_window": 30
}

response = client.post("/api/v1/paper-trading/backtest", json=backtest_request)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Strategy: {data['strategy']}")
    print(f"Trades Executed: {data['trades_executed']}")
    print(f"Final Value: ${data['final_value']:,.2f}")
    print(f"Total P&L: ${data['total_pnl']:,.2f} ({data['total_pnl_percent']:.2f}%)")
    print(f"Net P&L: ${data['net_pnl']:,.2f}")
else:
    print(f"Error: {response.text}")

print("\n" + "=" * 60)
print("✅ Paper Trading API tests complete!")
