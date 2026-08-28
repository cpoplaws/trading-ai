"""
Quick test to verify Alpaca API keys are working

Credentials are read from the environment ONLY. Never hard-code keys in this file.
Use a disposable, narrowly scoped paper-trading key.

Run with:
    ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python test_alpaca.py
"""
import os
import sys

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

missing = [
    name
    for name, value in (
        ("ALPACA_API_KEY", ALPACA_API_KEY),
        ("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY),
    )
    if not value
]
if missing:
    print(f"❌ Missing required environment variable(s): {', '.join(missing)}")
    print("Set them in your secret store / environment before running this test.")
    sys.exit(1)

print("Testing Alpaca API connection...")
print(f"API Key: {ALPACA_API_KEY[:4]}...")
print(f"Secret Key length: {len(ALPACA_SECRET_KEY)}")

try:
    from alpaca.trading.client import TradingClient

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    account = client.get_account()

    print("\n✅ SUCCESS! Connected to Alpaca")
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("1. Invalid API keys - check they're from paper trading account")
    print("2. alpaca-py not installed - run: pip install alpaca-py")
    print("3. Network/firewall blocking connection")
    sys.exit(1)
