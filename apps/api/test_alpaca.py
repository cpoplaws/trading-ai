"""
Quick test to verify Alpaca API keys are working
Run with: python test_alpaca.py
"""
import os

# Use the keys from Railway
ALPACA_API_KEY = "PKK242F67M34YRFUNDT7IIQASH"
ALPACA_SECRET_KEY = "7JZ5Wv4Zb8grKYUpbYpQ6qoxB5VhTmV77hUiiadaAhqM"

print("Testing Alpaca API connection...")
print(f"API Key: {ALPACA_API_KEY[:8]}...")
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
