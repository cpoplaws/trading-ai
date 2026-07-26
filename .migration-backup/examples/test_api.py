#!/usr/bin/env python3
"""
REST API Test Script

Tests all API endpoints to verify functionality.
"""
import requests
import json
from typing import Dict

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "sk_test_12345"  # Test API key

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def test_health_endpoints():
    """Test health check endpoints."""
    print("\n" + "=" * 70)
    print("TESTING HEALTH ENDPOINTS")
    print("=" * 70)

    # Basic health check
    response = requests.get(f"{API_BASE_URL}/health/")
    print(f"\nGET /health/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Detailed health check
    response = requests.get(f"{API_BASE_URL}/health/detailed")
    print(f"\nGET /health/detailed")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")


def test_agent_endpoints():
    """Test agent control endpoints."""
    print("\n" + "=" * 70)
    print("TESTING AGENT ENDPOINTS")
    print("=" * 70)

    # Create agent
    agent_config = {
        "initial_capital": 10000.0,
        "paper_trading": True,
        "check_interval_seconds": 5,
        "max_daily_loss": 500.0,
        "enabled_strategies": ["dca_bot", "momentum"],
        "send_alerts": False
    }

    response = requests.post(
        f"{API_BASE_URL}/api/v1/agents/",
        headers=headers,
        json=agent_config
    )
    print(f"\nPOST /api/v1/agents/")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    agent_id = result.get("agent_id")

    if agent_id:
        # Get agent status
        response = requests.get(
            f"{API_BASE_URL}/api/v1/agents/{agent_id}",
            headers=headers
        )
        print(f"\nGET /api/v1/agents/{agent_id}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        # Start agent
        response = requests.post(
            f"{API_BASE_URL}/api/v1/agents/{agent_id}/start",
            headers=headers
        )
        print(f"\nPOST /api/v1/agents/{agent_id}/start")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        # Get metrics
        response = requests.get(
            f"{API_BASE_URL}/api/v1/agents/{agent_id}/metrics",
            headers=headers
        )
        print(f"\nGET /api/v1/agents/{agent_id}/metrics")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        # Control strategy
        response = requests.post(
            f"{API_BASE_URL}/api/v1/agents/{agent_id}/strategies",
            headers=headers,
            json={"strategy_name": "market_making", "enabled": True}
        )
        print(f"\nPOST /api/v1/agents/{agent_id}/strategies")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        # Stop agent
        response = requests.post(
            f"{API_BASE_URL}/api/v1/agents/{agent_id}/stop",
            headers=headers
        )
        print(f"\nPOST /api/v1/agents/{agent_id}/stop")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_portfolio_endpoints():
    """Test portfolio endpoints."""
    print("\n" + "=" * 70)
    print("TESTING PORTFOLIO ENDPOINTS")
    print("=" * 70)

    # Get portfolio summary
    response = requests.get(
        f"{API_BASE_URL}/api/v1/portfolio/summary",
        headers=headers
    )
    print(f"\nGET /api/v1/portfolio/summary")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Get positions
    response = requests.get(
        f"{API_BASE_URL}/api/v1/portfolio/positions",
        headers=headers
    )
    print(f"\nGET /api/v1/portfolio/positions")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Get trades
    response = requests.get(
        f"{API_BASE_URL}/api/v1/portfolio/trades?limit=5",
        headers=headers
    )
    print(f"\nGET /api/v1/portfolio/trades")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_market_data_endpoints():
    """Test market data endpoints."""
    print("\n" + "=" * 70)
    print("TESTING MARKET DATA ENDPOINTS")
    print("=" * 70)

    # Get ticker
    response = requests.get(
        f"{API_BASE_URL}/api/v1/market/ticker/BTC-USD",
        headers=headers
    )
    print(f"\nGET /api/v1/market/ticker/BTC-USD")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Get order book
    response = requests.get(
        f"{API_BASE_URL}/api/v1/market/orderbook/BTC-USD?depth=5",
        headers=headers
    )
    print(f"\nGET /api/v1/market/orderbook/BTC-USD")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Bids: {result['bids'][:3]}")
    print(f"Asks: {result['asks'][:3]}")


def test_signals_endpoints():
    """Test trading signals endpoints."""
    print("\n" + "=" * 70)
    print("TESTING TRADING SIGNALS ENDPOINTS")
    print("=" * 70)

    # Get active signals
    response = requests.get(
        f"{API_BASE_URL}/api/v1/signals/active",
        headers=headers
    )
    print(f"\nGET /api/v1/signals/active")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # List strategies
    response = requests.get(
        f"{API_BASE_URL}/api/v1/signals/strategies",
        headers=headers
    )
    print(f"\nGET /api/v1/signals/strategies")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)[:300]}...")


def test_risk_endpoints():
    """Test risk management endpoints."""
    print("\n" + "=" * 70)
    print("TESTING RISK MANAGEMENT ENDPOINTS")
    print("=" * 70)

    # Calculate VaR
    response = requests.get(
        f"{API_BASE_URL}/api/v1/risk/var?portfolio_id=test",
        headers=headers
    )
    print(f"\nGET /api/v1/risk/var")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Calculate position size
    response = requests.get(
        f"{API_BASE_URL}/api/v1/risk/position-sizing?symbol=BTC-USD",
        headers=headers
    )
    print(f"\nGET /api/v1/risk/position-sizing")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("TRADING AI REST API TEST SUITE")
    print("=" * 70)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY}")

    try:
        # Test each endpoint group
        test_health_endpoints()
        test_agent_endpoints()
        test_portfolio_endpoints()
        test_market_data_endpoints()
        test_signals_endpoints()
        test_risk_endpoints()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server!")
        print("\nTo start the API server:")
        print("  cd src/api")
        print("  python main.py")
        print("\nOr:")
        print("  uvicorn src.api.main:app --reload")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
