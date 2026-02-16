#!/usr/bin/env python3
"""
Error Recovery Integration Examples

Demonstrates how to use the retry and error recovery utilities
throughout the trading system.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.retry import retry, async_retry, RetryStrategy, CircuitBreaker
from exchanges.binance_trading_client import BinanceTradingClient, OrderSide
from database.models import Trade, Portfolio, session_scope
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Example 1: API Calls with Retry
# ============================================================

class ResilientBinanceClient(BinanceTradingClient):
    """Binance client with automatic retry on transient failures."""

    @retry(
        max_attempts=3,
        base_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
        exceptions=(ConnectionError, TimeoutError)
    )
    def get_ticker_price_resilient(self, symbol: str) -> float:
        """Get ticker price with automatic retry."""
        return self.get_ticker_price(symbol)

    @retry(
        max_attempts=5,
        base_delay=2.0,
        max_delay=30.0,
        strategy=RetryStrategy.JITTERED
    )
    def place_market_order_resilient(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> dict:
        """Place market order with retry on rate limits."""
        return self.place_market_order(symbol, side, quantity)


# ============================================================
# Example 2: Database Operations with Retry
# ============================================================

@retry(
    max_attempts=3,
    base_delay=0.5,
    strategy=RetryStrategy.EXPONENTIAL,
    exceptions=(Exception,)  # Catch database connection errors
)
def save_trade_resilient(trade_data: dict) -> Trade:
    """
    Save trade to database with automatic retry.

    Handles transient database connection issues.
    """
    with session_scope() as session:
        trade = Trade(**trade_data)
        session.add(trade)
        session.commit()
        logger.info(f"Trade saved: {trade.id}")
        return trade


@retry(max_attempts=3, base_delay=1.0)
def get_portfolio_resilient(portfolio_id: str) -> Portfolio:
    """Get portfolio with retry on connection failures."""
    with session_scope() as session:
        portfolio = session.query(Portfolio).filter_by(id=portfolio_id).first()
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        return portfolio


# ============================================================
# Example 3: WebSocket with Circuit Breaker
# ============================================================

class ResilientWebSocket:
    """WebSocket client with circuit breaker pattern."""

    def __init__(self, url: str):
        self.url = url
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=ConnectionError
        )
        self.connection = None

    def connect(self):
        """Connect with circuit breaker protection."""
        def _connect():
            # Simulate WebSocket connection
            logger.info(f"Connecting to {self.url}...")
            # In real implementation, this would be:
            # self.connection = websocket.create_connection(self.url)
            return True

        return self.circuit_breaker.call(_connect)

    def send_message(self, message: str):
        """Send message with circuit breaker protection."""
        def _send():
            if not self.connection:
                raise ConnectionError("Not connected")
            # self.connection.send(message)
            logger.info(f"Message sent: {message}")
            return True

        return self.circuit_breaker.call(_send)


# ============================================================
# Example 4: Async Operations with Retry
# ============================================================

@async_retry(
    max_attempts=3,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL
)
async def fetch_market_data_async(symbol: str) -> dict:
    """Fetch market data with async retry."""
    # Simulate async API call
    await asyncio.sleep(0.1)

    # In real implementation:
    # async with aiohttp.ClientSession() as session:
    #     async with session.get(f"https://api.example.com/ticker/{symbol}") as resp:
    #         return await resp.json()

    logger.info(f"Fetched market data for {symbol}")
    return {"symbol": symbol, "price": 45000.0}


@async_retry(max_attempts=5, base_delay=2.0)
async def execute_trade_async(order_data: dict) -> dict:
    """Execute trade with retry on rate limits."""
    await asyncio.sleep(0.1)
    logger.info(f"Trade executed: {order_data}")
    return {"order_id": "12345", "status": "FILLED"}


# ============================================================
# Example 5: Custom Retry Callbacks
# ============================================================

def on_retry_callback(attempt: int, exception: Exception):
    """Called on each retry attempt."""
    logger.warning(f"Retry attempt {attempt + 1}: {exception}")
    # Could send alert, update metrics, etc.


def on_failure_callback(exception: Exception):
    """Called when all retries fail."""
    logger.error(f"Operation failed after all retries: {exception}")
    # Could send critical alert, trigger fallback, etc.


@retry(
    max_attempts=3,
    base_delay=1.0,
    on_retry=on_retry_callback,
    on_failure=on_failure_callback
)
def critical_operation():
    """Operation with custom retry callbacks."""
    logger.info("Executing critical operation...")
    # Simulate operation
    return True


# ============================================================
# Example 6: Graceful Degradation
# ============================================================

class MarketDataService:
    """Market data service with graceful degradation."""

    def __init__(self):
        self.primary_source = "binance"
        self.fallback_source = "coinbase"
        self.circuit_breaker_primary = CircuitBreaker(failure_threshold=3)
        self.circuit_breaker_fallback = CircuitBreaker(failure_threshold=3)

    def get_price(self, symbol: str) -> float:
        """
        Get price with graceful degradation.

        Tries primary source, falls back to secondary if primary fails.
        """
        try:
            # Try primary source
            def fetch_primary():
                logger.info(f"Fetching from {self.primary_source}")
                # return binance_client.get_ticker_price(symbol)
                return 45000.0

            return self.circuit_breaker_primary.call(fetch_primary)

        except Exception as e:
            logger.warning(f"Primary source failed: {e}, trying fallback...")

            try:
                # Try fallback source
                def fetch_fallback():
                    logger.info(f"Fetching from {self.fallback_source}")
                    # return coinbase_client.get_price(symbol)
                    return 45100.0

                return self.circuit_breaker_fallback.call(fetch_fallback)

            except Exception as e2:
                logger.error(f"All sources failed: {e2}")
                # Return cached value or raise
                raise


# ============================================================
# Example 7: Transaction Rollback
# ============================================================

class ResilientTradeExecutor:
    """Trade executor with transaction rollback on failure."""

    def __init__(self, exchange_client):
        self.exchange_client = exchange_client

    @retry(max_attempts=2, base_delay=1.0)
    def execute_trade_with_rollback(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> dict:
        """
        Execute trade with automatic rollback on failure.

        If trade execution fails after database update, rolls back database changes.
        """
        trade_data = None
        trade_id = None

        try:
            # Step 1: Save trade intent to database
            with session_scope() as session:
                trade_data = {
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'status': 'PENDING'
                }
                trade = Trade(**trade_data)
                session.add(trade)
                session.commit()
                trade_id = trade.id
                logger.info(f"Trade saved to database: {trade_id}")

            # Step 2: Execute trade on exchange
            order = self.exchange_client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )

            # Step 3: Update trade status
            with session_scope() as session:
                trade = session.query(Trade).filter_by(id=trade_id).first()
                trade.status = 'FILLED'
                trade.order_id = order['orderId']
                session.commit()
                logger.info(f"Trade updated: {trade_id}")

            return order

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

            # Rollback: Mark trade as failed in database
            if trade_id:
                try:
                    with session_scope() as session:
                        trade = session.query(Trade).filter_by(id=trade_id).first()
                        if trade:
                            trade.status = 'FAILED'
                            trade.error_message = str(e)
                            session.commit()
                            logger.info(f"Trade marked as failed: {trade_id}")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")

            raise


# ============================================================
# Example 8: Agent State Recovery
# ============================================================

class RecoverableAgent:
    """Trading agent with state recovery after crashes."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state_file = f"/tmp/agent_{agent_id}_state.json"

    def save_state(self, state: dict):
        """Save agent state for recovery."""
        import json
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            logger.info(f"Agent state saved: {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self) -> dict:
        """Load agent state after crash."""
        import json
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Agent state loaded: {self.agent_id}")
            return state
        except FileNotFoundError:
            logger.info("No saved state found, starting fresh")
            return {}
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return {}

    @retry(max_attempts=3, base_delay=1.0)
    def start_with_recovery(self):
        """Start agent with automatic state recovery."""
        # Try to recover previous state
        state = self.load_state()

        if state:
            logger.info(f"Recovering from previous state: {state}")
            # Restore positions, orders, etc.
        else:
            logger.info("Starting with fresh state")

        # Continue normal operation
        logger.info(f"Agent {self.agent_id} started")


# ============================================================
# Demo
# ============================================================

def main():
    """Demonstrate error recovery patterns."""
    print("=" * 70)
    print("ERROR RECOVERY INTEGRATION EXAMPLES")
    print("=" * 70)

    # Example 1: API with retry
    print("\n1. API Calls with Retry:")
    client = ResilientBinanceClient(testnet=True)
    try:
        price = client.get_ticker_price_resilient('BTCUSDT')
        print(f"   BTC Price: ${price:,.2f}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Example 2: Circuit breaker
    print("\n2. WebSocket with Circuit Breaker:")
    ws = ResilientWebSocket("wss://example.com/ws")
    try:
        ws.connect()
        ws.send_message("Hello")
    except Exception as e:
        print(f"   Circuit breaker opened: {e}")

    # Example 3: Graceful degradation
    print("\n3. Graceful Degradation:")
    market_data = MarketDataService()
    try:
        price = market_data.get_price('BTCUSDT')
        print(f"   Price from fallback: ${price:,.2f}")
    except Exception as e:
        print(f"   All sources failed: {e}")

    # Example 4: Async retry
    print("\n4. Async Operations with Retry:")
    async def test_async():
        data = await fetch_market_data_async('ETHUSDT')
        print(f"   Fetched: {data}")

        order = await execute_trade_async({'symbol': 'ETHUSDT', 'side': 'BUY'})
        print(f"   Order: {order}")

    asyncio.run(test_async())

    # Example 5: State recovery
    print("\n5. Agent State Recovery:")
    agent = RecoverableAgent("agent-001")
    agent.save_state({"portfolio_value": 10000.0, "positions": 3})
    agent.start_with_recovery()

    print("\n✅ Error recovery examples complete!")


if __name__ == '__main__':
    main()
