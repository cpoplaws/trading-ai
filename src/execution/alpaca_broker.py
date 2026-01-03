"""
Alpaca broker implementation.

This module provides integration with Alpaca's trading API for both
paper trading and live trading.
"""
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from execution.broker_interface import (
    Account,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker implementation.

    Connects to Alpaca's trading API for paper or live trading.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """
        Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key (optional, defaults to env)
            secret_key: Alpaca secret key (optional, defaults to env)
            base_url: Override base URL (optional)
            paper_trading: Use paper trading (True) or live trading (False)
        """
        self.paper_trading = paper_trading
        self.connected = False

        # Load API credentials from environment if not provided
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not found in environment variables")
            logger.warning("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")

        # Set base URLs
        if base_url:
            self.base_url = base_url
        elif paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"

        # Create session with auth headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key or "",
                "APCA-API-SECRET-KEY": self.secret_key or "",
                "Content-Type": "application/json",
            }
        )

        logger.info(f"AlpacaBroker initialized (Paper: {self.paper_trading})")

    def connect(self) -> bool:
        """Establish connection to Alpaca API with basic retry logic."""
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/v2/account", headers=self.session.headers
                )
                if response.status_code == 200:
                    account_data = response.json()
                    self.connected = True
                    logger.info(
                        f"✅ Connected to Alpaca - Account: {account_data.get('account_number', 'N/A')}"
                    )
                    return True
                logger.error(
                    f"❌ Connection failed (attempt {attempt}/{attempts}): {response.status_code} - {response.text}"
                )
            except Exception as e:
                logger.error(f"❌ Connection error (attempt {attempt}/{attempts}): {str(e)}")
            if attempt < attempts:
                import time

                time.sleep(1 * attempt)
        return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self.session.close()
        self.connected = False
        logger.info("Disconnected from Alpaca")

    def get_account_info(self) -> Account:
        """Get account information."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account", headers=self.session.headers
            )
            if response.status_code == 200:
                data = response.json()
                return Account(
                    account_number=data.get("account_number", ""),
                    cash=float(data.get("cash", 0)),
                    portfolio_value=float(data.get("portfolio_value", 0)),
                    buying_power=float(data.get("buying_power", 0)),
                    equity=float(data.get("equity", 0)),
                    last_equity=float(data.get("last_equity", 0)),
                    long_market_value=float(data.get("long_market_value", 0)),
                    short_market_value=float(data.get("short_market_value", 0)),
                )
            else:
                logger.error(f"Failed to get account info: {response.text}")
                raise Exception("Failed to get account info")
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions", headers=self.session.headers
            )
            if response.status_code == 200:
                positions_data = response.json()
                positions = []
                for pos in positions_data:
                    positions.append(
                        Position(
                            symbol=pos["symbol"],
                            qty=float(pos["qty"]),
                            avg_entry_price=float(pos["avg_entry_price"]),
                            current_price=float(pos["current_price"]),
                            market_value=float(pos["market_value"]),
                            unrealized_pl=float(pos["unrealized_pl"]),
                            unrealized_plpc=float(pos["unrealized_plpc"]) * 100,
                            side=pos["side"],
                        )
                    )
                return positions
            else:
                logger.error(f"Failed to get positions: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions/{symbol}", headers=self.session.headers
            )
            if response.status_code == 200:
                pos = response.json()
                return Position(
                    symbol=pos["symbol"],
                    qty=float(pos["qty"]),
                    avg_entry_price=float(pos["avg_entry_price"]),
                    current_price=float(pos["current_price"]),
                    market_value=float(pos["market_value"]),
                    unrealized_pl=float(pos["unrealized_pl"]),
                    unrealized_plpc=float(pos["unrealized_plpc"]) * 100,
                    side=pos["side"],
                )
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None

    def place_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        side: OrderSide = OrderSide.BUY,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> Optional[Order]:
        """Place an order."""
        try:
            qty = qty if qty is not None else quantity
            if qty is None:
                raise ValueError("Quantity (qty or quantity) is required")

            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side.value if isinstance(side, Enum) else side,
                "type": order_type.value if isinstance(order_type, Enum) else order_type,
                "time_in_force": time_in_force.value
                if isinstance(time_in_force, Enum)
                else time_in_force,
            }

            if limit_price is not None:
                order_data["limit_price"] = limit_price

            if stop_price is not None:
                order_data["stop_price"] = stop_price

            response = requests.post(
                f"{self.base_url}/v2/orders",
                json=order_data,
                headers=self.session.headers,
            )

            if response.status_code in (200, 201):
                order_resp = response.json()
                logger.info(
                    f"✅ Order placed: {order_data['side'].upper()} {qty} {symbol} @ {order_data['type']}"
                )
                return self._parse_order(order_resp)
            logger.error(f"❌ Order failed: {response.text}")
            return None

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}", headers=self.session.headers
            )
            if response.status_code == 204:
                logger.info(f"✅ Order {order_id} canceled")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            return False

    def modify_order(self, order_id: str, new_params: Dict) -> bool:
        """Modify an open order."""
        try:
            response = requests.patch(
                f"{self.base_url}/v2/orders/{order_id}",
                json=new_params,
                headers=self.session.headers,
            )
            if response.status_code in (200, 201):
                logger.info(f"✅ Order {order_id} modified")
                return True
            logger.error(f"Failed to modify order {order_id}: {response.text}")
            return False
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}", headers=self.session.headers
            )
            if response.status_code == 200:
                return self._parse_order(response.json())
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting order: {str(e)}")
            return None

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status."""
        try:
            params = {}
            if status:
                params["status"] = status.value

            response = requests.get(
                f"{self.base_url}/v2/orders", params=params, headers=self.session.headers
            )
            if response.status_code == 200:
                orders_data = response.json()
                return [self._parse_order(order) for order in orders_data]
            else:
                logger.error(f"Failed to get orders: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            response = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                },
            )
            if response.status_code == 200:
                data = response.json()
                # Use mid-price (average of bid and ask)
                bid = float(data["quote"]["bp"])
                ask = float(data["quote"]["ap"])
                return (bid + ask) / 2
            else:
                logger.warning(f"Could not get price for {symbol}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def _parse_order(self, order_data: Dict) -> Order:
        """Parse order data from API response."""
        created_at_raw = order_data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            if created_at_raw
            else datetime.now()
        )
        time_in_force = order_data.get("time_in_force", TimeInForce.DAY.value)
        return Order(
            order_id=order_data.get("id", ""),
            symbol=order_data.get("symbol", ""),
            qty=float(order_data.get("qty") or order_data.get("quantity") or 0),
            side=OrderSide(order_data.get("side", OrderSide.BUY.value)),
            order_type=OrderType(order_data.get("type", OrderType.MARKET.value)),
            time_in_force=TimeInForce(time_in_force),
            status=OrderStatus(order_data.get("status", OrderStatus.NEW.value)),
            created_at=created_at,
            filled_qty=float(order_data.get("filled_qty", 0)),
            filled_avg_price=float(order_data.get("filled_avg_price", 0))
            if order_data.get("filled_avg_price")
            else 0.0,
            limit_price=float(order_data["limit_price"]) if order_data.get("limit_price") else None,
            stop_price=float(order_data["stop_price"]) if order_data.get("stop_price") else None,
        )


if __name__ == "__main__":
    # Test Alpaca broker
    print("🏦 Testing Alpaca Broker...")

    broker = AlpacaBroker(paper_trading=True)

    if broker.connect():
        print("\n📊 Account Info:")
        account = broker.get_account_info()
        print(f"  Cash: ${account.cash:,.2f}")
        print(f"  Portfolio Value: ${account.portfolio_value:,.2f}")
        print(f"  Buying Power: ${account.buying_power:,.2f}")

        print("\n📈 Positions:")
        positions = broker.get_positions()
        if positions:
            for pos in positions:
                print(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}")
                print(f"    Current: ${pos.current_price:.2f}, P/L: ${pos.unrealized_pl:.2f}")
        else:
            print("  No open positions")

        broker.disconnect()
        print("\n✅ Test complete!")
    else:
        print("❌ Failed to connect to Alpaca")
