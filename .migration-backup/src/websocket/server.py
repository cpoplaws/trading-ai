"""
WebSocket Server for Real-time Updates
Broadcasts live trading data, portfolio updates, and alerts to connected clients.
"""
import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Try importing websockets
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not available. Install with: pip install websockets")


class MessageType(Enum):
    """WebSocket message types."""
    PRICE_UPDATE = "price_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    TRADE_EXECUTED = "trade_executed"
    ALERT = "alert"
    ML_PREDICTION = "ml_prediction"
    PATTERN_DETECTED = "pattern_detected"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"


@dataclass
class PriceUpdate:
    """Real-time price update."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PortfolioUpdate:
    """Portfolio update."""
    total_value: float
    pnl: float
    pnl_percent: float
    balances: Dict[str, float]
    timestamp: datetime

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TradeExecuted:
    """Trade execution notification."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:
    """Alert notification."""
    alert_type: str
    severity: str  # info, warning, error
    message: str
    timestamp: datetime

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class WebSocketServer:
    """
    WebSocket server for real-time updates.

    Manages client connections and broadcasts updates to all connected clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.

        Args:
            host: Server host
            port: Server port
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets required. Install with: pip install websockets")

        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.is_running = False

        logger.info(f"WebSocket server initialized on {host}:{port}")

    async def register(self, websocket: WebSocketServerProtocol):
        """Register new client."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send welcome message
        await self.send_to_client(
            websocket,
            MessageType.SYSTEM_STATUS,
            {"status": "connected", "message": "Connected to trading system"}
        )

    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister client."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def send_to_client(
        self,
        websocket: WebSocketServerProtocol,
        message_type: MessageType,
        data: Dict[str, Any]
    ):
        """
        Send message to specific client.

        Args:
            websocket: Client websocket
            message_type: Message type
            data: Message data
        """
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending to client: {e}")

    async def broadcast(self, message_type: MessageType, data: Dict[str, Any]):
        """
        Broadcast message to all connected clients.

        Args:
            message_type: Message type
            data: Message data
        """
        if not self.clients:
            return

        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all clients
        websockets_list = list(self.clients)
        await asyncio.gather(
            *[ws.send(json.dumps(message)) for ws in websockets_list],
            return_exceptions=True
        )

        logger.debug(f"Broadcasted {message_type.value} to {len(websockets_list)} clients")

    async def broadcast_price_update(self, update: PriceUpdate):
        """Broadcast price update."""
        await self.broadcast(MessageType.PRICE_UPDATE, update.to_dict())

    async def broadcast_portfolio_update(self, update: PortfolioUpdate):
        """Broadcast portfolio update."""
        await self.broadcast(MessageType.PORTFOLIO_UPDATE, update.to_dict())

    async def broadcast_trade(self, trade: TradeExecuted):
        """Broadcast trade execution."""
        await self.broadcast(MessageType.TRADE_EXECUTED, trade.to_dict())

    async def broadcast_alert(self, alert: Alert):
        """Broadcast alert."""
        await self.broadcast(MessageType.ALERT, alert.to_dict())

    async def broadcast_ml_prediction(self, prediction: Dict[str, Any]):
        """Broadcast ML prediction."""
        await self.broadcast(MessageType.ML_PREDICTION, prediction)

    async def broadcast_pattern(self, pattern: Dict[str, Any]):
        """Broadcast pattern detection."""
        await self.broadcast(MessageType.PATTERN_DETECTED, pattern)

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle client connection.

        Args:
            websocket: Client websocket
            path: Connection path
        """
        await self.register(websocket)

        try:
            # Keep connection alive
            async for message in websocket:
                # Handle incoming messages from client
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            await self.unregister(websocket)

    async def handle_message(self, websocket: WebSocketServerProtocol, data: Dict):
        """
        Handle message from client.

        Args:
            websocket: Client websocket
            data: Message data
        """
        message_type = data.get("type")

        if message_type == "ping":
            # Respond to ping
            await self.send_to_client(
                websocket,
                MessageType.HEARTBEAT,
                {"message": "pong"}
            )
        elif message_type == "subscribe":
            # Handle subscription (e.g., to specific symbols)
            symbols = data.get("symbols", [])
            logger.info(f"Client subscribed to: {symbols}")
        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def heartbeat(self):
        """Send periodic heartbeat to clients."""
        while self.is_running:
            await asyncio.sleep(30)  # Every 30 seconds
            await self.broadcast(
                MessageType.HEARTBEAT,
                {"message": "Server alive", "clients": len(self.clients)}
            )

    async def start(self):
        """Start WebSocket server."""
        self.is_running = True

        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.heartbeat())

        # Start server
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("WebSocket server started")
            await asyncio.Future()  # Run forever

    def run(self):
        """Run server (blocking)."""
        asyncio.run(self.start())


# ==================== DATA PUBLISHER ====================

class DataPublisher:
    """
    Publishes data to WebSocket server.

    Used by other components to push updates to clients.
    """

    def __init__(self, server: WebSocketServer):
        """
        Initialize publisher.

        Args:
            server: WebSocket server instance
        """
        self.server = server

    async def publish_price(
        self,
        symbol: str,
        price: float,
        change_24h: float = 0.0,
        volume_24h: float = 0.0
    ):
        """Publish price update."""
        update = PriceUpdate(
            symbol=symbol,
            price=price,
            change_24h=change_24h,
            volume_24h=volume_24h,
            timestamp=datetime.now()
        )
        await self.server.broadcast_price_update(update)

    async def publish_portfolio(
        self,
        total_value: float,
        pnl: float,
        pnl_percent: float,
        balances: Dict[str, float]
    ):
        """Publish portfolio update."""
        update = PortfolioUpdate(
            total_value=total_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            balances=balances,
            timestamp=datetime.now()
        )
        await self.server.broadcast_portfolio_update(update)

    async def publish_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """Publish trade execution."""
        trade = TradeExecuted(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )
        await self.server.broadcast_trade(trade)

    async def publish_alert(
        self,
        alert_type: str,
        severity: str,
        message: str
    ):
        """Publish alert."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now()
        )
        await self.server.broadcast_alert(alert)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    if not WEBSOCKETS_AVAILABLE:
        print("❌ websockets not installed!")
        print("Install with: pip install websockets")
        exit(1)

    print("🌐 WebSocket Server Demo")
    print("=" * 60)

    async def demo():
        """Demo WebSocket server with simulated data."""
        # Create server
        server = WebSocketServer(host="localhost", port=8765)
        publisher = DataPublisher(server)

        # Start server in background
        server_task = asyncio.create_task(server.start())

        # Wait for server to start
        await asyncio.sleep(1)

        print("\n✅ WebSocket server started!")
        print(f"   URL: ws://localhost:8765")
        print(f"   Clients: {len(server.clients)}")
        print("\nTest with:")
        print("   websocat ws://localhost:8765")
        print("   or connect from browser/app")

        # Simulate data updates
        print("\n📊 Simulating data updates...")

        for i in range(10):
            await asyncio.sleep(2)

            # Price update
            await publisher.publish_price(
                symbol="ETH-USDC",
                price=2000 + i * 10,
                change_24h=i * 0.5,
                volume_24h=1000000 + i * 10000
            )

            # Portfolio update
            await publisher.publish_portfolio(
                total_value=10000 + i * 100,
                pnl=i * 50,
                pnl_percent=i * 0.5,
                balances={"ETH": 5.0, "USDC": 5000 + i * 100}
            )

            if i % 3 == 0:
                # Trade notification
                await publisher.publish_trade(
                    order_id=f"ORDER-{i}",
                    symbol="ETH-USDC",
                    side="buy" if i % 2 == 0 else "sell",
                    quantity=1.0,
                    price=2000 + i * 10
                )

            if i % 5 == 0:
                # Alert
                await publisher.publish_alert(
                    alert_type="price_alert",
                    severity="info",
                    message=f"ETH price reached ${2000 + i * 10}"
                )

            print(f"   Update {i+1}/10 sent")

        print("\n✅ Demo complete!")
        print("Server will continue running...")
        print("Press Ctrl+C to stop")

        # Keep running
        await server_task

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
