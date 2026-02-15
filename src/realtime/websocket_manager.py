"""
WebSocket Manager
Manages WebSocket connections to multiple exchanges for real-time data.
"""
import asyncio
import json
import logging
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from enum import Enum
import websockets
from dataclasses import dataclass, field
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration."""
    url: str
    name: str
    heartbeat_interval: int = 30  # seconds
    reconnect_delay: int = 5  # seconds
    max_reconnect_attempts: int = 10
    timeout: int = 10  # seconds
    ping_interval: int = 20  # seconds
    ping_timeout: int = 10  # seconds


@dataclass
class MarketData:
    """Market data message."""
    exchange: str
    symbol: str
    data_type: str  # 'ticker', 'trade', 'orderbook', 'kline'
    timestamp: datetime
    data: dict
    raw_message: dict = field(default_factory=dict)


class WebSocketManager:
    """
    Manages WebSocket connections to exchanges.

    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong monitoring
    - Message queuing and callback handling
    - Connection pooling
    - Error recovery
    """

    def __init__(self, config: WebSocketConfig):
        """
        Initialize WebSocket manager.

        Args:
            config: WebSocket configuration
        """
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.ws = None
        self.reconnect_attempts = 0
        self.last_message_time = None
        self.message_count = 0
        self.error_count = 0

        # Callbacks
        self.message_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        self.state_change_handlers: List[Callable] = []

        # Tasks
        self.receive_task = None
        self.heartbeat_task = None

        logger.info(f"WebSocket manager initialized: {config.name}")

    def on_message(self, handler: Callable):
        """Register message handler callback."""
        self.message_handlers.append(handler)
        return handler

    def on_error(self, handler: Callable):
        """Register error handler callback."""
        self.error_handlers.append(handler)
        return handler

    def on_state_change(self, handler: Callable):
        """Register state change handler callback."""
        self.state_change_handlers.append(handler)
        return handler

    def _set_state(self, new_state: ConnectionState):
        """Update connection state and notify handlers."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.info(f"{self.config.name}: {old_state.value} -> {new_state.value}")

            # Notify state change handlers
            for handler in self.state_change_handlers:
                try:
                    handler(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change handler error: {e}")

    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self._set_state(ConnectionState.CONNECTING)

            self.ws = await websockets.connect(
                self.config.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.timeout
            )

            self._set_state(ConnectionState.CONNECTED)
            self.reconnect_attempts = 0
            self.last_message_time = time.time()

            # Start background tasks
            self.receive_task = asyncio.create_task(self._receive_messages())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            logger.info(f"Connected to {self.config.name}")

        except Exception as e:
            self._set_state(ConnectionState.FAILED)
            logger.error(f"Connection failed to {self.config.name}: {e}")
            await self._handle_error(e)
            raise

    async def disconnect(self):
        """Close WebSocket connection."""
        logger.info(f"Disconnecting from {self.config.name}")

        # Cancel background tasks
        if self.receive_task:
            self.receive_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        # Close connection
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

        self._set_state(ConnectionState.DISCONNECTED)

    async def send(self, message: dict):
        """
        Send message through WebSocket.

        Args:
            message: Message to send (will be JSON serialized)
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(f"Cannot send message, state is {self.state.value}")

        try:
            await self.ws.send(json.dumps(message))
            logger.debug(f"Sent message: {message}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self._handle_error(e)
            raise

    async def _receive_messages(self):
        """Receive and process messages from WebSocket."""
        try:
            async for message in self.ws:
                try:
                    self.message_count += 1
                    self.last_message_time = time.time()

                    # Parse message
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')

                    data = json.loads(message) if isinstance(message, str) else message

                    # Process message
                    await self._process_message(data)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.error_count += 1

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Receive error: {e}")
            await self._handle_error(e)

    async def _process_message(self, data: dict):
        """Process received message and call handlers."""
        # Call message handlers
        for handler in self.message_handlers:
            try:
                # Support both sync and async handlers
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Message handler error: {e}")

    async def _heartbeat_monitor(self):
        """Monitor connection health via heartbeat."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Check if we're receiving messages
                if self.last_message_time:
                    time_since_last = time.time() - self.last_message_time

                    if time_since_last > self.config.heartbeat_interval * 2:
                        logger.warning(f"No messages for {time_since_last:.0f}s, reconnecting")
                        await self._handle_disconnect()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _handle_disconnect(self):
        """Handle disconnection and attempt reconnection."""
        self._set_state(ConnectionState.RECONNECTING)

        # Attempt reconnection
        await self._reconnect()

    async def _handle_error(self, error: Exception):
        """Handle errors and notify error handlers."""
        self.error_count += 1

        # Call error handlers
        for handler in self.error_handlers:
            try:
                result = handler(error)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error handler failed: {e}")

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            self.reconnect_attempts += 1

            # Exponential backoff
            delay = self.config.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
            delay = min(delay, 300)  # Max 5 minutes

            logger.info(
                f"Reconnection attempt {self.reconnect_attempts}/"
                f"{self.config.max_reconnect_attempts} in {delay}s"
            )

            await asyncio.sleep(delay)

            try:
                await self.connect()
                logger.info("Reconnection successful")
                return
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

        # Max attempts reached
        self._set_state(ConnectionState.FAILED)
        logger.error("Max reconnection attempts reached")

    def get_stats(self) -> dict:
        """Get connection statistics."""
        uptime = None
        if self.last_message_time:
            uptime = time.time() - self.last_message_time

        return {
            'name': self.config.name,
            'state': self.state.value,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'reconnect_attempts': self.reconnect_attempts,
            'last_message_time': self.last_message_time,
            'uptime_seconds': uptime
        }


class WebSocketPool:
    """
    Manages multiple WebSocket connections.

    Features:
    - Connection pooling
    - Broadcast subscriptions
    - Unified message routing
    - Health monitoring
    """

    def __init__(self):
        """Initialize WebSocket pool."""
        self.connections: Dict[str, WebSocketManager] = {}
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)

        logger.info("WebSocket pool initialized")

    def add_connection(self, name: str, config: WebSocketConfig) -> WebSocketManager:
        """
        Add a WebSocket connection to the pool.

        Args:
            name: Connection name (unique identifier)
            config: WebSocket configuration

        Returns:
            WebSocket manager instance
        """
        if name in self.connections:
            raise ValueError(f"Connection '{name}' already exists")

        manager = WebSocketManager(config)
        self.connections[name] = manager

        # Register message router
        manager.on_message(lambda msg: self._route_message(name, msg))

        logger.info(f"Added connection: {name}")
        return manager

    def get_connection(self, name: str) -> Optional[WebSocketManager]:
        """Get connection by name."""
        return self.connections.get(name)

    async def connect_all(self):
        """Connect all WebSocket connections."""
        tasks = [conn.connect() for conn in self.connections.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any connection failures
        for name, result in zip(self.connections.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect {name}: {result}")

    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        tasks = [conn.disconnect() for conn in self.connections.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe(self, channel: str, handler: Callable):
        """
        Subscribe to messages from specific channel.

        Args:
            channel: Channel name (e.g., 'ticker', 'trade', 'orderbook')
            handler: Callback function for messages
        """
        self.subscriptions[channel].append(handler)
        logger.info(f"Subscribed to channel: {channel}")

    def _route_message(self, connection_name: str, message: dict):
        """Route message to appropriate subscribers."""
        # Determine channel from message
        channel = self._extract_channel(message)

        if channel and channel in self.subscriptions:
            for handler in self.subscriptions[channel]:
                try:
                    handler(connection_name, message)
                except Exception as e:
                    logger.error(f"Subscription handler error for {channel}: {e}")

    def _extract_channel(self, message: dict) -> Optional[str]:
        """Extract channel name from message (exchange-specific)."""
        # Common patterns
        if 'channel' in message:
            return message['channel']
        if 'type' in message:
            return message['type']
        if 'e' in message:  # Binance event type
            return message['e']

        return None

    def get_stats(self) -> dict:
        """Get statistics for all connections."""
        return {
            name: conn.get_stats()
            for name, conn in self.connections.items()
        }

    def get_healthy_connections(self) -> List[str]:
        """Get list of healthy connection names."""
        return [
            name for name, conn in self.connections.items()
            if conn.state == ConnectionState.CONNECTED
        ]

    async def send_to_connection(self, name: str, message: dict):
        """Send message to specific connection."""
        conn = self.get_connection(name)
        if not conn:
            raise ValueError(f"Connection '{name}' not found")

        await conn.send(message)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🌐 WebSocket Manager Demo")
    print("=" * 60)

    async def demo():
        # Create WebSocket pool
        pool = WebSocketPool()

        # Add connection (example with a test WebSocket echo server)
        config = WebSocketConfig(
            url="wss://echo.websocket.org",
            name="echo-server",
            heartbeat_interval=30,
            reconnect_delay=5
        )

        manager = pool.add_connection("echo", config)

        # Register message handler
        @manager.on_message
        def handle_message(data):
            print(f"📨 Received: {data}")

        # Register state change handler
        @manager.on_state_change
        def handle_state_change(old_state, new_state):
            print(f"🔄 State: {old_state.value} -> {new_state.value}")

        # Connect
        print("\n🔌 Connecting...")
        await manager.connect()

        # Send test messages
        print("\n📤 Sending test messages...")
        for i in range(3):
            await manager.send({"test": f"message {i+1}"})
            await asyncio.sleep(1)

        # Get stats
        print("\n📊 Connection Stats:")
        stats = manager.get_stats()
        print(f"   State: {stats['state']}")
        print(f"   Messages: {stats['message_count']}")
        print(f"   Errors: {stats['error_count']}")

        # Disconnect
        print("\n🔌 Disconnecting...")
        await manager.disconnect()

        print("\n✅ WebSocket manager demo complete!")

    # Run demo
    asyncio.run(demo())
