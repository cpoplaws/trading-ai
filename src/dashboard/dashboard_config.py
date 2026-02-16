"""
Dashboard Configuration
Settings and data sources for the unified dashboard.
"""
import os
from dataclasses import dataclass
from typing import Optional
import redis
import psycopg2


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    # Display settings
    auto_refresh_seconds: int = 30
    show_debug_info: bool = False
    theme: str = "dark"  # dark or light

    # Data sources
    use_live_data: bool = False
    redis_enabled: bool = False
    postgres_enabled: bool = False

    # Redis connection
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # PostgreSQL connection
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_ai"
    postgres_user: str = "trading_user"
    postgres_password: str = ""

    # Features
    enable_exports: bool = True
    enable_agent_monitoring: bool = True
    enable_realtime_charts: bool = True

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            auto_refresh_seconds=int(os.getenv("DASHBOARD_REFRESH", "30")),
            use_live_data=os.getenv("DASHBOARD_LIVE_DATA", "false").lower() == "true",
            redis_enabled=os.getenv("REDIS_ENABLED", "false").lower() == "true",
            postgres_enabled=os.getenv("POSTGRES_ENABLED", "false").lower() == "true",
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_db=os.getenv("POSTGRES_DB", "trading_ai"),
            postgres_user=os.getenv("POSTGRES_USER", "trading_user"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
        )


class DataConnector:
    """Connects dashboard to live data sources."""

    def __init__(self, config: DashboardConfig):
        """Initialize data connector."""
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.postgres_conn: Optional[psycopg2.extensions.connection] = None

        self._connect()

    def _connect(self):
        """Connect to data sources."""
        # Redis
        if self.config.redis_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True,
                    socket_timeout=2
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e}")
                self.redis_client = None

        # PostgreSQL
        if self.config.postgres_enabled and self.config.postgres_password:
            try:
                self.postgres_conn = psycopg2.connect(
                    host=self.config.postgres_host,
                    port=self.config.postgres_port,
                    database=self.config.postgres_db,
                    user=self.config.postgres_user,
                    password=self.config.postgres_password,
                    connect_timeout=2
                )
            except Exception as e:
                print(f"⚠️  PostgreSQL connection failed: {e}")
                self.postgres_conn = None

    def get_portfolio_value(self) -> Optional[float]:
        """Get current portfolio value from live data."""
        if not self.redis_client:
            return None

        try:
            value = self.redis_client.get("portfolio:total_value")
            return float(value) if value else None
        except:
            return None

    def get_agent_status(self) -> Optional[dict]:
        """Get agent swarm status from live data."""
        if not self.redis_client:
            return None

        try:
            status = self.redis_client.hgetall("agent:status")
            return status if status else None
        except:
            return None

    def get_recent_trades(self, limit: int = 10) -> list:
        """Get recent trades from database."""
        if not self.postgres_conn:
            return []

        try:
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                SELECT timestamp, strategy, symbol, side, quantity, price, pnl
                FROM trades
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

            trades = cursor.fetchall()
            cursor.close()
            return trades
        except:
            return []

    def get_strategy_performance(self) -> dict:
        """Get strategy performance metrics."""
        if not self.redis_client:
            return {}

        try:
            # Get all strategy metrics from Redis
            strategies = {}
            keys = self.redis_client.keys("strategy:*:return")

            for key in keys:
                strategy_name = key.split(":")[1]
                return_pct = float(self.redis_client.get(key) or 0)
                win_rate = float(self.redis_client.get(f"strategy:{strategy_name}:win_rate") or 0)
                trades = int(self.redis_client.get(f"strategy:{strategy_name}:trades") or 0)

                strategies[strategy_name] = {
                    'return': return_pct,
                    'win_rate': win_rate,
                    'trades': trades
                }

            return strategies
        except:
            return {}

    def is_connected(self) -> dict:
        """Check connection status."""
        return {
            'redis': self.redis_client is not None,
            'postgres': self.postgres_conn is not None
        }

    def close(self):
        """Close connections."""
        if self.postgres_conn:
            self.postgres_conn.close()
        if self.redis_client:
            self.redis_client.close()
