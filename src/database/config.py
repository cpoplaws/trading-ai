"""
Database Configuration
Connection management, session handling, and TimescaleDB setup.
"""
import logging
import os
from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool

from .models import Base, PriceData

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration and connection management."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        """
        Initialize database configuration.

        Args:
            database_url: Database URL (default: from env or SQLite)
            echo: Echo SQL statements
            pool_size: Connection pool size
            max_overflow: Max connections beyond pool_size
            pool_timeout: Timeout for getting connection
            pool_recycle: Recycle connections after seconds
        """
        # Get database URL from env or use default SQLite
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL",
                "sqlite:///trading_ai.db"
            )

        self.database_url = database_url
        self.is_sqlite = database_url.startswith("sqlite")
        self.is_postgres = database_url.startswith("postgresql")

        # Create engine
        engine_kwargs = {
            "echo": echo,
        }

        # Connection pooling (not for SQLite)
        if not self.is_sqlite:
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout,
                "pool_recycle": pool_recycle,
                "pool_pre_ping": True  # Verify connections before using
            })

        self.engine = create_engine(database_url, **engine_kwargs)

        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Scoped session for thread-safety
        self.ScopedSession = scoped_session(self.SessionLocal)

        # Enable foreign keys for SQLite
        if self.is_sqlite:
            self._enable_sqlite_foreign_keys()

        logger.info(f"Database configured: {self._safe_url()}")

    def _safe_url(self) -> str:
        """Get safe URL for logging (hide password)."""
        url = self.database_url
        if "@" in url:
            # Hide password
            parts = url.split("@")
            auth_parts = parts[0].split("://")
            if len(auth_parts) > 1 and ":" in auth_parts[1]:
                user = auth_parts[1].split(":")[0]
                url = f"{auth_parts[0]}://{user}:****@{parts[1]}"
        return url

    def _enable_sqlite_foreign_keys(self):
        """Enable foreign key constraints for SQLite."""
        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        logger.info("SQLite foreign keys enabled")

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all database tables (DANGEROUS!)."""
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped")

    def setup_timescaledb(self):
        """Setup TimescaleDB hypertables for time-series data."""
        if not self.is_postgres:
            logger.warning("TimescaleDB only works with PostgreSQL")
            return

        with self.get_session() as session:
            try:
                # Enable TimescaleDB extension
                session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))

                # Convert price_data to hypertable
                session.execute(text("""
                    SELECT create_hypertable(
                        'price_data',
                        'timestamp',
                        if_not_exists => TRUE,
                        migrate_data => TRUE
                    )
                """))

                # Create continuous aggregate for 1-hour candles
                session.execute(text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS price_data_hourly
                    WITH (timescaledb.continuous) AS
                    SELECT
                        symbol,
                        exchange,
                        time_bucket('1 hour', timestamp) AS bucket,
                        first(open, timestamp) AS open,
                        max(high) AS high,
                        min(low) AS low,
                        last(close, timestamp) AS close,
                        sum(volume) AS volume
                    FROM price_data
                    GROUP BY symbol, exchange, bucket
                    WITH NO DATA
                """))

                # Add refresh policy (refresh last 7 days every 10 minutes)
                session.execute(text("""
                    SELECT add_continuous_aggregate_policy(
                        'price_data_hourly',
                        start_offset => INTERVAL '7 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '10 minutes',
                        if_not_exists => TRUE
                    )
                """))

                # Compression policy (compress data older than 7 days)
                session.execute(text("""
                    SELECT add_compression_policy(
                        'price_data',
                        INTERVAL '7 days',
                        if_not_exists => TRUE
                    )
                """))

                # Retention policy (drop data older than 1 year)
                session.execute(text("""
                    SELECT add_retention_policy(
                        'price_data',
                        INTERVAL '1 year',
                        if_not_exists => TRUE
                    )
                """))

                session.commit()
                logger.info("TimescaleDB hypertables configured")

            except Exception as e:
                logger.error(f"Failed to setup TimescaleDB: {e}")
                session.rollback()
                raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session (context manager).

        Usage:
            with db.get_session() as session:
                user = session.query(User).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_scoped_session(self) -> Session:
        """
        Get thread-local scoped session.

        Usage:
            session = db.get_scoped_session()
            user = session.query(User).first()
            db.remove_scoped_session()  # Clean up
        """
        return self.ScopedSession()

    def remove_scoped_session(self):
        """Remove thread-local scoped session."""
        self.ScopedSession.remove()

    def health_check(self) -> bool:
        """
        Check database connection health.

        Returns:
            True if healthy
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Database stats
        """
        stats = {
            "url": self._safe_url(),
            "is_sqlite": self.is_sqlite,
            "is_postgres": self.is_postgres,
            "pool_size": self.engine.pool.size() if not self.is_sqlite else None,
            "checked_out": self.engine.pool.checkedout() if not self.is_sqlite else None
        }

        try:
            with self.get_session() as session:
                # Count tables
                result = session.execute(text("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                """))
                stats["num_tables"] = result.scalar()
        except Exception:
            stats["num_tables"] = None

        return stats

    def close(self):
        """Close database connections."""
        self.ScopedSession.remove()
        self.engine.dispose()
        logger.info("Database connections closed")


# Global database instance
_db: Optional[DatabaseConfig] = None


def get_database(
    database_url: Optional[str] = None,
    **kwargs
) -> DatabaseConfig:
    """
    Get global database instance (singleton).

    Args:
        database_url: Database URL
        **kwargs: Additional config options

    Returns:
        Database configuration
    """
    global _db
    if _db is None:
        _db = DatabaseConfig(database_url, **kwargs)
    return _db


def init_database(
    database_url: Optional[str] = None,
    create_tables: bool = True,
    setup_timescale: bool = False,
    **kwargs
) -> DatabaseConfig:
    """
    Initialize database (convenience function).

    Args:
        database_url: Database URL
        create_tables: Whether to create tables
        setup_timescale: Whether to setup TimescaleDB
        **kwargs: Additional config options

    Returns:
        Database configuration
    """
    db = get_database(database_url, **kwargs)

    if create_tables:
        db.create_tables()

    if setup_timescale:
        db.setup_timescaledb()

    return db


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("🗄️  Database Configuration Demo")
    print("=" * 60)

    # Initialize with SQLite
    print("\n1. Initializing database...")
    db = init_database(
        database_url="sqlite:///test_trading.db",
        create_tables=True,
        echo=False
    )

    print(f"\n2. Database stats:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Health check
    print(f"\n3. Health check: {'✅ Healthy' if db.health_check() else '❌ Unhealthy'}")

    # Test session
    print(f"\n4. Testing session...")
    with db.get_session() as session:
        from .models import User
        
        # Try to query (will be empty)
        count = session.query(User).count()
        print(f"   Users in database: {count}")

    # Clean up
    db.close()
    print(f"\n✅ Database configuration demo complete!")
