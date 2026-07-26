"""
Database Manager
High-level interface for database operations with connection pooling.
"""
import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, and_, or_, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.database.models import (
    Base, OHLCV, Trade, OrderBookSnapshot, MarketMetrics,
    AgentDecision, TradingSignal,
    create_hypertables, create_continuous_aggregates
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations.

    Features:
    - Connection pooling for performance
    - Session management with context managers
    - Bulk inserts for efficiency
    - Query helpers for common operations
    - TimescaleDB hypertable management
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        Initialize database manager.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            pool_size: Connection pool size
            max_overflow: Max overflow connections
        """
        # Get config from environment or parameters
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or int(os.getenv('POSTGRES_PORT', 5432))
        self.database = database or os.getenv('POSTGRES_DB', 'trading_db')
        self.user = user or os.getenv('POSTGRES_USER', 'trading_user')
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'postgres')

        # Create connection string
        self.connection_string = (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

        # Create engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL debugging
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info(f"Database manager initialized: {self.host}:{self.port}/{self.database}")

    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup.

        Usage:
            with db.get_session() as session:
                session.query(OHLCV).filter(...).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def init_database(self):
        """
        Initialize database schema.

        Creates tables and TimescaleDB hypertables.
        """
        try:
            # Create tables
            Base.metadata.create_all(self.engine)
            logger.info("✓ Created tables")

            # Create hypertables
            create_hypertables(self.engine)
            logger.info("✓ Created hypertables")

            # Create continuous aggregates
            create_continuous_aggregates(self.engine)
            logger.info("✓ Created continuous aggregates")

            logger.info("Database initialization complete")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False

    # ============================================================
    # OHLCV Operations
    # ============================================================

    def insert_ohlcv(self, data: List[Dict]) -> int:
        """
        Insert OHLCV candlestick data.

        Args:
            data: List of dicts with keys: timestamp, exchange, symbol, interval,
                  open, high, low, close, volume, etc.

        Returns:
            Number of records inserted
        """
        if not data:
            return 0

        with self.get_session() as session:
            records = [OHLCV(**record) for record in data]
            session.bulk_save_objects(records)
            return len(records)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = '1m',
        exchange: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[OHLCV]:
        """
        Query OHLCV data.

        Args:
            symbol: Trading symbol
            interval: Candlestick interval
            exchange: Exchange name (optional)
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum records to return

        Returns:
            List of OHLCV records
        """
        with self.get_session() as session:
            query = session.query(OHLCV).filter(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval
            )

            if exchange:
                query = query.filter(OHLCV.exchange == exchange)

            if start_time:
                query = query.filter(OHLCV.timestamp >= start_time)

            if end_time:
                query = query.filter(OHLCV.timestamp <= end_time)

            return query.order_by(OHLCV.timestamp.desc()).limit(limit).all()

    def get_latest_ohlcv(
        self,
        symbol: str,
        interval: str = '1m',
        exchange: str = None
    ) -> Optional[OHLCV]:
        """Get latest OHLCV record."""
        records = self.get_ohlcv(symbol, interval, exchange, limit=1)
        return records[0] if records else None

    # ============================================================
    # Trade Operations
    # ============================================================

    def insert_trades(self, data: List[Dict]) -> int:
        """Insert trade data."""
        if not data:
            return 0

        with self.get_session() as session:
            records = [Trade(**record) for record in data]
            session.bulk_save_objects(records)
            return len(records)

    def get_trades(
        self,
        symbol: str,
        exchange: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[Trade]:
        """Query trade data."""
        with self.get_session() as session:
            query = session.query(Trade).filter(Trade.symbol == symbol)

            if exchange:
                query = query.filter(Trade.exchange == exchange)

            if start_time:
                query = query.filter(Trade.timestamp >= start_time)

            if end_time:
                query = query.filter(Trade.timestamp <= end_time)

            return query.order_by(Trade.timestamp.desc()).limit(limit).all()

    # ============================================================
    # Order Book Operations
    # ============================================================

    def insert_orderbook_snapshot(self, data: Dict) -> bool:
        """Insert order book snapshot."""
        with self.get_session() as session:
            snapshot = OrderBookSnapshot(**data)
            session.add(snapshot)
            return True

    def get_orderbook_snapshots(
        self,
        symbol: str,
        exchange: str = None,
        start_time: datetime = None,
        limit: int = 100
    ) -> List[OrderBookSnapshot]:
        """Query order book snapshots."""
        with self.get_session() as session:
            query = session.query(OrderBookSnapshot).filter(
                OrderBookSnapshot.symbol == symbol
            )

            if exchange:
                query = query.filter(OrderBookSnapshot.exchange == exchange)

            if start_time:
                query = query.filter(OrderBookSnapshot.timestamp >= start_time)

            return query.order_by(OrderBookSnapshot.timestamp.desc()).limit(limit).all()

    # ============================================================
    # Market Metrics Operations
    # ============================================================

    def insert_market_metrics(self, data: Dict) -> bool:
        """Insert market metrics."""
        with self.get_session() as session:
            metrics = MarketMetrics(**data)
            session.add(metrics)
            return True

    def get_market_metrics(
        self,
        symbol: str,
        interval: str = '1h',
        start_time: datetime = None,
        limit: int = 168  # 1 week of hourly data
    ) -> List[MarketMetrics]:
        """Query market metrics."""
        with self.get_session() as session:
            query = session.query(MarketMetrics).filter(
                MarketMetrics.symbol == symbol,
                MarketMetrics.interval == interval
            )

            if start_time:
                query = query.filter(MarketMetrics.timestamp >= start_time)

            return query.order_by(MarketMetrics.timestamp.desc()).limit(limit).all()

    # ============================================================
    # Agent Decision Operations
    # ============================================================

    def insert_agent_decision(self, data: Dict) -> bool:
        """Insert agent decision."""
        with self.get_session() as session:
            decision = AgentDecision(**data)
            session.add(decision)
            return True

    def get_agent_decisions(
        self,
        agent_id: str = None,
        symbol: str = None,
        start_time: datetime = None,
        limit: int = 1000
    ) -> List[AgentDecision]:
        """Query agent decisions."""
        with self.get_session() as session:
            query = session.query(AgentDecision)

            if agent_id:
                query = query.filter(AgentDecision.agent_id == agent_id)

            if symbol:
                query = query.filter(AgentDecision.symbol == symbol)

            if start_time:
                query = query.filter(AgentDecision.timestamp >= start_time)

            return query.order_by(AgentDecision.timestamp.desc()).limit(limit).all()

    # ============================================================
    # Trading Signal Operations
    # ============================================================

    def insert_trading_signal(self, data: Dict) -> bool:
        """Insert trading signal."""
        with self.get_session() as session:
            signal = TradingSignal(**data)
            session.add(signal)
            return True

    def get_trading_signals(
        self,
        strategy_name: str = None,
        symbol: str = None,
        signal_type: str = None,
        executed: bool = None,
        start_time: datetime = None,
        limit: int = 100
    ) -> List[TradingSignal]:
        """Query trading signals."""
        with self.get_session() as session:
            query = session.query(TradingSignal)

            if strategy_name:
                query = query.filter(TradingSignal.strategy_name == strategy_name)

            if symbol:
                query = query.filter(TradingSignal.symbol == symbol)

            if signal_type:
                query = query.filter(TradingSignal.signal_type == signal_type)

            if executed is not None:
                query = query.filter(TradingSignal.executed == executed)

            if start_time:
                query = query.filter(TradingSignal.timestamp >= start_time)

            return query.order_by(TradingSignal.timestamp.desc()).limit(limit).all()

    # ============================================================
    # Aggregation Queries
    # ============================================================

    def get_price_stats(
        self,
        symbol: str,
        interval: str = '1d',
        lookback_days: int = 30
    ) -> Dict:
        """
        Get price statistics for a symbol.

        Returns:
            Dict with mean, std, min, max, current price
        """
        start_time = datetime.now() - timedelta(days=lookback_days)

        with self.get_session() as session:
            stats = session.query(
                func.avg(OHLCV.close).label('mean'),
                func.stddev(OHLCV.close).label('std'),
                func.min(OHLCV.low).label('min'),
                func.max(OHLCV.high).label('max'),
                func.count(OHLCV.id).label('count')
            ).filter(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
                OHLCV.timestamp >= start_time
            ).first()

            # Get latest price
            latest = session.query(OHLCV.close).filter(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval
            ).order_by(OHLCV.timestamp.desc()).first()

            return {
                'mean': float(stats.mean) if stats.mean else None,
                'std': float(stats.std) if stats.std else None,
                'min': float(stats.min) if stats.min else None,
                'max': float(stats.max) if stats.max else None,
                'current': float(latest[0]) if latest else None,
                'count': int(stats.count) if stats.count else 0
            }

    def get_volume_profile(
        self,
        symbol: str,
        interval: str = '1h',
        lookback_hours: int = 24
    ) -> List[Dict]:
        """
        Get volume profile (volume at different price levels).

        Returns:
            List of dicts with price_level and volume
        """
        start_time = datetime.now() - timedelta(hours=lookback_hours)

        with self.get_session() as session:
            # Group by price buckets
            results = session.query(
                func.floor(OHLCV.close / 100) * 100,  # $100 buckets
                func.sum(OHLCV.volume).label('total_volume')
            ).filter(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
                OHLCV.timestamp >= start_time
            ).group_by(
                func.floor(OHLCV.close / 100)
            ).all()

            return [
                {'price_level': float(price), 'volume': float(vol)}
                for price, vol in results
            ]

    def get_strategy_performance(
        self,
        strategy_name: str,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get strategy performance metrics.

        Returns:
            Dict with win_rate, avg_pnl, total_signals, etc.
        """
        start_time = datetime.now() - timedelta(days=lookback_days)

        with self.get_session() as session:
            # Get executed signals
            signals = session.query(TradingSignal).filter(
                TradingSignal.strategy_name == strategy_name,
                TradingSignal.executed == True,
                TradingSignal.realized_pnl.isnot(None),
                TradingSignal.timestamp >= start_time
            ).all()

            if not signals:
                return {
                    'total_signals': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                }

            total = len(signals)
            wins = sum(1 for s in signals if s.realized_pnl > 0)
            pnls = [s.realized_pnl for s in signals]

            return {
                'total_signals': total,
                'win_rate': wins / total * 100 if total > 0 else 0,
                'avg_pnl': sum(pnls) / total if total > 0 else 0,
                'total_pnl': sum(pnls),
                'best_trade': max(pnls) if pnls else 0,
                'worst_trade': min(pnls) if pnls else 0
            }

    # ============================================================
    # Utility Functions
    # ============================================================

    def health_check(self) -> bool:
        """Check database connection health."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_table_sizes(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        tables = {
            'ohlcv': OHLCV,
            'trades': Trade,
            'orderbook_snapshots': OrderBookSnapshot,
            'market_metrics': MarketMetrics,
            'agent_decisions': AgentDecision,
            'trading_signals': TradingSignal
        }

        sizes = {}
        with self.get_session() as session:
            for name, model in tables.items():
                count = session.query(func.count(model.id)).scalar()
                sizes[name] = count

        return sizes


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🗄️  Database Manager Demo")
    print("=" * 60)

    # Initialize database
    db = DatabaseManager()

    print("\n📊 Initializing database...")
    db.init_database()

    # Insert sample OHLCV data
    print("\n📈 Inserting sample OHLCV data...")
    sample_ohlcv = [
        {
            'timestamp': datetime.now() - timedelta(minutes=i),
            'exchange': 'binance',
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'open': 45000 + i,
            'high': 45100 + i,
            'low': 44900 + i,
            'close': 45050 + i,
            'volume': 100.5
        }
        for i in range(10)
    ]
    count = db.insert_ohlcv(sample_ohlcv)
    print(f"✓ Inserted {count} OHLCV records")

    # Query data
    print("\n📖 Querying OHLCV data...")
    records = db.get_ohlcv('BTCUSDT', '1m', limit=5)
    print(f"✓ Retrieved {len(records)} records")
    for record in records[:3]:
        print(f"   {record.timestamp}: O={record.open}, C={record.close}, V={record.volume}")

    # Get price stats
    print("\n📊 Price statistics...")
    stats = db.get_price_stats('BTCUSDT', '1m', lookback_days=1)
    print(f"   Mean: ${stats['mean']:,.2f}")
    print(f"   Std: ${stats['std']:,.2f}")
    print(f"   Min: ${stats['min']:,.2f}")
    print(f"   Max: ${stats['max']:,.2f}")
    print(f"   Current: ${stats['current']:,.2f}")

    # Table sizes
    print("\n📦 Table sizes...")
    sizes = db.get_table_sizes()
    for table, count in sizes.items():
        print(f"   {table}: {count:,} rows")

    # Health check
    print("\n🏥 Health check...")
    healthy = db.health_check()
    print(f"   Database: {'✓ Healthy' if healthy else '✗ Unhealthy'}")

    print("\n✅ Database manager demo complete!")
