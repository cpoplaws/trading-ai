"""
Database Models
SQLAlchemy ORM models for trading system.

Tables:
- users: User accounts and authentication
- portfolios: User portfolios
- trades: Trade history
- orders: Order book
- prices: Historical price data (TimescaleDB)
- ml_predictions: ML model predictions
- alerts: System alerts
- strategies: Strategy configurations
"""
import logging
from typing import Optional
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Enum as SQLEnum, Text, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from enum import Enum

logger = logging.getLogger(__name__)

Base = declarative_base()


# Enums
class UserRole(str, Enum):
    """User roles."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class StrategyType(str, Enum):
    """Strategy type."""
    GRID_TRADING = "grid_trading"
    DCA = "dca"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_MAKING = "market_making"
    WHALE_FOLLOWING = "whale_following"
    LIQUIDATION_HUNTING = "liquidation_hunting"
    YIELD_OPTIMIZATION = "yield_optimization"


class AlertSeverity(str, Enum):
    """Alert severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Models
class User(Base):
    """User account."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.TRADER, nullable=False)

    # API keys (encrypted)
    api_key = Column(String(255), unique=True)
    api_secret = Column(Text)  # Encrypted

    # Settings
    two_factor_enabled = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user")
    strategies = relationship("Strategy", back_populates="user")
    alerts = relationship("Alert", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Portfolio(Base):
    """User portfolio."""
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)

    # Balance
    total_value_usd = Column(Float, default=0.0)
    cash_balance_usd = Column(Float, default=0.0)

    # Performance
    total_pnl = Column(Float, default=0.0)
    total_pnl_percent = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)

    # Risk metrics
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)

    # Settings
    is_paper = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio")

    __table_args__ = (
        Index('idx_portfolio_user', 'user_id'),
    )

    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}', value=${self.total_value_usd:.2f})>"


class Position(Base):
    """Current position in an asset."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)

    # Asset
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50))

    # Position details
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)

    # P&L
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_percent = Column(Float, default=0.0)

    # Metadata
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")

    __table_args__ = (
        Index('idx_position_portfolio', 'portfolio_id'),
        Index('idx_position_symbol', 'symbol'),
    )

    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', qty={self.quantity}, entry=${self.entry_price})>"


class Order(Base):
    """Order (pending or executed)."""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))

    # External IDs
    exchange_order_id = Column(String(100), unique=True)
    client_order_id = Column(String(100), unique=True)

    # Order details
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False)

    # Pricing
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0.0)
    price = Column(Float)  # Limit price (null for market orders)
    average_fill_price = Column(Float)

    # Fees
    fee = Column(Float, default=0.0)
    fee_currency = Column(String(10))

    # Strategy
    strategy_id = Column(Integer, ForeignKey("strategies.id"))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="orders")
    strategy = relationship("Strategy", back_populates="orders")
    trades = relationship("Trade", back_populates="order")

    __table_args__ = (
        Index('idx_order_user', 'user_id'),
        Index('idx_order_symbol', 'symbol'),
        Index('idx_order_status', 'status'),
        Index('idx_order_created', 'created_at'),
    )

    def __repr__(self):
        return f"<Order(id={self.id}, symbol='{self.symbol}', side='{self.side}', status='{self.status}')>"


class Trade(Base):
    """Executed trade."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    order_id = Column(Integer, ForeignKey("orders.id"))

    # External IDs
    exchange_trade_id = Column(String(100), unique=True)

    # Trade details
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)

    # Execution
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    value = Column(Float, nullable=False)

    # Fees
    fee = Column(Float, default=0.0)
    fee_currency = Column(String(10))

    # P&L (for closed trades)
    pnl = Column(Float)
    pnl_percent = Column(Float)

    # Metadata
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="trades")
    order = relationship("Order", back_populates="trades")

    __table_args__ = (
        Index('idx_trade_portfolio', 'portfolio_id'),
        Index('idx_trade_symbol', 'symbol'),
        Index('idx_trade_executed', 'executed_at'),
    )

    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side}', qty={self.quantity}, price=${self.price})>"


class PriceData(Base):
    """Historical price data (TimescaleDB hypertable)."""
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Asset
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)

    # OHLCV
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Additional metrics
    vwap = Column(Float)  # Volume-weighted average price
    num_trades = Column(Integer)

    __table_args__ = (
        Index('idx_price_symbol_time', 'symbol', 'timestamp'),
        Index('idx_price_exchange_time', 'exchange', 'timestamp'),
    )

    def __repr__(self):
        return f"<PriceData(symbol='{self.symbol}', time={self.timestamp}, close=${self.close})>"


class MLPrediction(Base):
    """ML model predictions."""
    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Model info
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))

    # Prediction
    symbol = Column(String(20), nullable=False)
    predicted_price = Column(Float)
    predicted_direction = Column(String(10))  # 'up', 'down', 'neutral'
    confidence = Column(Float)

    # Features used
    feature_importance = Column(JSON)

    # Metadata
    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    prediction_horizon = Column(Integer)  # Minutes ahead

    # Validation (backfill after prediction period)
    actual_price = Column(Float)
    prediction_error = Column(Float)
    was_correct = Column(Boolean)

    __table_args__ = (
        Index('idx_prediction_symbol_time', 'symbol', 'predicted_at'),
        Index('idx_prediction_model', 'model_name'),
    )

    def __repr__(self):
        return f"<MLPrediction(model='{self.model_name}', symbol='{self.symbol}', price=${self.predicted_price})>"


class Strategy(Base):
    """Strategy configuration and state."""
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Strategy info
    name = Column(String(100), nullable=False)
    strategy_type = Column(SQLEnum(StrategyType), nullable=False)
    description = Column(Text)

    # Configuration (JSON)
    config = Column(JSON, nullable=False)

    # State
    is_active = Column(Boolean, default=True)
    is_backtest = Column(Boolean, default=False)

    # Performance
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    num_trades = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_signal_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="strategies")
    orders = relationship("Order", back_populates="strategy")

    __table_args__ = (
        Index('idx_strategy_user', 'user_id'),
        Index('idx_strategy_type', 'strategy_type'),
    )

    def __repr__(self):
        return f"<Strategy(id={self.id}, name='{self.name}', type='{self.strategy_type}')>"


class Alert(Base):
    """System alert."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    # Alert details
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(SQLEnum(AlertSeverity), default=AlertSeverity.INFO, nullable=False)

    # Categorization
    category = Column(String(50))  # 'price', 'trade', 'system', 'ml', 'pattern'

    # Related entities
    symbol = Column(String(20))
    order_id = Column(Integer, ForeignKey("orders.id"))
    strategy_id = Column(Integer, ForeignKey("strategies.id"))

    # Status
    is_read = Column(Boolean, default=False)
    is_sent = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="alerts")

    __table_args__ = (
        Index('idx_alert_user_created', 'user_id', 'created_at'),
        Index('idx_alert_severity', 'severity'),
    )

    def __repr__(self):
        return f"<Alert(title='{self.title}', severity='{self.severity}')>"


class APIKey(Base):
    """API key for programmatic access."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Key
    key = Column(String(100), unique=True, nullable=False, index=True)
    key_hash = Column(String(255), nullable=False)

    # Metadata
    name = Column(String(100))
    description = Column(Text)

    # Permissions
    permissions = Column(JSON)  # List of allowed operations

    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)

    # Status
    is_active = Column(Boolean, default=True)

    # Usage
    last_used_at = Column(DateTime)
    total_requests = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)

    def __repr__(self):
        return f"<APIKey(key='{self.key[:8]}...', user_id={self.user_id})>"


# Utility functions
def create_all_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(engine)
    logger.info("All database tables created")


def drop_all_tables(engine):
    """Drop all database tables (DANGEROUS!)."""
    Base.metadata.drop_all(engine)
    logger.warning("All database tables dropped")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("📊 Database Models")
    print("=" * 60)

    print("\nTables:")
    for table_name in Base.metadata.tables.keys():
        table = Base.metadata.tables[table_name]
        print(f"\n{table_name}:")
        print(f"  Columns: {len(table.columns)}")
        print(f"  Indexes: {len(table.indexes)}")
        print(f"  Foreign Keys: {len(table.foreign_keys)}")

    print(f"\n✅ Total: {len(Base.metadata.tables)} tables defined")
