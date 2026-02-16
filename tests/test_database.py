"""
Comprehensive tests for database operations.
"""
import pytest
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.config import DatabaseConfig
from database.models import (
    User, Portfolio, Trade, Position,
    PriceData, Alert, UserRole, OrderSide, AlertSeverity, Strategy
)


@pytest.fixture
def test_db():
    """Create test database connection."""
    database_url = os.getenv(
        'TEST_DATABASE_URL',
        'postgresql://trader:changeme@localhost:5432/trading_ai_test'
    )
    db = DatabaseConfig(database_url=database_url, echo=False)

    # Create tables
    db.create_tables()

    yield db

    # Cleanup - drop all tables after tests
    db.Base.metadata.drop_all(db.engine)


@pytest.fixture
def test_user(test_db):
    """Create a test user."""
    with test_db.get_session() as session:
        user = User(
            username='test_trader',
            email='test@example.com',
            password_hash='test_hash',
            role=UserRole.TRADER
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.id


@pytest.fixture
def test_portfolio(test_db, test_user):
    """Create a test portfolio."""
    with test_db.get_session() as session:
        portfolio = Portfolio(
            user_id=test_user,
            name='Test Portfolio',
            total_value_usd=10000.0,
            cash_balance_usd=10000.0,
            total_pnl=0.0,
            is_paper=True
        )
        session.add(portfolio)
        session.commit()
        session.refresh(portfolio)
        return portfolio.id


class TestDatabaseConnection:
    """Tests for database connection and setup."""

    def test_connection(self, test_db):
        """Test database connection."""
        assert test_db.health_check() is True

    def test_table_creation(self, test_db):
        """Test that all tables are created."""
        inspector = test_db.engine.dialect.get_inspector(test_db.engine)
        tables = inspector.get_table_names()

        expected_tables = [
            'users', 'portfolios', 'trades', 'positions',
            'price_data', 'alerts', 'strategies'
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"


class TestUserOperations:
    """Tests for User model operations."""

    def test_create_user(self, test_db):
        """Test creating a user."""
        with test_db.get_session() as session:
            user = User(
                username='new_user',
                email='newuser@example.com',
                password_hash='hash123',
                role=UserRole.TRADER
            )
            session.add(user)
            session.commit()

            assert user.id is not None
            assert user.username == 'new_user'
            assert user.role == UserRole.TRADER

    def test_query_user(self, test_db, test_user):
        """Test querying a user."""
        with test_db.get_session() as session:
            user = session.query(User).filter_by(id=test_user).first()
            assert user is not None
            assert user.username == 'test_trader'

    def test_update_user(self, test_db, test_user):
        """Test updating a user."""
        with test_db.get_session() as session:
            user = session.query(User).filter_by(id=test_user).first()
            user.email = 'updated@example.com'
            session.commit()

        with test_db.get_session() as session:
            user = session.query(User).filter_by(id=test_user).first()
            assert user.email == 'updated@example.com'


class TestPortfolioOperations:
    """Tests for Portfolio model operations."""

    def test_create_portfolio(self, test_db, test_user):
        """Test creating a portfolio."""
        with test_db.get_session() as session:
            portfolio = Portfolio(
                user_id=test_user,
                name='Test Portfolio',
                total_value_usd=10000.0,
                cash_balance_usd=10000.0,
                total_pnl=0.0,
                is_paper=True
            )
            session.add(portfolio)
            session.commit()

            assert portfolio.id is not None
            assert portfolio.total_value_usd == 10000.0

    def test_portfolio_user_relationship(self, test_db, test_user, test_portfolio):
        """Test relationship between portfolio and user."""
        with test_db.get_session() as session:
            portfolio = session.query(Portfolio).filter_by(id=test_portfolio).first()
            assert portfolio.user_id == test_user
            assert portfolio.user.username == 'test_trader'

    def test_update_portfolio_value(self, test_db, test_portfolio):
        """Test updating portfolio value."""
        with test_db.get_session() as session:
            portfolio = session.query(Portfolio).filter_by(id=test_portfolio).first()
            portfolio.total_value_usd = 12000.0
            portfolio.total_pnl = 2000.0
            session.commit()

        with test_db.get_session() as session:
            portfolio = session.query(Portfolio).filter_by(id=test_portfolio).first()
            assert portfolio.total_value_usd == 12000.0
            assert portfolio.total_pnl == 2000.0


class TestTradeOperations:
    """Tests for Trade model operations."""

    def test_create_trade(self, test_db, test_portfolio):
        """Test creating a trade."""
        with test_db.get_session() as session:
            trade = Trade(
                portfolio_id=test_portfolio,
                symbol='BTC-USD',
                exchange='coinbase',
                side=OrderSide.BUY,
                quantity=0.1,
                price=40000.0,
                value=4000.0,
                executed_at=datetime.now()
            )
            session.add(trade)
            session.commit()

            assert trade.id is not None
            assert trade.symbol == 'BTC-USD'
            assert trade.side == OrderSide.BUY

    def test_trade_portfolio_relationship(self, test_db, test_portfolio):
        """Test relationship between trade and portfolio."""
        with test_db.get_session() as session:
            trade = Trade(
                portfolio_id=test_portfolio,
                symbol='ETH-USD',
                exchange='coinbase',
                side=OrderSide.SELL,
                quantity=1.0,
                price=2500.0,
                value=2500.0,
                executed_at=datetime.now()
            )
            session.add(trade)
            session.commit()

        with test_db.get_session() as session:
            portfolio = session.query(Portfolio).filter_by(id=test_portfolio).first()
            assert len(portfolio.trades) > 0
            assert portfolio.trades[0].symbol in ['BTC-USD', 'ETH-USD']

    def test_query_trades_by_symbol(self, test_db, test_portfolio):
        """Test querying trades by symbol."""
        with test_db.get_session() as session:
            # Create multiple trades
            for i in range(3):
                trade = Trade(
                    portfolio_id=test_portfolio,
                    symbol='BTC-USD',
                    exchange='coinbase',
                    side=OrderSide.BUY,
                    quantity=0.1,
                    price=40000.0 + i * 100,
                    value=4000.0,
                    executed_at=datetime.now()
                )
                session.add(trade)
            session.commit()

        with test_db.get_session() as session:
            trades = session.query(Trade).filter_by(
                portfolio_id=test_portfolio,
                symbol='BTC-USD'
            ).all()
            assert len(trades) >= 3


class TestPositionOperations:
    """Tests for Position model operations."""

    def test_create_position(self, test_db, test_portfolio):
        """Test creating a position."""
        with test_db.get_session() as session:
            position = Position(
                portfolio_id=test_portfolio,
                symbol='BTC-USD',
                quantity=0.5,
                entry_price=40000.0,
                current_price=42000.0,
                unrealized_pnl=1000.0
            )
            session.add(position)
            session.commit()

            assert position.id is not None
            assert position.quantity == 0.5

    def test_update_position(self, test_db, test_portfolio):
        """Test updating a position."""
        with test_db.get_session() as session:
            position = Position(
                portfolio_id=test_portfolio,
                symbol='ETH-USD',
                quantity=2.0,
                entry_price=2500.0,
                current_price=2500.0,
                unrealized_pnl=0.0
            )
            session.add(position)
            session.commit()
            position_id = position.id

        with test_db.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            position.current_price = 2600.0
            position.unrealized_pnl = 200.0
            session.commit()

        with test_db.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            assert position.current_price == 2600.0
            assert position.unrealized_pnl == 200.0


class TestAlertOperations:
    """Tests for Alert model operations."""

    def test_create_alert(self, test_db, test_user):
        """Test creating an alert."""
        with test_db.get_session() as session:
            alert = Alert(
                user_id=test_user,
                severity=AlertSeverity.INFO,
                message='Trade executed successfully',
                is_read=False
            )
            session.add(alert)
            session.commit()

            assert alert.id is not None
            assert alert.severity == AlertSeverity.INFO

    def test_update_alert_status(self, test_db, test_user):
        """Test updating alert status."""
        with test_db.get_session() as session:
            alert = Alert(
                user_id=test_user,
                severity=AlertSeverity.WARNING,
                message='Max drawdown exceeded',
                is_read=False
            )
            session.add(alert)
            session.commit()
            alert_id = alert.id

        with test_db.get_session() as session:
            alert = session.query(Alert).filter_by(id=alert_id).first()
            alert.is_read = True
            session.commit()

        with test_db.get_session() as session:
            alert = session.query(Alert).filter_by(id=alert_id).first()
            assert alert.is_read is True


class TestStrategyOperations:
    """Tests for Strategy model operations."""

    def test_create_strategy(self, test_db, test_user):
        """Test creating a strategy configuration."""
        with test_db.get_session() as session:
            strategy = Strategy(
                user_id=test_user,
                name='DCA Bot Strategy',
                strategy_type='dca',
                config={'base_amount': 100.0, 'frequency': 'daily'},
                is_active=True
            )
            session.add(strategy)
            session.commit()

            assert strategy.id is not None
            assert strategy.name == 'DCA Bot Strategy'
            assert strategy.is_active is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
