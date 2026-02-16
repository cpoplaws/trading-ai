#!/usr/bin/env python3
"""
Example: Autonomous Agent with Database Integration

This example shows how to connect the autonomous trading agent
to the production PostgreSQL database for data persistence.
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autonomous_agent.trading_agent import AutonomousTradingAgent, AgentConfig
from database.config import DatabaseConfig
from database.models import User, Portfolio, Trade, OrderSide, UserRole
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseIntegratedAgent(AutonomousTradingAgent):
    """
    Enhanced autonomous agent with database persistence.
    """

    def __init__(self, config: AgentConfig, db_config: DatabaseConfig):
        super().__init__(config)
        self.db = db_config
        self.user_id = None
        self.portfolio_id = None

    async def start(self):
        """Start agent with database setup."""
        # Initialize database entities
        await self._setup_database_entities()

        # Start normal agent operation
        await super().start()

    async def _setup_database_entities(self):
        """Create or get user and portfolio from database."""
        logger.info("Setting up database entities...")

        with self.db.get_session() as session:
            # Get or create agent user
            user = session.query(User).filter_by(username='autonomous_agent').first()

            if not user:
                logger.info("Creating new agent user...")
                user = User(
                    username='autonomous_agent',
                    email='agent@trading-ai.local',
                    password_hash='automated_agent',
                    role=UserRole.TRADER
                )
                session.add(user)
                session.flush()

            self.user_id = user.id
            logger.info(f"Agent user ID: {self.user_id}")

            # Get or create portfolio
            portfolio = session.query(Portfolio).filter_by(
                user_id=user.id,
                name='Autonomous Agent Portfolio'
            ).first()

            if not portfolio:
                logger.info("Creating new portfolio...")
                portfolio = Portfolio(
                    user_id=user.id,
                    name='Autonomous Agent Portfolio',
                    total_value_usd=self.config.initial_capital,
                    cash_balance_usd=self.config.initial_capital,
                    total_pnl=0.0,
                    is_paper=self.config.paper_trading
                )
                session.add(portfolio)
                session.flush()

            self.portfolio_id = portfolio.id
            logger.info(f"Portfolio ID: {self.portfolio_id}")

            session.commit()

    async def _execute_trade(self, signal: dict) -> dict:
        """Execute trade and save to database."""
        # Execute trade using parent method
        trade_result = await super()._execute_trade(signal)

        # Save to database
        if trade_result and self.portfolio_id:
            await self._save_trade_to_db(trade_result)

        return trade_result

    async def _save_trade_to_db(self, trade_result: dict):
        """Save executed trade to database."""
        try:
            with self.db.get_session() as session:
                trade = Trade(
                    portfolio_id=self.portfolio_id,
                    symbol=trade_result.get('symbol', 'UNKNOWN'),
                    exchange='simulated',
                    side=OrderSide.BUY if trade_result.get('action') == 'BUY' else OrderSide.SELL,
                    quantity=trade_result.get('size', 0.0),
                    price=trade_result.get('price', 0.0),
                    value=trade_result.get('value', 0.0),
                    executed_at=datetime.now(),
                    strategy_name=trade_result.get('strategy', 'unknown')
                )
                session.add(trade)

                # Update portfolio
                portfolio = session.query(Portfolio).filter_by(id=self.portfolio_id).first()
                if portfolio:
                    portfolio.total_value_usd = self.portfolio_value
                    portfolio.total_pnl = self.total_pnl

                session.commit()
                logger.info(f"✅ Trade saved to database: {trade.symbol} {trade.side.value}")

        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")

    async def stop(self):
        """Stop agent and save final state."""
        # Save final portfolio state
        if self.portfolio_id:
            with self.db.get_session() as session:
                portfolio = session.query(Portfolio).filter_by(id=self.portfolio_id).first()
                if portfolio:
                    portfolio.total_value_usd = self.portfolio_value
                    portfolio.cash_balance_usd = self.portfolio_value - sum(
                        p.get('value', 0) for p in self.positions.values()
                    )
                    portfolio.total_pnl = self.total_pnl
                    session.commit()
                    logger.info(f"✅ Final portfolio state saved")

        await super().stop()


async def main():
    """Main entry point with database integration."""
    print("=" * 60)
    print("Autonomous Trading Agent with Database Integration")
    print("=" * 60)

    # Database configuration
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://trader:changeme@localhost:5432/trading_ai'
    )

    print(f"\nDatabase: {database_url}")

    # Initialize database connection
    db = DatabaseConfig(database_url=database_url, echo=False)

    # Test database connection
    if not db.health_check():
        print("❌ Database connection failed!")
        return

    print("✅ Database connected")

    # Agent configuration
    config = AgentConfig(
        initial_capital=10000.0,
        paper_trading=True,
        check_interval_seconds=5,
        max_daily_loss=500.0,
        send_alerts=False  # Disable alerts for demo
    )

    # Create and start agent with database
    agent = DatabaseIntegratedAgent(config, db)

    try:
        print("\n🤖 Starting agent with database persistence...")
        print("   Press Ctrl+C to stop\n")

        # Run for 60 seconds as demo
        await asyncio.wait_for(agent.start(), timeout=60.0)

    except asyncio.TimeoutError:
        print("\n⏱️  Demo timeout reached")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    finally:
        await agent.stop()

    # Show final statistics
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)

    with db.get_session() as session:
        # Get portfolio
        portfolio = session.query(Portfolio).filter_by(id=agent.portfolio_id).first()
        if portfolio:
            print(f"Portfolio Value: ${portfolio.total_value_usd:,.2f}")
            print(f"Total P&L: ${portfolio.total_pnl:+,.2f}")

        # Get trade count
        trade_count = session.query(Trade).filter_by(portfolio_id=agent.portfolio_id).count()
        print(f"Trades Executed: {trade_count}")

    print("\n✅ Agent stopped - all data persisted to database")


if __name__ == '__main__':
    asyncio.run(main())
