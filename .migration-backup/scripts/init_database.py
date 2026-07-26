#!/usr/bin/env python3
"""
Initialize database schema and create tables.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.config import DatabaseConfig
from database.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database(database_url: str = None):
    """Initialize database with schema."""
    if database_url is None:
        database_url = os.getenv('DATABASE_URL', 'postgresql://trader:changeme@localhost:5432/trading_ai')

    logger.info(f"Initializing database: {database_url}")

    try:
        # Create database config
        db = DatabaseConfig(database_url=database_url, echo=True)

        # Create all tables
        logger.info("Creating database tables...")
        db.create_tables()

        # Setup TimescaleDB hypertables if using PostgreSQL
        if 'postgresql' in database_url:
            logger.info("Setting up TimescaleDB...")
            db.setup_timescaledb()

        # Verify tables created
        with db.get_session() as session:
            # Check if tables exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()

            logger.info(f"✅ Created {len(tables)} tables:")
            for table in sorted(tables):
                logger.info(f"   - {table}")

        logger.info("✅ Database initialization complete!")
        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Initialize Trading AI database')
    parser.add_argument('--url', help='Database URL (default: from DATABASE_URL env var)')
    args = parser.parse_args()

    success = init_database(args.url)
    sys.exit(0 if success else 1)
