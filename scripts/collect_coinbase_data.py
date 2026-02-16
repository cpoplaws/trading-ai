#!/usr/bin/env python3
"""
Coinbase Historical Data Collection Script
Collects OHLCV data from Coinbase and stores in database.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from datetime import datetime, timedelta
import time

from src.data_collection.coinbase_collector import CoinbaseCollector
from src.database.database_manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_historical_data(
    symbols: list,
    days_back: int = 7,
    granularity: str = '3600',  # 1 hour
    save_to_db: bool = True
):
    """
    Collect historical OHLCV data from Coinbase.

    Args:
        symbols: List of trading pairs (e.g., ['BTC-USD', 'ETH-USD'])
        days_back: Number of days to collect
        granularity: Candle size ('60'=1m, '300'=5m, '3600'=1h, '86400'=1d)
        save_to_db: Whether to save to database
    """
    print("=" * 70)
    print("📊 COINBASE HISTORICAL DATA COLLECTION")
    print("=" * 70)

    # Initialize collector
    collector = CoinbaseCollector()

    # Initialize database if saving
    db = None
    if save_to_db:
        db = DatabaseManager()
        print(f"✓ Database connected")

    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    print(f"\n⚙️  Configuration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {start_time.date()} to {end_time.date()} ({days_back} days)")
    print(f"  Granularity: {granularity}s")
    print(f"  Save to DB: {save_to_db}")

    total_candles = 0

    # Collect data for each symbol
    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"Collecting {symbol}...")
        print(f"{'=' * 70}")

        try:
            # Get candles
            candles = collector.get_candles_range(
                symbol=symbol,
                granularity=granularity,
                start=start_time,
                end=end_time
            )

            if not candles:
                logger.warning(f"No candles retrieved for {symbol}")
                continue

            print(f"✓ Retrieved {len(candles):,} candles")

            # Show sample
            print(f"\nSample data (first 3 candles):")
            for candle in candles[:3]:
                print(f"  {candle.timestamp}: "
                      f"O={candle.open:,.2f} H={candle.high:,.2f} "
                      f"L={candle.low:,.2f} C={candle.close:,.2f}")

            # Save to database
            if save_to_db and db:
                count = collector.save_candles_to_db(candles, db)
                print(f"✓ Saved {count:,} candles to database")
                total_candles += count

        except Exception as e:
            logger.error(f"Failed to collect {symbol}: {e}")
            continue

        # Rate limiting
        time.sleep(1)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"📊 COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total candles collected: {total_candles:,}")
    print(f"Symbols processed: {len(symbols)}")

    if save_to_db:
        print(f"\n✅ Data saved to database and ready for analysis!")
    else:
        print(f"\n⚠️  Data not saved (save_to_db=False)")


if __name__ == '__main__':
    # Default configuration
    SYMBOLS = [
        'BTC-USD',
        'ETH-USD',
        'SOL-USD',
        'AVAX-USD'
    ]

    DAYS_BACK = 7  # Last 7 days
    GRANULARITY = '3600'  # 1 hour candles
    SAVE_TO_DB = True

    # Allow command-line override
    if len(sys.argv) > 1:
        # Parse command-line arguments
        if '--symbols' in sys.argv:
            idx = sys.argv.index('--symbols')
            SYMBOLS = sys.argv[idx + 1].split(',')

        if '--days' in sys.argv:
            idx = sys.argv.index('--days')
            DAYS_BACK = int(sys.argv[idx + 1])

        if '--granularity' in sys.argv:
            idx = sys.argv.index('--granularity')
            GRANULARITY = sys.argv[idx + 1]

        if '--no-save' in sys.argv:
            SAVE_TO_DB = False

    # Run collection
    try:
        collect_historical_data(
            symbols=SYMBOLS,
            days_back=DAYS_BACK,
            granularity=GRANULARITY,
            save_to_db=SAVE_TO_DB
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
