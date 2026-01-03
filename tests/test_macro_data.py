"""
Tests for daily macroeconomic data ingestion.
"""
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_ingestion.macro_data import MacroDataFetcher  # noqa: E402


def test_build_daily_macro_dataset_simulated(tmp_path):
    """Ensure daily macro dataset is built and has no missing values."""
    fetcher = MacroDataFetcher(api_key=None)
    start = datetime.now() - timedelta(days=60)
    end = datetime.now()
    save_path = tmp_path / "macro_daily.csv"

    df = fetcher.build_daily_macro_dataset(
        start_date=start, end_date=end, save_path=str(save_path)
    )

    assert not df.empty
    assert save_path.exists()
    for column in ["fed_funds_rate", "cpi_inflation_rate", "unemployment_rate"]:
        assert column in df.columns
        assert not df[column].isna().any()
