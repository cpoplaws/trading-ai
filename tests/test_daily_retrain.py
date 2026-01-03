import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from execution.daily_retrain import _resolve_start_date, archive_model  # noqa: E402


def test_resolve_start_date_uses_rolling_window():
    fixed_now = datetime(2024, 1, 10)
    start_date = _resolve_start_date(None, 30, current_time=fixed_now)
    assert start_date == "2023-12-11"


def test_archive_model_creates_dated_copy(tmp_path: Path):
    model_dir = tmp_path / "models"
    archive_dir = tmp_path / "archive"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model_TEST.joblib"
    model_path.write_text("dummy-model")

    run_date = datetime(2024, 2, 3)
    archived_path = archive_model(
        str(model_path),
        "TEST",
        run_date=run_date,
        archive_dir=str(archive_dir),
    )

    assert archived_path is not None
    assert os.path.exists(archived_path)
    assert archived_path.endswith("model_TEST_20240203.joblib")
    with open(archived_path) as f:
        assert f.read() == "dummy-model"
