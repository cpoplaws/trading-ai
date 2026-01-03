import logging
import schedule
import sys
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
src_path = tests_dir.parent / "src"
sys.path.append(str(src_path))

from execution import scheduler


def test_configure_schedule_sets_daily_jobs():
    schedule.clear()
    scheduler.configure_schedule("00:00", "00:05")

    retrain_jobs = schedule.get_jobs("scheduler_retrain")
    strategy_jobs = schedule.get_jobs("scheduler_strategy")

    assert retrain_jobs
    assert strategy_jobs
    assert retrain_jobs[0].at_time.strftime("%H:%M") == "00:00"
    assert strategy_jobs[0].at_time.strftime("%H:%M") == "00:05"

    schedule.clear()


def test_run_retrain_logs_start_and_end(caplog):
    caplog.set_level(logging.INFO, logger="scheduler")

    called = {"ran": False}

    def fake_pipeline():
        called["ran"] = True
        return True

    result = scheduler.run_retrain(fake_pipeline)

    assert result is True
    assert called["ran"] is True
    assert "Retrain job started" in caplog.text
    assert "Retrain job ended" in caplog.text


def test_run_strategy_evaluation_logs_start_and_end(caplog):
    caplog.set_level(logging.INFO, logger="scheduler")

    called = {"ran": False}

    def fake_evaluator():
        called["ran"] = True
        return True

    result = scheduler.run_strategy_evaluation(fake_evaluator)

    assert result is True
    assert called["ran"] is True
    assert "Strategy evaluation job started" in caplog.text
    assert "Strategy evaluation job ended" in caplog.text
