"""
Daily scheduler for retraining and strategy evaluation jobs.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import schedule

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from execution.daily_retrain import daily_pipeline
from strategy.simple_strategy import analyze_signals
from utils.logger import setup_logger

logger = setup_logger("scheduler", "INFO")

RETRAIN_TIME = os.getenv("RETRAIN_TIME", "07:00")
STRATEGY_EVAL_TIME = os.getenv("STRATEGY_EVAL_TIME", "07:30")
SIGNALS_DIR = os.getenv("SIGNALS_DIR", "./signals")


def run_retrain(pipeline_fn: Callable[[], bool] = daily_pipeline) -> bool:
    """
    Execute the retraining pipeline and log lifecycle events.
    """
    start_time = datetime.now()
    logger.info("Retrain job started")

    try:
        success = pipeline_fn()
        if success:
            logger.info("Retrain job completed successfully")
        else:
            logger.warning("Retrain job completed with issues")
        return success
    except Exception:
        logger.exception("Retrain job failed with an exception")
        return False
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrain job ended (duration: {duration:.2f}s)")


def _evaluate_existing_signals(signals_dir: str = SIGNALS_DIR) -> bool:
    """
    Evaluate existing strategy signals.
    """
    signals_path = Path(signals_dir)
    if not signals_path.exists():
        logger.warning("Signals directory not found; skipping evaluation")
        return False

    signal_files = list(signals_path.glob("*_signals.csv"))
    if not signal_files:
        logger.warning("No signal files found for evaluation")
        return False

    processed_any = False
    for signal_file in signal_files:
        try:
            analysis = analyze_signals(str(signal_file))
            if isinstance(analysis, dict) and analysis.get("error"):
                logger.warning(
                    "Strategy evaluation for %s returned error: %s",
                    signal_file.name,
                    analysis.get("error"),
                )
            else:
                processed_any = True
                logger.info("Strategy evaluation completed for %s", signal_file.name)
            logger.debug("Analysis details for %s: %s", signal_file.name, analysis)
        except Exception:
            logger.exception("Error evaluating signal file %s", signal_file.name)

    return processed_any


def run_strategy_evaluation(
    evaluation_fn: Optional[Callable[[str], bool]] = None,
    signals_dir: str = SIGNALS_DIR,
) -> bool:
    """
    Execute strategy evaluation and log lifecycle events.
    """
    start_time = datetime.now()
    logger.info("Strategy evaluation job started")

    evaluation_callable = evaluation_fn or _evaluate_existing_signals

    try:
        success = evaluation_callable(signals_dir)
        if success:
            logger.info("Strategy evaluation completed successfully")
        else:
            logger.warning("Strategy evaluation completed with issues")
        return success
    except Exception:
        logger.exception("Strategy evaluation failed with an exception")
        return False
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Strategy evaluation job ended (duration: {duration:.2f}s)")


def configure_schedule(
    retrain_time: str = RETRAIN_TIME, strategy_time: str = STRATEGY_EVAL_TIME
):
    """
    Configure scheduled jobs for retraining and strategy evaluation.
    """
    validated_retrain = _validate_time(retrain_time, "retrain_time")
    validated_strategy = _validate_time(strategy_time, "strategy_time")

    schedule.clear("scheduler_retrain")
    schedule.clear("scheduler_strategy")

    schedule.every().day.at(validated_retrain).do(run_retrain).tag("scheduler_retrain")
    schedule.every().day.at(validated_strategy).do(run_strategy_evaluation).tag(
        "scheduler_strategy"
    )

    logger.info(
        f"Scheduler configured with retrain_time={validated_retrain} and strategy_time={validated_strategy}"
    )


def _validate_time(time_str: str, label: str) -> str:
    """
    Validate HH:MM formatted time strings before scheduling/logging.
    """
    try:
        datetime.strptime(time_str, "%H:%M")
        return time_str
    except ValueError as exc:
        logger.error("Invalid %s format '%s'; expected HH:MM", label, time_str)
        raise exc


def main():
    configure_schedule()
    logger.info("Scheduler started; entering run loop")

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except Exception:
            logger.exception("Scheduler loop encountered an error")


if __name__ == "__main__":
    main()
