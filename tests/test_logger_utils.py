from pathlib import Path
from logging.handlers import RotatingFileHandler

from utils.logger import setup_logger


def test_setup_logger_rotates_files(tmp_path: Path):
    log_file = tmp_path / "rotation.log"
    logger = setup_logger(
        name="test_rotation",
        log_level="INFO",
        log_file=str(log_file),
        max_bytes=200,
        backup_count=1,
    )

    try:
        for i in range(20):
            logger.info("log-entry-%s-%s", i, "x" * 20)
        logger.info("format-check")

        for handler in logger.handlers:
            handler.flush()

        rotating_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert rotating_handlers, "Expected a RotatingFileHandler to be attached"

        rotated_path = Path(f"{log_file}.1")
        assert rotated_path.exists()

        log_contents = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        rotated_contents = rotated_path.read_text(encoding="utf-8") if rotated_path.exists() else ""

        assert "format-check" in log_contents or "format-check" in rotated_contents
        assert logger.name in log_contents or logger.name in rotated_contents
        assert "INFO" in log_contents or "INFO" in rotated_contents
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
