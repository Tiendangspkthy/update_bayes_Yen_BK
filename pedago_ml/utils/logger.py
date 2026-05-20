import logging as logging_logger
import sys

from loguru import logger as loguru_logger

from pedago_ml.utils.singleton import SingletonMeta


class Logger(metaclass=SingletonMeta):
    """Singleton Logger class that provides a consistent logging interface
    using Loguru or standard logging."""

    def __init__(self, is_loguru=True, update=False):
        self.loguru_logger = loguru_logger
        self.loguru_logger.remove()
        self.loguru_logger.add(
            sys.stderr,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}"
            "</green> | <level>{level: <8}</level>| <cyan>{name}"
            "</cyan>:<cyan>{function}</cyan>:<cyan>{line"
            "}</cyan>- <level>{message}</level>",
            level="INFO",
        )
        self.logging_logger = logging_logger
        self.logging_logger.basicConfig(level=logging_logger.INFO)

        self.is_loguru = is_loguru

    def get_logger(self):
        if self.is_loguru:
            return self.loguru_logger
        else:
            return self.logging_logger


logger = Logger().get_logger()


def add_pipeline_file_log(log_path) -> int:
    """Attach a loguru file sink; returns sink id for later removal."""
    from pathlib import Path

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return logger.add(
        str(log_path),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        level="INFO",
        encoding="utf-8",
    )


def remove_pipeline_file_log(sink_id: int) -> None:
    """Remove a file sink previously added by add_pipeline_file_log."""
    logger.remove(sink_id)
