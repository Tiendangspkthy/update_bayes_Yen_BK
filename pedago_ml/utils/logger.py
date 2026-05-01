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
