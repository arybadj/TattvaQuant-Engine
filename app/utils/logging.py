import json
import logging
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class AppLogger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, event: str, **fields) -> None:
        self._logger.info(self._serialize(event, **fields))

    def warning(self, event: str, **fields) -> None:
        self._logger.warning(self._serialize(event, **fields))

    def error(self, event: str, **fields) -> None:
        self._logger.error(self._serialize(event, **fields))

    def exception(self, event: str, **fields) -> None:
        self._logger.exception(self._serialize(event, **fields))

    @staticmethod
    def _serialize(event: str, **fields) -> str:
        payload = {"event": event, "request_id": request_id_context.get(), **fields}
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter("%(message)s"))
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def get_logger(name: str) -> AppLogger:
    return AppLogger(logging.getLogger(name))

