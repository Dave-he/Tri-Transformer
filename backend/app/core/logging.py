import logging
import sys
import time
from typing import Callable

from fastapi import Request, Response


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("tri-transformer")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    logger.info(
        '{"method": "%s", "path": "%s", "status_code": %d, "duration_ms": %s}',
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response
