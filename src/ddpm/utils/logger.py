"""Structured logging setup.

Configures a console logger with timestamps and log levels. The same
configuration is reused across all entry points (train, sample, evaluate)
so logs are consistent.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "ddpm", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    If the logger has already been configured (i.e. it has handlers),
    returns it as-is to avoid duplicate log lines.

    Parameters
    ----------
    name : str, optional
        Logger name. Defaults to ``"ddpm"``.
    level : int, optional
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        Defaults to ``logging.INFO``.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
