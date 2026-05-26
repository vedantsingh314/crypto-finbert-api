"""
logger.py — Structured logging setup.
All modules call get_logger(__name__) to get a namespaced logger.
In production, swap the StreamHandler for a log aggregator (Datadog, Loki, etc.)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger           # Already configured (avoid duplicate handlers)

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
