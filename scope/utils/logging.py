"""Logging configuration for SCOPE."""

import logging
import sys
from typing import Optional


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for SCOPE.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("scope")

    # Set level
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    if verbose:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter("%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    """Get the SCOPE logger instance.

    Returns:
        Logger instance
    """
    return logging.getLogger("scope")
