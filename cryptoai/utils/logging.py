"""Logging configuration for CryptoAI."""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = True,
    rotation: str = "1 day",
    retention: str = "30 days",
) -> None:
    """
    Configure logging for CryptoAI.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        structured: Whether to use structured JSON logging
        rotation: Log rotation interval
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()

    # Console handler with rich formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if structured:
            logger.add(
                str(log_file),
                format="{message}",
                level=level,
                rotation=rotation,
                retention=retention,
                serialize=True,
                backtrace=True,
                diagnose=True,
            )
        else:
            logger.add(
                str(log_file),
                format=log_format,
                level=level,
                rotation=rotation,
                retention=retention,
                backtrace=True,
                diagnose=True,
            )

    logger.info(f"Logging configured with level={level}")


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)
