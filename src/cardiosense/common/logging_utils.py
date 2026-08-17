"""Logging setup shared by scripts and notebooks.

``print`` is fine for a scratch notebook but useless when a Colab session dies
at epoch 27 and you need to know what the learning rate was. Every module uses::

    from cardiosense.common.logging_utils import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .paths import PATHS, ensure_dir

__all__ = ["get_logger", "configure_logging", "log_section"]

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    to_file: bool = True,
    filename: str = "cardiosense.log",
    log_dir: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root ``cardiosense`` logger exactly once.

    Args:
        level: Logging level name or constant.
        to_file: Also write to ``results/logs/<filename>``.
        filename: Log file name.
        log_dir: Override the log directory.
        force: Reconfigure even if already configured (handy in notebooks where
            a cell is re-run).

    Returns:
        The configured ``cardiosense`` logger.
    """
    global _CONFIGURED
    logger = logging.getLogger("cardiosense")

    if _CONFIGURED and not force:
        return logger

    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if to_file:
        directory = ensure_dir(Path(log_dir) if log_dir else PATHS.results / "logs")
        file_handler = logging.FileHandler(directory / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "cardiosense") -> logging.Logger:
    """Return a child logger, configuring the root handler on first use."""
    configure_logging()
    if name == "cardiosense" or name.startswith("cardiosense."):
        return logging.getLogger(name)
    return logging.getLogger(f"cardiosense.{name}")


def log_section(logger: logging.Logger, title: str, width: int = 78) -> None:
    """Print a visually distinct section banner into the log."""
    logger.info("=" * width)
    logger.info(title.upper())
    logger.info("=" * width)
