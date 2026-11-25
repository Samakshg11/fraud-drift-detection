"""Logging helpers.

One consistent logger setup for all scripts. Repeated calls with the same name
do not add duplicate handlers, which is a common cause of duplicated output in
notebooks and in walk-forward loops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logger(
    name: str = "fraud",
    *,
    config: Mapping[str, Any] | None = None,
    level: str | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name. Reusing a name returns the same configured logger.
    config:
        Optional loaded config. If given, ``logging.level`` and
        ``logging.format`` are used unless overridden by explicit arguments.
    level:
        Explicit logging level (e.g. ``"DEBUG"``), overrides config.
    log_file:
        Optional path for a file handler. Parent directories are created.
    """
    cfg_logging = config.get("logging", {}) if config else {}
    chosen_level = level or cfg_logging.get("level", _DEFAULT_LEVEL)
    fmt = cfg_logging.get("format", _DEFAULT_FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(chosen_level)
    logger.propagate = False

    # Remove handlers we previously installed so re-configuring is safe.
    for handler in list(logger.handlers):
        if getattr(handler, "_fraud_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream._fraud_handler = True  # type: ignore[attr-defined]
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._fraud_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    return logger
