"""
logger.py
=========
Professional, rotating, multi-handler logger for the Mamba-TKAN TEC
Reconstruction project.

Features
--------
* Console and rotating file handlers with configurable levels.
* Single ``get_logger`` factory keeps one logger per name across all modules.
* ``PipelineTimer`` context-manager for measuring execution time of any block.
* Helper ``log_dataframe_summary`` for quick DataFrame diagnostics.
* Colour-coded console output on POSIX systems (gracefully disabled on Windows
  when colour codes are not supported).

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Optional colour support
# ---------------------------------------------------------------------------
_COLOURS_ENABLED: bool = (
    sys.platform != "win32"                     # ANSI not available on raw cmd.exe
    or os.environ.get("TERM", "") == "xterm-256color"
    or "WT_SESSION" in os.environ               # Windows Terminal supports ANSI
    or os.environ.get("FORCE_COLOR", "0") == "1"
)

_COLOUR_MAP: dict[str, str] = {
    "DEBUG":    "\033[36m",    # cyan
    "INFO":     "\033[32m",    # green
    "WARNING":  "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[1;31m",  # bold red
    "RESET":    "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    """Formatter that prepends ANSI colour codes to the level-name portion."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        msg = super().format(record)
        if _COLOURS_ENABLED:
            colour = _COLOUR_MAP.get(record.levelname, "")
            reset = _COLOUR_MAP["RESET"]
            msg = msg.replace(record.levelname, f"{colour}{record.levelname}{reset}", 1)
        return msg


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

# Registry: module name -> Logger  (avoids duplicate handlers)
_REGISTRY: dict[str, logging.Logger] = {}

_DEFAULT_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
_DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    *,
    level: str = "DEBUG",
    log_file: Optional[str] = None,
    console: bool = True,
    rotation: str = "midnight",
    backup_count: int = 7,
    fmt: str = _DEFAULT_FORMAT,
    date_fmt: str = _DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    Return a named logger, creating it on the first call and returning the
    same instance on subsequent calls (idempotent).

    Parameters
    ----------
    name : str
        Logger name - typically ``__name__`` of the calling module.
    level : str
        Root logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    log_file : str, optional
        Absolute or relative path to the log file.  If *None*, no file
        handler is attached.
    console : bool
        Whether to add a ``StreamHandler`` writing to *stderr*.
    rotation : str
        Rotation frequency for ``TimedRotatingFileHandler``
        (``midnight`` | ``h`` | ``d`` | ``w0``-``w6``).
    backup_count : int
        Number of rotated log files to retain.
    fmt : str
        Log record format string.
    date_fmt : str
        Date/time format string.

    Returns
    -------
    logging.Logger
    """
    if name in _REGISTRY:
        return _REGISTRY[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    logger.propagate = False  # prevent double-logging to root logger

    # -- Console handler -------------------------------------------------------
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        ch.setFormatter(_ColourFormatter(fmt, datefmt=date_fmt))
        logger.addHandler(ch)

    # -- File handler ----------------------------------------------------------
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(
            filename=str(log_path),
            when=rotation,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)   # always capture everything in file
        fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(fh)

    _REGISTRY[name] = logger
    return logger


def configure_from_dict(cfg: dict) -> None:
    """
    Re-configure the root logger using a configuration dictionary (e.g., from
    the YAML config file).

    Parameters
    ----------
    cfg : dict
        Sub-dictionary under the ``logging`` key of the project config.

    Example
    -------
    >>> from src.logger import configure_from_dict
    >>> configure_from_dict(config["logging"])
    """
    global_level: str = cfg.get("level", "DEBUG")
    log_file: Optional[str] = cfg.get("log_file") if cfg.get("file", True) else None
    console: bool = cfg.get("console", True)
    rotation: str = cfg.get("rotation", "midnight")
    backup_count: int = cfg.get("backup_count", 7)
    fmt: str = cfg.get("format", _DEFAULT_FORMAT)
    date_fmt: str = cfg.get("date_format", _DEFAULT_DATE_FORMAT)

    # Apply settings to root logger so child loggers inherit the level
    root = logging.getLogger()
    root.setLevel(getattr(logging, global_level.upper(), logging.DEBUG))

    # Re-configure any already-registered loggers with new file path
    for name, existing_logger in _REGISTRY.items():
        existing_logger.setLevel(getattr(logging, global_level.upper(), logging.DEBUG))
        if log_file and not any(
            isinstance(h, TimedRotatingFileHandler) for h in existing_logger.handlers
        ):
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = TimedRotatingFileHandler(
                filename=str(log_path),
                when=rotation,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
            existing_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Pipeline Timer
# ---------------------------------------------------------------------------

@contextmanager
def pipeline_timer(
    step_name: str,
    logger: Optional[logging.Logger] = None,
) -> Generator[None, None, None]:
    """
    Context manager that measures the wall-clock execution time of a code
    block and logs it at INFO level.

    Parameters
    ----------
    step_name : str
        Human-readable label for the step being timed.
    logger : logging.Logger, optional
        Logger to write timing info to.  Falls back to the root logger.

    Example
    -------
    >>> with pipeline_timer("Feature Engineering", logger):
    ...     run_feature_engineering()
    """
    _log = logger or logging.getLogger(__name__)
    _log.info("[START] %s", step_name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - t0
        _log.error("[FAILED] %s  (after %.3f s)", step_name, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - t0
        _log.info("[END]   %s  (%.3f s)", step_name, elapsed)


# ---------------------------------------------------------------------------
# DataFrame Diagnostic Helper
# ---------------------------------------------------------------------------

def log_dataframe_summary(
    df: pd.DataFrame,
    tag: str = "DataFrame",
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> None:
    """
    Log a concise summary of a DataFrame including shape, dtypes, and NaN
    counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarise.
    tag : str
        Label printed before the summary.
    logger : logging.Logger, optional
        Destination logger.
    verbose : bool
        If *True*, include per-column NaN counts and dtypes.
    """
    _log = logger or logging.getLogger(__name__)
    rows, cols = df.shape
    total_nan = int(df.isna().sum().sum())
    nan_pct = 100.0 * total_nan / max(rows * cols, 1)

    _log.info(
        "[%s] shape=(%d, %d)  total_NaN=%d (%.2f%%)  memory=%.2f MB",
        tag,
        rows,
        cols,
        total_nan,
        nan_pct,
        df.memory_usage(deep=True).sum() / 1e6,
    )

    if verbose and total_nan > 0:
        nan_series = df.isna().sum()
        nan_series = nan_series[nan_series > 0].sort_values(ascending=False)
        for col, count in nan_series.items():
            _log.debug(
                "  [%s] missing=%d / %d  (%.1f%%)",
                col,
                count,
                rows,
                100.0 * count / max(rows, 1),
            )

    if verbose:
        dtype_info = df.dtypes.value_counts().to_dict()
        _log.debug("[%s] dtypes: %s", tag, dtype_info)


# ---------------------------------------------------------------------------
# Missing Data Reporter
# ---------------------------------------------------------------------------

def log_missing_data_report(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> dict[str, float]:
    """
    Log a detailed missing data report per column and return a dictionary
    mapping column name -> missing fraction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    logger : logging.Logger, optional
        Destination logger.

    Returns
    -------
    dict[str, float]
        Mapping ``{column_name: missing_fraction}``.
    """
    _log = logger or logging.getLogger(__name__)
    report: dict[str, float] = {}
    n_rows = len(df)

    _log.info("=== Missing Data Report (n_rows=%d) ===", n_rows)
    has_missing = False
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        frac = n_missing / max(n_rows, 1)
        report[col] = frac
        if n_missing > 0:
            has_missing = True
            level = logging.WARNING if frac > 0.10 else logging.INFO
            _log.log(level, "  %-25s  missing=%6d / %d  (%.2f%%)", col, n_missing, n_rows, frac * 100)

    if not has_missing:
        _log.info("  [OK] No missing values detected.")

    return report
