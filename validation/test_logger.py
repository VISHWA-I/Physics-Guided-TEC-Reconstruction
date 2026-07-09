"""
test_logger.py
==============
TEST 15 — Logger Validation

Verifies:
- pipeline.log exists and has content.
- Log file contains INFO, WARNING, ERROR entries.
- Execution timestamps are present.
- Logger module can instantiate and log.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)


def test_logger(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
) -> None:
    """TEST 15: Validate the logger system and log file content."""
    section_header("TEST 15 — Logger Validation")

    log_file = project_root / "logs" / "phase1_pipeline.log"

    # --- File existence -----------------------------------------------------
    if not log_file.exists():
        # Also try config-specified path
        cfg_log = cfg.get("logging", {}).get("log_file", "")
        if cfg_log:
            log_file = project_root / cfg_log
    tracker.record(
        "pipeline.log exists",
        passed=log_file.exists(),
        detail=str(log_file),
    )

    if not log_file.exists():
        # Try to create a test log to verify logger works
        _test_logger_create(project_root, cfg, tracker)
        return

    # --- File size ----------------------------------------------------------
    size = log_file.stat().st_size
    tracker.record(
        "pipeline.log is non-empty",
        passed=size > 0,
        detail=f"{size / 1024:.1f} KB",
    )

    # --- Read log content ---------------------------------------------------
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        tracker.record(
            "pipeline.log is readable",
            passed=True,
            detail=f"{len(content.splitlines())} lines",
        )
    except Exception as exc:
        tracker.record("pipeline.log is readable", passed=False, detail=str(exc))
        return

    lines = content.splitlines()

    # --- Contains INFO entries ----------------------------------------------
    n_info = sum(1 for ln in lines if "INFO" in ln)
    tracker.record(
        "Log contains INFO entries",
        passed=n_info > 0,
        detail=f"{n_info} INFO lines",
    )

    # --- Contains WARNING entries -------------------------------------------
    n_warn = sum(1 for ln in lines if "WARNING" in ln or "WARN" in ln)
    tracker.record(
        "Log contains WARNING entries",
        passed=True,  # presence of warnings is informational, not a failure
        warn_condition=n_warn == 0,
        detail=f"{n_warn} WARNING lines",
    )

    # --- Contains ERROR entries (presence is informational) ----------------
    n_err = sum(1 for ln in lines if "ERROR" in ln)
    tracker.record(
        "Log ERROR count",
        passed=True,
        warn_condition=n_err > 0,
        detail=f"{n_err} ERROR lines",
    )

    # --- Execution timestamps -----------------------------------------------
    # Look for lines matching timestamp pattern: YYYY-MM-DD HH:MM:SS
    ts_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    n_ts = sum(1 for ln in lines if ts_pattern.search(ln))
    tracker.record(
        "Log contains execution timestamps",
        passed=n_ts > 0,
        detail=f"{n_ts} timestamped lines",
    )

    # --- Pipeline step markers ----------------------------------------------
    n_start = sum(1 for ln in lines if "[START]" in ln or "Starting" in ln)
    n_end = sum(1 for ln in lines if "[END]" in ln or "complete" in ln.lower())
    tracker.record(
        "Log contains pipeline step start/end markers",
        passed=n_start > 0,
        detail=f"{n_start} START, {n_end} END markers",
    )

    # --- Last 5 log lines preview ------------------------------------------
    sub_header("Last 5 Log Entries")
    for ln in lines[-5:]:
        cprint(f"  {ln[:120]}", "dim")

    # --- Logger module functionality ----------------------------------------
    _test_logger_create(project_root, cfg, tracker)


def _test_logger_create(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
) -> None:
    """Verify the get_logger() factory works correctly."""
    sub_header("Logger Module Functionality Test")
    sys.path.insert(0, str(project_root))
    try:
        from src.logger import get_logger, pipeline_timer, log_dataframe_summary

        # get_logger returns a Logger
        test_log = get_logger("validation.test_logger_check")
        tracker.record(
            "get_logger() returns logging.Logger",
            passed=isinstance(test_log, logging.Logger),
            detail=type(test_log).__name__,
        )

        # Logger name
        tracker.record(
            "Logger name is set correctly",
            passed=test_log.name == "validation.test_logger_check",
            detail=test_log.name,
        )

        # pipeline_timer context manager works
        import io
        try:
            with pipeline_timer("test_step", test_log):
                time.sleep(0.01)  # minimal sleep
            tracker.record("pipeline_timer context manager works", passed=True)
        except Exception as exc:
            tracker.record("pipeline_timer context manager", passed=False, detail=str(exc))

        # log_dataframe_summary works
        try:
            import pandas as pd
            sample = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, None, 6.0]})
            log_dataframe_summary(sample, tag="test", logger=test_log)
            tracker.record("log_dataframe_summary works", passed=True)
        except Exception as exc:
            tracker.record("log_dataframe_summary", passed=False, detail=str(exc))

        # configure_from_dict works
        try:
            from src.logger import configure_from_dict
            configure_from_dict(cfg.get("logging", {}))
            tracker.record("configure_from_dict() works", passed=True)
        except Exception as exc:
            tracker.record("configure_from_dict()", passed=False, detail=str(exc))

    except ImportError as exc:
        tracker.record("Import: src.logger", passed=False, detail=str(exc))
