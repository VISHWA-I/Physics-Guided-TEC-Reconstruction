"""
validation_utils.py
===================
Shared utilities for the Phase 1 Validation Suite.

Provides:
- Colourized console output (PASS=green, WARNING=yellow, FAIL=red)
- TestResult dataclass for structured test outcomes
- ResultTracker for aggregating test results
- Helpers for printing formatted section headers / summary tables

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# ANSI colour support
# ---------------------------------------------------------------------------
_WIN_ANSI_ENABLED: bool = (
    "WT_SESSION" in os.environ                       # Windows Terminal
    or os.environ.get("TERM", "") == "xterm-256color"
    or os.environ.get("FORCE_COLOR", "0") == "1"
    or os.environ.get("COLORTERM", "") in ("truecolor", "24bit")
)

_ANSI_AVAILABLE: bool = sys.platform != "win32" or _WIN_ANSI_ENABLED

# Enable VT100 on Windows if possible
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        _ANSI_AVAILABLE = True
    except Exception:
        pass

_C: dict[str, str]
if _ANSI_AVAILABLE:
    _C = {
        "green":   "\033[92m",
        "yellow":  "\033[93m",
        "red":     "\033[91m",
        "cyan":    "\033[96m",
        "magenta": "\033[95m",
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "reset":   "\033[0m",
    }
else:
    _C = {k: "" for k in ("green", "yellow", "red", "cyan", "magenta", "bold", "dim", "reset")}


def cprint(msg: str, colour: str = "") -> None:
    """Print *msg* with an optional ANSI colour code, then reset."""
    code = _C.get(colour, "")
    print(f"{code}{msg}{_C['reset']}", flush=True)


def pass_msg(label: str, detail: str = "") -> None:
    detail_str = f"  — {detail}" if detail else ""
    cprint(f"  [PASS]    {label}{detail_str}", "green")


def warn_msg(label: str, detail: str = "") -> None:
    detail_str = f"  — {detail}" if detail else ""
    cprint(f"  [WARNING] {label}{detail_str}", "yellow")


def fail_msg(label: str, detail: str = "") -> None:
    detail_str = f"  — {detail}" if detail else ""
    cprint(f"  [FAIL]    {label}{detail_str}", "red")


def section_header(title: str) -> None:
    width = 70
    bar = "=" * width
    cprint(f"\n{bar}", "cyan")
    cprint(f"  {title}", "bold")
    cprint(f"{bar}", "cyan")


def sub_header(title: str) -> None:
    cprint(f"\n  {'─' * 60}", "dim")
    cprint(f"  {title}", "cyan")
    cprint(f"  {'─' * 60}", "dim")


# ---------------------------------------------------------------------------
# TestResult dataclass
# ---------------------------------------------------------------------------

STATUS_PASS = "PASS"
STATUS_WARN = "WARNING"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"


@dataclass
class TestResult:
    """Structured result for a single test check."""

    name: str
    status: str          # PASS | WARNING | FAIL | SKIP
    detail: str = ""
    elapsed_s: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------------------
# ResultTracker
# ---------------------------------------------------------------------------


class ResultTracker:
    """Aggregate and report test results across all test modules."""

    def __init__(self) -> None:
        self._results: List[TestResult] = []
        self._logger = logging.getLogger("Phase1Validator")

    # ------------------------------------------------------------------
    def add(self, result: TestResult) -> None:
        """Register a single TestResult."""
        self._results.append(result)
        if result.status == STATUS_PASS:
            pass_msg(result.name, result.detail)
        elif result.status == STATUS_WARN:
            warn_msg(result.name, result.detail)
        elif result.status == STATUS_FAIL:
            fail_msg(result.name, result.detail)
        elif result.status == STATUS_SKIP:
            cprint(f"  [SKIP]    {result.name}  — {result.detail}", "dim")

    def record(
        self,
        name: str,
        passed: bool,
        detail: str = "",
        warn_condition: bool = False,
        elapsed_s: float = 0.0,
    ) -> TestResult:
        """
        Convenience builder that maps a boolean to PASS / WARNING / FAIL.

        Parameters
        ----------
        name         : human-readable test label
        passed       : True  -> PASS,  False + warn_condition -> WARNING, else FAIL
        detail       : short explanatory string
        warn_condition : if True and not passed -> WARNING instead of FAIL
        elapsed_s    : timing info
        """
        if passed:
            status = STATUS_PASS
        elif warn_condition:
            status = STATUS_WARN
        else:
            status = STATUS_FAIL
        r = TestResult(name=name, status=status, detail=detail, elapsed_s=elapsed_s)
        self.add(r)
        return r

    # ------------------------------------------------------------------
    @property
    def results(self) -> List[TestResult]:
        return list(self._results)

    @property
    def n_pass(self) -> int:
        return sum(r.status == STATUS_PASS for r in self._results)

    @property
    def n_warn(self) -> int:
        return sum(r.status == STATUS_WARN for r in self._results)

    @property
    def n_fail(self) -> int:
        return sum(r.status == STATUS_FAIL for r in self._results)

    @property
    def n_skip(self) -> int:
        return sum(r.status == STATUS_SKIP for r in self._results)

    @property
    def total(self) -> int:
        return len(self._results)

    @property
    def overall_pass(self) -> bool:
        """True iff zero FAIL results."""
        return self.n_fail == 0

    # ------------------------------------------------------------------
    def print_summary_table(self) -> None:
        """Print a formatted summary table to stdout."""
        width = 70
        section_header("PHASE 1 VALIDATION SUMMARY")
        print()
        cprint(f"  {'Test':<45}  {'Status':^9}  {'Detail'}", "bold")
        cprint(f"  {'─' * 45}  {'─' * 9}  {'─' * 20}", "dim")

        for r in self._results:
            if r.status == STATUS_PASS:
                colour = "green"
                badge = f"[PASS]"
            elif r.status == STATUS_WARN:
                colour = "yellow"
                badge = f"[WARN]"
            elif r.status == STATUS_FAIL:
                colour = "red"
                badge = f"[FAIL]"
            else:
                colour = "dim"
                badge = f"[SKIP]"

            detail_short = r.detail[:40] if r.detail else ""
            label = r.name[:44]
            line = f"  {label:<45}  "
            print(line, end="", flush=True)
            cprint(f"{badge:^9}", colour)

        print()
        cprint(f"  {'─' * width}", "dim")
        cprint(f"  TOTAL:   {self.total:3d}   "
               f"PASS: {self.n_pass}   "
               f"WARN: {self.n_warn}   "
               f"FAIL: {self.n_fail}   "
               f"SKIP: {self.n_skip}", "bold")
        print()

        if self.overall_pass and self.n_warn == 0:
            cprint(f"  [OK] OVERALL STATUS: READY FOR PHASE 2", "green")
        elif self.overall_pass:
            cprint(f"  [OK] OVERALL STATUS: READY FOR PHASE 2 (with warnings)", "yellow")
        else:
            cprint(f"  [FAIL] OVERALL STATUS: NOT READY -- fix {self.n_fail} failure(s) before Phase 2", "red")
        print()
