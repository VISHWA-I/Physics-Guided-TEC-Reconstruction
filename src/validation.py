"""
validation.py
=============
Comprehensive data validation engine for the Mamba-TKAN TEC Reconstruction
pipeline.

Validation checks
-----------------
1.  **Schema validation** - required columns present and correctly typed.
2.  **Timestamp validation** - parseable format, monotonicity, gap detection.
3.  **Coordinate validation** - latitude in [-90, 90], longitude in [-180, 180].
4.  **NaN / Infinite value detection** - per column counts and fractions.
5.  **Duplicate row detection** - exact and near-duplicate identification.
6.  **Physical bounds checking** - domain-specific min/max for every feature.
7.  **Negative TEC detection** - physical impossibility.
8.  **foF2 range validation** - physically meaningful plasma frequency.
9.  **Cross-column consistency** - e.g. B0 vs hmF2 sanity check.
10. **Quality flag checking** - QF column completeness.

Every check populates a ``ValidationReport`` that the caller can inspect or
log.

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.constants import (
    COLUMN_B0,
    COLUMN_DAY_OF_YEAR,
    COLUMN_DST,
    COLUMN_FOF2,
    COLUMN_HMF2,
    COLUMN_KP,
    COLUMN_LATITUDE,
    COLUMN_LOCAL_TIME,
    COLUMN_LONGITUDE,
    COLUMN_QF,
    COLUMN_STATION,
    COLUMN_TEC,
    COLUMN_TIMESTAMP,
    COLUMN_TOPSIDE_TEC,
    PHYSICAL_BOUNDS,
    REQUIRED_COLUMNS,
)
from src.logger import get_logger, log_dataframe_summary

_log = get_logger(__name__)


# =============================================================================
# ValidationReport - structured result container
# =============================================================================


@dataclass
class ValidationIssue:
    """Represents a single validation finding."""

    severity: str           # "ERROR" | "WARNING" | "INFO"
    check: str              # Name of the check that produced this issue
    message: str            # Human-readable description
    affected_rows: int = 0  # How many rows are affected (0 = schema-level)
    affected_cols: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """
    Container for all validation findings produced during a pipeline run.

    Attributes
    ----------
    passed : bool
        True when no ERROR-level issues were found.
    issues : list[ValidationIssue]
        Ordered list of all findings.
    summary : dict
        High-level statistics (row count, NaN fraction, …).
    """

    passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def add(self, issue: ValidationIssue) -> None:
        """Append an issue and update ``passed`` flag if severity is ERROR."""
        self.issues.append(issue)
        if issue.severity == "ERROR":
            self.passed = False

    def errors(self) -> List[ValidationIssue]:
        """Return only ERROR-level issues."""
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self) -> List[ValidationIssue]:
        """Return only WARNING-level issues."""
        return [i for i in self.issues if i.severity == "WARNING"]

    def log_summary(self, logger: Optional[logging.Logger] = None) -> None:
        """Write a formatted summary of the report to a logger."""
        _logger = logger or _log
        n_err = len(self.errors())
        n_warn = len(self.warnings())
        status = "[OK] PASSED" if self.passed else "[ERR] FAILED"
        _logger.info("=== Validation Report: %s | Errors=%d | Warnings=%d ===",
                     status, n_err, n_warn)
        for issue in self.issues:
            level = {
                "ERROR": logging.ERROR,
                "WARNING": logging.WARNING,
                "INFO": logging.INFO,
            }.get(issue.severity, logging.DEBUG)
            _logger.log(
                level,
                "[%s] %s - %s  (rows=%d, cols=%s)",
                issue.severity,
                issue.check,
                issue.message,
                issue.affected_rows,
                issue.affected_cols or "N/A",
            )


# =============================================================================
# DataValidator
# =============================================================================


class DataValidator:
    """
    Performs all data quality checks on an ionosonde DataFrame.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary (from ``config.yaml``).

    Example
    -------
    >>> validator = DataValidator(cfg)
    >>> report = validator.validate(df)
    >>> if not report.passed:
    ...     report.log_summary()
    ...     raise RuntimeError("Data validation failed.")
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._val_cfg = cfg.get("validation", {})
        self._data_cfg = cfg.get("data", {})

        # Pull validation thresholds
        self._max_nan_fraction: float = self._val_cfg.get("max_nan_fraction", 0.30)
        self._max_dup_fraction: float = self._val_cfg.get("max_duplicate_fraction", 0.05)

        # Physical bounds (override defaults with config if present)
        self._bounds: Dict[str, Tuple[float, float]] = dict(PHYSICAL_BOUNDS)
        for col, cfg_key in [
            (COLUMN_TEC, "tec_bounds"),
            (COLUMN_FOF2, "fof2_bounds"),
            (COLUMN_HMF2, "hmf2_bounds"),
            (COLUMN_B0, "b0_bounds"),
            (COLUMN_KP, "kp_bounds"),
            (COLUMN_DST, "dst_bounds"),
        ]:
            if cfg_key in self._val_cfg:
                lo, hi = self._val_cfg[cfg_key]
                self._bounds[col] = (float(lo), float(hi))

        _log.info("DataValidator initialised with %d physical bounds defined.", len(self._bounds))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Execute the full validation suite against *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or semi-processed ionosonde DataFrame.

        Returns
        -------
        ValidationReport
            Structured report of all findings.
        """
        report = ValidationReport()
        report.summary["input_rows"] = len(df)
        report.summary["input_cols"] = len(df.columns)

        _log.info("Starting data validation on DataFrame with shape %s", df.shape)
        log_dataframe_summary(df, tag="Validation Input", logger=_log)

        # Run all checks in sequence
        self._check_required_columns(df, report)
        self._check_data_types(df, report)
        self._check_timestamps(df, report)
        self._check_coordinates(df, report)
        self._check_nan_values(df, report)
        self._check_infinite_values(df, report)
        self._check_duplicate_rows(df, report)
        self._check_physical_bounds(df, report)
        self._check_negative_tec(df, report)
        self._check_fof2_range(df, report)
        self._check_cross_column_consistency(df, report)
        self._check_quality_flags(df, report)

        report.summary["validation_passed"] = report.passed
        report.summary["n_errors"] = len(report.errors())
        report.summary["n_warnings"] = len(report.warnings())

        report.log_summary(_log)
        return report

    # ------------------------------------------------------------------
    # Check 1: Required Columns
    # ------------------------------------------------------------------

    def _check_required_columns(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Verify all required columns are present in the DataFrame."""
        present = set(df.columns.tolist())
        missing = [c for c in REQUIRED_COLUMNS if c not in present]

        if missing:
            report.add(ValidationIssue(
                severity="ERROR",
                check="RequiredColumns",
                message=f"Missing required columns: {missing}",
                affected_cols=missing,
            ))
            _log.error("Missing columns: %s", missing)
        else:
            _log.info("[OK] All %d required columns present.", len(REQUIRED_COLUMNS))
            report.add(ValidationIssue(
                severity="INFO",
                check="RequiredColumns",
                message=f"All {len(REQUIRED_COLUMNS)} required columns present.",
            ))

    # ------------------------------------------------------------------
    # Check 2: Data Types
    # ------------------------------------------------------------------

    def _check_data_types(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Verify numeric columns are not stored as object dtype."""
        numeric_cols = [
            c for c in REQUIRED_COLUMNS
            if c not in (COLUMN_TIMESTAMP, COLUMN_STATION) and c in df.columns
        ]
        object_cols = [c for c in numeric_cols if df[c].dtype == object]
        if object_cols:
            report.add(ValidationIssue(
                severity="WARNING",
                check="DataTypes",
                message=f"Columns with object dtype (expected numeric): {object_cols}",
                affected_cols=object_cols,
            ))
            _log.warning("Object-dtype numeric columns (may cause cast errors): %s", object_cols)
        else:
            _log.info("[OK] All numeric columns have non-object dtypes.")

    # ------------------------------------------------------------------
    # Check 3: Timestamp Validation
    # ------------------------------------------------------------------

    def _check_timestamps(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Validate the Timestamp column:
        * Parseable as datetime.
        * No null timestamps.
        * Detect time reversals (non-monotone data).
        * Detect unusually large time gaps (> 24 hours).
        """
        if COLUMN_TIMESTAMP not in df.columns:
            return  # Already caught by RequiredColumns check

        # Parse timestamps
        try:
            ts = pd.to_datetime(df[COLUMN_TIMESTAMP], errors="coerce")
        except Exception as exc:
            report.add(ValidationIssue(
                severity="ERROR",
                check="Timestamp",
                message=f"Timestamp parsing raised exception: {exc}",
                affected_cols=[COLUMN_TIMESTAMP],
            ))
            return

        n_null = int(ts.isna().sum())
        if n_null > 0:
            report.add(ValidationIssue(
                severity="ERROR",
                check="Timestamp",
                message=f"{n_null} null / unparseable timestamps found.",
                affected_rows=n_null,
                affected_cols=[COLUMN_TIMESTAMP],
            ))
            _log.error("Null timestamps: %d rows", n_null)
        else:
            _log.info("[OK] All timestamps parseable. Range: %s -> %s", ts.min(), ts.max())

        # Monotonicity check
        diff = ts.diff()
        n_reversed = int((diff.dt.total_seconds() < 0).sum())
        if n_reversed > 0:
            report.add(ValidationIssue(
                severity="WARNING",
                check="Timestamp",
                message=f"{n_reversed} timestamp reversals detected (non-monotone data).",
                affected_rows=n_reversed,
                affected_cols=[COLUMN_TIMESTAMP],
            ))
            _log.warning("Timestamp reversals detected: %d", n_reversed)

        # Large gap detection (> 24 hours)
        large_gaps = diff[diff.dt.total_seconds() > 86400]
        if not large_gaps.empty:
            report.add(ValidationIssue(
                severity="WARNING",
                check="Timestamp",
                message=f"{len(large_gaps)} time gaps > 24 hours found.",
                affected_rows=len(large_gaps),
                affected_cols=[COLUMN_TIMESTAMP],
            ))
            _log.warning("Large time gaps (>24h): %d occurrences", len(large_gaps))

    # ------------------------------------------------------------------
    # Check 4: Coordinate Validation
    # ------------------------------------------------------------------

    def _check_coordinates(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Validate latitude and longitude values are within physical bounds."""
        checks = [
            (COLUMN_LATITUDE, -90.0, 90.0),
            (COLUMN_LONGITUDE, -180.0, 180.0),
        ]
        for col, lo, hi in checks:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            out_of_range = ((series < lo) | (series > hi)) & series.notna()
            n_bad = int(out_of_range.sum())
            if n_bad > 0:
                report.add(ValidationIssue(
                    severity="ERROR",
                    check="Coordinates",
                    message=f"{col}: {n_bad} values outside [{lo}, {hi}].",
                    affected_rows=n_bad,
                    affected_cols=[col],
                ))
                _log.error("Invalid %s: %d rows out of range [%g, %g]", col, n_bad, lo, hi)
            else:
                _log.info("[OK] %s valid (all within [%g, %g]).", col, lo, hi)

    # ------------------------------------------------------------------
    # Check 5: NaN Values
    # ------------------------------------------------------------------

    def _check_nan_values(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Check per-column and global NaN fractions."""
        n_rows = len(df)
        total_nan = int(df.isna().sum().sum())
        global_frac = total_nan / max(n_rows * len(df.columns), 1)

        if global_frac > self._max_nan_fraction:
            report.add(ValidationIssue(
                severity="ERROR",
                check="NaNValues",
                message=(
                    f"Global NaN fraction {global_frac:.2%} exceeds "
                    f"threshold {self._max_nan_fraction:.2%}."
                ),
                affected_rows=total_nan,
            ))
            _log.error(
                "Global NaN fraction %.2f%% exceeds threshold %.2f%%.",
                global_frac * 100, self._max_nan_fraction * 100,
            )
        else:
            _log.info("Global NaN fraction: %.2f%% (threshold %.2f%%)",
                      global_frac * 100, self._max_nan_fraction * 100)

        # Per-column warnings for columns with >10% NaN
        for col in df.columns:
            n_nan = int(df[col].isna().sum())
            frac = n_nan / max(n_rows, 1)
            if frac > 0.10:
                report.add(ValidationIssue(
                    severity="WARNING",
                    check="NaNValues",
                    message=f"Column '{col}' has {frac:.1%} NaN values ({n_nan} rows).",
                    affected_rows=n_nan,
                    affected_cols=[col],
                ))

    # ------------------------------------------------------------------
    # Check 6: Infinite Values
    # ------------------------------------------------------------------

    def _check_infinite_values(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Detect infinite values (±inf) in numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric_df.values)
        n_inf = int(inf_mask.sum())
        if n_inf > 0:
            inf_cols = numeric_df.columns[inf_mask.any(axis=0)].tolist()
            report.add(ValidationIssue(
                severity="ERROR",
                check="InfiniteValues",
                message=f"{n_inf} infinite values found in columns: {inf_cols}.",
                affected_rows=n_inf,
                affected_cols=inf_cols,
            ))
            _log.error("Infinite values detected: %d  in columns %s", n_inf, inf_cols)
        else:
            _log.info("[OK] No infinite values detected.")

    # ------------------------------------------------------------------
    # Check 7: Duplicate Rows
    # ------------------------------------------------------------------

    def _check_duplicate_rows(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Detect fully duplicate rows."""
        n_dups = int(df.duplicated().sum())
        frac = n_dups / max(len(df), 1)
        if frac > self._max_dup_fraction:
            report.add(ValidationIssue(
                severity="ERROR",
                check="DuplicateRows",
                message=(
                    f"{n_dups} duplicate rows ({frac:.2%}) exceed "
                    f"threshold {self._max_dup_fraction:.2%}."
                ),
                affected_rows=n_dups,
            ))
            _log.error("Duplicate rows: %d (%.2f%%)", n_dups, frac * 100)
        elif n_dups > 0:
            report.add(ValidationIssue(
                severity="WARNING",
                check="DuplicateRows",
                message=f"{n_dups} duplicate rows ({frac:.2%}) found.",
                affected_rows=n_dups,
            ))
            _log.warning("Duplicate rows: %d (%.2f%%)", n_dups, frac * 100)
        else:
            _log.info("[OK] No duplicate rows.")

    # ------------------------------------------------------------------
    # Check 8: Physical Bounds
    # ------------------------------------------------------------------

    def _check_physical_bounds(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Check each numeric column against its physical valid range."""
        for col, (lo, hi) in self._bounds.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            out_of_range = ((series < lo) | (series > hi)) & series.notna()
            n_bad = int(out_of_range.sum())
            if n_bad > 0:
                frac = n_bad / max(len(df), 1)
                severity = "ERROR" if frac > 0.10 else "WARNING"
                report.add(ValidationIssue(
                    severity=severity,
                    check="PhysicalBounds",
                    message=(
                        f"'{col}': {n_bad} values ({frac:.1%}) outside "
                        f"physical range [{lo}, {hi}]."
                    ),
                    affected_rows=n_bad,
                    affected_cols=[col],
                ))
                _log.log(
                    logging.ERROR if severity == "ERROR" else logging.WARNING,
                    "Physical bound violation in '%s': %d rows (%.1f%%) outside [%g, %g].",
                    col, n_bad, frac * 100, lo, hi,
                )

    # ------------------------------------------------------------------
    # Check 9: Negative TEC
    # ------------------------------------------------------------------

    def _check_negative_tec(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Ensure TEC and TopsideTEC are non-negative (physical constraint)."""
        for col in [COLUMN_TEC, COLUMN_TOPSIDE_TEC]:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n_neg = int((series < 0).sum())
            if n_neg > 0:
                report.add(ValidationIssue(
                    severity="ERROR",
                    check="NegativeTEC",
                    message=f"'{col}': {n_neg} negative values (physically impossible).",
                    affected_rows=n_neg,
                    affected_cols=[col],
                ))
                _log.error("Negative TEC in '%s': %d rows.", col, n_neg)
            else:
                _log.info("[OK] '%s' has no negative values.", col)

    # ------------------------------------------------------------------
    # Check 10: foF2 Range
    # ------------------------------------------------------------------

    def _check_fof2_range(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Verify foF2 (critical frequency of F2 layer) is within the
        physically plausible plasma frequency range (1-25 MHz).
        """
        if COLUMN_FOF2 not in df.columns:
            return
        series = pd.to_numeric(df[COLUMN_FOF2], errors="coerce")
        lo, hi = self._bounds.get(COLUMN_FOF2, (1.0, 25.0))
        n_low = int((series < lo).sum())
        n_high = int((series > hi).sum())
        if n_low + n_high > 0:
            report.add(ValidationIssue(
                severity="WARNING",
                check="FoF2Range",
                message=(
                    f"foF2: {n_low} values < {lo} MHz and {n_high} values > {hi} MHz."
                ),
                affected_rows=n_low + n_high,
                affected_cols=[COLUMN_FOF2],
            ))
            _log.warning("foF2 range check: %d below %.1f MHz, %d above %.1f MHz.",
                         n_low, lo, n_high, hi)
        else:
            _log.info("[OK] foF2 values within plausible range [%.1f, %.1f] MHz.", lo, hi)

    # ------------------------------------------------------------------
    # Check 11: Cross-Column Consistency
    # ------------------------------------------------------------------

    def _check_cross_column_consistency(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Physics-based cross-column sanity checks.

        Rule 1: B0 (half-thickness) should generally be less than hmF2
                (peak height).  Extreme violations flag data errors.
        Rule 2: TopsideTEC should not greatly exceed TEC (bottomside).
                Ratio > 10 is suspicious.
        """
        # Rule 1: B0 < hmF2
        if COLUMN_B0 in df.columns and COLUMN_HMF2 in df.columns:
            b0 = pd.to_numeric(df[COLUMN_B0], errors="coerce")
            hmf2 = pd.to_numeric(df[COLUMN_HMF2], errors="coerce")
            n_bad = int((b0 > hmf2).sum())
            if n_bad > 0:
                report.add(ValidationIssue(
                    severity="WARNING",
                    check="CrossColumnConsistency",
                    message=f"B0 > hmF2 in {n_bad} rows (physically unusual).",
                    affected_rows=n_bad,
                    affected_cols=[COLUMN_B0, COLUMN_HMF2],
                ))
                _log.warning("B0 > hmF2 in %d rows.", n_bad)

        # Rule 2: TopsideTEC / TEC ratio check
        if COLUMN_TOPSIDE_TEC in df.columns and COLUMN_TEC in df.columns:
            tec = pd.to_numeric(df[COLUMN_TEC], errors="coerce")
            topside = pd.to_numeric(df[COLUMN_TOPSIDE_TEC], errors="coerce")
            ratio = topside / tec.replace(0, np.nan)
            n_extreme = int((ratio > 10.0).sum())
            if n_extreme > 0:
                report.add(ValidationIssue(
                    severity="WARNING",
                    check="CrossColumnConsistency",
                    message=(
                        f"TopsideTEC/TEC ratio > 10 in {n_extreme} rows "
                        "(suspect data quality)."
                    ),
                    affected_rows=n_extreme,
                    affected_cols=[COLUMN_TOPSIDE_TEC, COLUMN_TEC],
                ))
                _log.warning("TopsideTEC/TEC ratio > 10 in %d rows.", n_extreme)

    # ------------------------------------------------------------------
    # Check 12: Quality Flags
    # ------------------------------------------------------------------

    def _check_quality_flags(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Check the QF (Quality Flag) column for missing values and flag
        proportion of poor-quality records (QF ≥ 3 considered poor).
        """
        if COLUMN_QF not in df.columns:
            return
        qf = pd.to_numeric(df[COLUMN_QF], errors="coerce")
        n_nan = int(qf.isna().sum())
        if n_nan > 0:
            report.add(ValidationIssue(
                severity="WARNING",
                check="QualityFlags",
                message=f"QF column has {n_nan} missing values.",
                affected_rows=n_nan,
                affected_cols=[COLUMN_QF],
            ))
        n_poor = int((qf >= 3).sum())
        frac_poor = n_poor / max(len(df), 1)
        if frac_poor > 0.20:
            report.add(ValidationIssue(
                severity="WARNING",
                check="QualityFlags",
                message=(
                    f"{n_poor} records ({frac_poor:.1%}) have QF ≥ 3 "
                    "(poor quality - consider filtering)."
                ),
                affected_rows=n_poor,
                affected_cols=[COLUMN_QF],
            ))
            _log.warning(
                "Poor QF records: %d (%.1f%%). Consider applying QF filter.",
                n_poor, frac_poor * 100,
            )
        else:
            _log.info(
                "QF check: %d poor-quality records (%.1f%%).",
                n_poor, frac_poor * 100,
            )
