"""
test_dataset.py
===============
TEST 3  - Real Data Existence & Readability
TEST 4  - CSV Schema Validation
TEST 5  - Data-type Validation
TEST 6  - Missing Value Analysis
TEST 7  - Duplicate Detection
TEST 8  - Outlier Detection

Operates on the actual GIRO / OMNI CSV files in data/raw/ and the
processed dataset in data/processed/.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from validation.validation_utils import (
    ResultTracker,
    STATUS_PASS, STATUS_WARN, STATUS_FAIL,
    section_header, sub_header, cprint,
)

# ---------------------------------------------------------------------------
# Required CSV schema columns
# ---------------------------------------------------------------------------
REQUIRED_SCHEMA_COLUMNS: List[str] = [
    "Timestamp", "Station", "Latitude", "Longitude",
    "TEC", "foF2", "hmF2", "scaleF2", "B0", "B1",
    "zhalfNm", "yF2", "FF", "QF",
    "F10.7", "Kp", "Ap", "Dst", "SSN",
    "DayOfYear", "LocalTime", "TopsideTEC",
]

# Columns that must be float
FLOAT_COLUMNS: List[str] = [
    "Latitude", "Longitude", "TEC", "foF2", "hmF2", "scaleF2",
    "B0", "B1", "zhalfNm", "yF2", "FF", "QF",
    "F10.7", "Kp", "Ap", "Dst", "SSN",
    "DayOfYear", "LocalTime", "TopsideTEC",
]


# ---------------------------------------------------------------------------
# TEST 3 — Real Data Existence
# ---------------------------------------------------------------------------

def test_real_data(project_root: Path, tracker: ResultTracker) -> Optional[pd.DataFrame]:
    """
    TEST 3: Verify that GIRO ionosonde dataset exists, is readable and non-empty.

    Returns the loaded raw DataFrame if successful, else None.
    """
    section_header("TEST 3 — Real Data (GIRO Ionosonde)")

    raw_dir = project_root / "data" / "raw"
    ionosonde_path = raw_dir / "ionosonde_data.csv"

    # --- File existence -----------------------------------------------------
    if not ionosonde_path.exists():
        tracker.record("ionosonde_data.csv exists", passed=False, detail=str(ionosonde_path))
        return None
    tracker.record("ionosonde_data.csv exists", passed=True, detail=str(ionosonde_path))

    # --- File size ----------------------------------------------------------
    size_bytes = ionosonde_path.stat().st_size
    tracker.record(
        "ionosonde_data.csv is non-empty",
        passed=size_bytes > 0,
        detail=f"{size_bytes / 1024:.1f} KB",
    )
    if size_bytes == 0:
        return None

    # --- Readability --------------------------------------------------------
    try:
        t0 = time.perf_counter()
        df = pd.read_csv(ionosonde_path, low_memory=False)
        elapsed = time.perf_counter() - t0
        tracker.record(
            "ionosonde_data.csv is readable",
            passed=True,
            detail=f"{len(df)} rows × {len(df.columns)} cols  [{elapsed:.2f}s]",
        )
    except Exception as exc:
        tracker.record("ionosonde_data.csv is readable", passed=False, detail=str(exc))
        return None

    # --- Non-trivial row count ---------------------------------------------
    tracker.record(
        "ionosonde_data.csv has ≥100 rows",
        passed=len(df) >= 100,
        detail=f"{len(df)} rows",
    )

    # --- No OMNI separate file (merged into ionosonde_data.csv) -----------
    sub_header("Space Weather (OMNI) columns check")
    omni_cols = ["F10.7", "Kp", "Ap", "Dst", "SSN"]
    for col in omni_cols:
        tracker.record(
            f"OMNI column present: {col}",
            passed=col in df.columns,
            warn_condition=True,
            detail="not found" if col not in df.columns else "ok",
        )

    return df


# ---------------------------------------------------------------------------
# TEST 4 — CSV Schema Validation
# ---------------------------------------------------------------------------

def test_schema(df: pd.DataFrame, tracker: ResultTracker) -> None:
    """TEST 4: Verify all required columns exist in the dataset."""
    section_header("TEST 4 — CSV Schema Validation")

    for col in REQUIRED_SCHEMA_COLUMNS:
        tracker.record(
            f"Column present: {col}",
            passed=col in df.columns,
            detail="MISSING" if col not in df.columns else str(df[col].dtype),
        )

    # Extra: report any unexpected columns
    extra = [c for c in df.columns if c not in REQUIRED_SCHEMA_COLUMNS]
    if extra:
        cprint(f"\n  [INFO] Additional columns found ({len(extra)}): {extra[:10]}", "dim")


# ---------------------------------------------------------------------------
# TEST 5 — Data Type Validation
# ---------------------------------------------------------------------------

def test_datatypes(df: pd.DataFrame, tracker: ResultTracker) -> None:
    """TEST 5: Verify correct dtypes for key columns."""
    section_header("TEST 5 — Data Type Validation")

    # Timestamp must be datetime64
    if "Timestamp" in df.columns:
        ts_series = pd.to_datetime(df["Timestamp"], errors="coerce")
        n_valid = ts_series.notna().sum()
        frac = n_valid / max(len(df), 1)
        tracker.record(
            "Timestamp parseable as datetime64",
            passed=frac >= 0.95,
            warn_condition=frac >= 0.80,
            detail=f"{n_valid}/{len(df)} valid ({frac*100:.1f}%)",
        )
    else:
        tracker.record("Timestamp column present for dtype check", passed=False)

    # Station must be object/string
    if "Station" in df.columns:
        # Accept both legacy 'object' dtype and new pandas StringDtype ('str')
        is_str = (
            df["Station"].dtype == object
            or str(df["Station"].dtype) in ("string", "str", "StringDtype")
            or hasattr(df["Station"].dtype, "name") and "string" in str(df["Station"].dtype).lower()
        )
        tracker.record(
            "Station is string/object dtype",
            passed=is_str,
            detail=str(df["Station"].dtype),
        )

    # Latitude / Longitude must be numeric
    for col in ("Latitude", "Longitude"):
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_valid = coerced.notna().sum()
            frac = n_valid / max(len(df), 1)
            tracker.record(
                f"{col} is numeric (float)",
                passed=frac >= 0.90,
                warn_condition=frac >= 0.70,
                detail=f"{n_valid}/{len(df)} valid ({frac*100:.1f}%)",
            )

    # All scientific parameters must be numeric
    sub_header("Scientific Parameter Dtypes")
    sci_cols = [c for c in FLOAT_COLUMNS if c in df.columns]
    for col in sci_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_valid = coerced.notna().sum()
        frac = n_valid / max(len(df), 1)
        tracker.record(
            f"Numeric: {col}",
            passed=frac >= 0.70,
            warn_condition=frac >= 0.50,
            detail=f"{frac*100:.1f}% numeric ({str(df[col].dtype)})",
        )


# ---------------------------------------------------------------------------
# TEST 6 — Missing Value Analysis
# ---------------------------------------------------------------------------

def test_missing_values(df: pd.DataFrame, tracker: ResultTracker) -> Dict[str, float]:
    """
    TEST 6: Column-wise NaN / Null / Infinite value report.

    Returns dict mapping column -> missing fraction.
    """
    section_header("TEST 6 — Missing Values Analysis")

    MAX_NAN_FRACTION = 0.30  # from config validation.max_nan_fraction

    report: Dict[str, float] = {}
    numeric_df = df.select_dtypes(include=[np.number])

    # Check infinites
    n_inf = int(np.isinf(numeric_df.values).sum())
    tracker.record(
        "No infinite values in dataset",
        passed=n_inf == 0,
        warn_condition=n_inf < len(df) * 0.01,
        detail=f"{n_inf} infinite values" if n_inf > 0 else "clean",
    )

    sub_header("Per-Column Missing Value Report")
    cprint(f"\n  {'Column':<30}  {'Missing':>8}  {'Total':>8}  {'Fraction':>9}  Status", "bold")
    cprint(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}", "dim")

    overall_any_critical = False
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        frac = n_miss / max(len(df), 1)
        report[col] = frac

        if frac == 0:
            status, colour = "CLEAN", "green"
        elif frac <= 0.05:
            status, colour = "OK", "green"
        elif frac <= MAX_NAN_FRACTION:
            status, colour = "WARN", "yellow"
        else:
            status, colour = "HIGH", "red"
            overall_any_critical = True

        col_short = col[:29]
        print(f"  {col_short:<30}  {n_miss:>8}  {len(df):>8}  {frac*100:>8.2f}%  ", end="")
        cprint(status, colour)

    print()
    tracker.record(
        "No column exceeds 30% missing values",
        passed=not overall_any_critical,
        warn_condition=False,
        detail=f"Critical columns: {[c for c, f in report.items() if f > MAX_NAN_FRACTION]}",
    )
    return report


# ---------------------------------------------------------------------------
# TEST 7 — Duplicate Detection
# ---------------------------------------------------------------------------

def test_duplicates(df: pd.DataFrame, tracker: ResultTracker) -> None:
    """TEST 7: Check duplicated rows, timestamps, and station records."""
    section_header("TEST 7 — Duplicate Detection")

    MAX_DUP_FRACTION = 0.05

    # Duplicated rows
    n_dup_rows = int(df.duplicated().sum())
    frac_rows = n_dup_rows / max(len(df), 1)
    tracker.record(
        "Duplicated rows",
        passed=frac_rows < MAX_DUP_FRACTION,
        warn_condition=frac_rows < 0.10,
        detail=f"{n_dup_rows} ({frac_rows*100:.2f}%)",
    )

    # Duplicated timestamp + station combos
    if "Timestamp" in df.columns and "Station" in df.columns:
        n_dup_ts = int(df.duplicated(subset=["Timestamp", "Station"]).sum())
        frac_ts = n_dup_ts / max(len(df), 1)
        tracker.record(
            "Duplicated (Timestamp, Station) pairs",
            passed=frac_ts < MAX_DUP_FRACTION,
            warn_condition=frac_ts < 0.10,
            detail=f"{n_dup_ts} ({frac_ts*100:.2f}%)",
        )

    # Station record counts
    if "Station" in df.columns:
        n_stations = df["Station"].nunique()
        tracker.record(
            f"Number of unique stations",
            passed=n_stations > 0,
            detail=f"{n_stations} stations",
        )
        station_counts = df["Station"].value_counts()
        cprint(f"\n  Station record counts (top 10):", "dim")
        for stn, cnt in station_counts.head(10).items():
            cprint(f"    {stn:<12}  {cnt:>6} records", "dim")

    # Timestamp range
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce").dropna()
        if len(ts) > 0:
            cprint(f"\n  Timestamp range: {ts.min()} → {ts.max()}", "dim")
            cprint(f"  Temporal span: {(ts.max() - ts.min()).days} days", "dim")


# ---------------------------------------------------------------------------
# TEST 8 — Outlier Detection
# ---------------------------------------------------------------------------

def test_outliers(df: pd.DataFrame, tracker: ResultTracker) -> None:
    """
    TEST 8: Run IQR, Z-score, and Isolation Forest outlier detection.
    Report counts — does NOT remove or modify data.
    """
    section_header("TEST 8 — Outlier Detection")

    # Columns of interest
    OUTLIER_COLS = [c for c in ["TEC", "foF2", "hmF2", "F10.7", "Kp", "Ap", "Dst", "SSN",
                                "TopsideTEC"] if c in df.columns]

    numeric_sub = df[OUTLIER_COLS].apply(pd.to_numeric, errors="coerce")

    # --- IQR ----------------------------------------------------------------
    sub_header("IQR Outlier Detection (k=2.5)")
    total_iqr = 0
    for col in OUTLIER_COLS:
        series = numeric_sub[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 2.5 * iqr
        hi = q3 + 2.5 * iqr
        n_out = int(((series < lo) | (series > hi)).sum())
        total_iqr += n_out
        frac = n_out / max(len(series), 1)
        colour = "yellow" if frac > 0.05 else "green"
        print(f"  {col:<20}  {n_out:>6} outliers ({frac*100:.2f}%)", end="  ")
        cprint(f"IQR: [{lo:.2f}, {hi:.2f}]", colour)

    tracker.record(
        f"IQR outlier detection completed",
        passed=True,
        detail=f"Total IQR outliers: {total_iqr}",
    )

    # --- Z-score ------------------------------------------------------------
    sub_header("Z-Score Outlier Detection (threshold=3.5)")
    total_z = 0
    for col in OUTLIER_COLS:
        series = numeric_sub[col].dropna()
        if len(series) < 10:
            continue
        mu, sigma = series.mean(), series.std()
        if sigma < 1e-10:
            continue
        z = (series - mu).abs() / sigma
        n_out = int((z > 3.5).sum())
        total_z += n_out
        frac = n_out / max(len(series), 1)
        colour = "yellow" if frac > 0.05 else "green"
        print(f"  {col:<20}  {n_out:>6} outliers ({frac*100:.2f}%)", end="  ")
        cprint(f"μ={mu:.2f}, σ={sigma:.2f}", colour)

    tracker.record(
        "Z-score outlier detection completed",
        passed=True,
        detail=f"Total Z-score outliers: {total_z}",
    )

    # --- Isolation Forest ---------------------------------------------------
    sub_header("Isolation Forest Outlier Detection (contamination=0.05)")
    try:
        from sklearn.ensemble import IsolationForest
        IF_COLS = [c for c in ["TEC", "foF2", "hmF2", "F10.7", "Kp", "Dst"] if c in df.columns]
        sub_if = numeric_sub[IF_COLS].dropna()
        if len(sub_if) >= 50:
            clf = IsolationForest(contamination=0.05, n_estimators=100, random_state=42, n_jobs=-1)
            preds = clf.fit_predict(sub_if.values)
            n_out_if = int((preds == -1).sum())
            frac_if = n_out_if / max(len(sub_if), 1)
            tracker.record(
                "Isolation Forest outlier detection completed",
                passed=True,
                detail=f"{n_out_if} outliers ({frac_if*100:.2f}%) on {IF_COLS}",
            )
        else:
            tracker.record(
                "Isolation Forest outlier detection",
                passed=False,
                warn_condition=True,
                detail=f"Insufficient rows ({len(sub_if)}) after dropna",
            )
    except ImportError:
        tracker.record(
            "Isolation Forest (sklearn available)",
            passed=False,
            warn_condition=True,
            detail="sklearn not installed",
        )
    except Exception as exc:
        tracker.record(
            "Isolation Forest outlier detection",
            passed=False,
            warn_condition=True,
            detail=str(exc),
        )
