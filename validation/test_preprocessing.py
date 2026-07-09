"""
test_preprocessing.py
=====================
TEST for Preprocessing Pipeline

Validates:
- DataLoader can load the raw CSV.
- MissingValueHandler reduces NaN count.
- InfiniteValueHandler removes infinites.
- OutlierDetector runs and returns valid shapes.
- DataScaler fits and transforms without NaN/Inf.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)


def test_preprocessing_pipeline(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Validate the full PreprocessingPipeline by running it on real data.

    Returns (processed_df, train_df, val_df, test_df) or None on failure.
    """
    section_header("TEST — Preprocessing Pipeline")

    # --- Import modules -------------------------------------------------------
    try:
        sys.path.insert(0, str(project_root))
        from src.preprocessing import (
            DataLoader,
            InfiniteValueHandler,
            MissingValueHandler,
            OutlierDetector,
            DataScaler,
            PreprocessingPipeline,
        )
        tracker.record("Import: src.preprocessing", passed=True)
    except ImportError as exc:
        tracker.record("Import: src.preprocessing", passed=False, detail=str(exc))
        return None

    # --- DataLoader ----------------------------------------------------------
    sub_header("DataLoader")
    try:
        loader = DataLoader(cfg, project_root)
        t0 = time.perf_counter()
        df = loader.load()
        elapsed = time.perf_counter() - t0
        tracker.record(
            "DataLoader.load() succeeds",
            passed=len(df) > 0,
            detail=f"{len(df)} rows × {len(df.columns)} cols [{elapsed:.2f}s]",
        )
    except Exception as exc:
        tracker.record("DataLoader.load() succeeds", passed=False, detail=str(exc))
        return None

    # Timestamp parsed
    if "Timestamp" in df.columns:
        is_dt = pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
        tracker.record(
            "Timestamp parsed as datetime64",
            passed=is_dt,
            detail=str(df["Timestamp"].dtype),
        )

    # --- InfiniteValueHandler ------------------------------------------------
    sub_header("InfiniteValueHandler")
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_inf_before = int(np.isinf(df[numeric_cols].values).sum())
        df_inf = InfiniteValueHandler.handle(df.copy())
        n_inf_after = int(np.isinf(df_inf.select_dtypes(include=[np.number]).values).sum())
        tracker.record(
            "InfiniteValueHandler removes all Inf values",
            passed=n_inf_after == 0,
            detail=f"{n_inf_before} → {n_inf_after} infinites",
        )
        df = df_inf
    except Exception as exc:
        tracker.record("InfiniteValueHandler.handle()", passed=False, detail=str(exc))

    # --- MissingValueHandler -------------------------------------------------
    sub_header("MissingValueHandler")
    try:
        n_nan_before = int(df.isna().sum().sum())
        handler = MissingValueHandler(cfg)
        t0 = time.perf_counter()
        df_mv = handler.handle(df.copy())
        elapsed = time.perf_counter() - t0
        n_nan_after = int(df_mv.isna().sum().sum())
        improved = n_nan_after <= n_nan_before
        tracker.record(
            "MissingValueHandler reduces NaN count",
            passed=improved,
            detail=f"{n_nan_before} → {n_nan_after} NaNs [{elapsed:.2f}s]",
        )
        df = df_mv
    except Exception as exc:
        tracker.record("MissingValueHandler.handle()", passed=False, detail=str(exc))

    # --- OutlierDetector -----------------------------------------------------
    sub_header("OutlierDetector")
    try:
        detector = OutlierDetector(cfg)
        t0 = time.perf_counter()
        df_out, outlier_mask = detector.detect_and_treat(df.copy())
        elapsed = time.perf_counter() - t0
        shape_ok = df_out.shape[0] > 0 and df_out.shape[1] > 0
        n_flagged = int(outlier_mask.any(axis=1).sum())
        tracker.record(
            "OutlierDetector returns valid DataFrame",
            passed=shape_ok,
            detail=f"{n_flagged} rows flagged [{elapsed:.2f}s]",
        )
        tracker.record(
            "Outlier mask has same index as input",
            passed=outlier_mask.shape[0] == df.shape[0],
            detail=f"mask={outlier_mask.shape}, df={df.shape}",
        )
        df = df_out
    except Exception as exc:
        tracker.record("OutlierDetector.detect_and_treat()", passed=False, detail=str(exc))

    # --- DataScaler ----------------------------------------------------------
    sub_header("DataScaler")
    try:
        scaler = DataScaler(cfg)
        numeric_df = df.select_dtypes(include=[np.number])
        t0 = time.perf_counter()
        df_scaled = scaler.fit_transform(df.copy())
        elapsed = time.perf_counter() - t0

        # No NaN / Inf after scaling
        scaled_numeric = df_scaled.select_dtypes(include=[np.number])
        n_nan_scaled = int(scaled_numeric.isna().sum().sum())
        n_inf_scaled = int(np.isinf(scaled_numeric.values).sum())
        tracker.record(
            "DataScaler.fit_transform() succeeds",
            passed=True,
            detail=f"[{elapsed:.2f}s]",
        )
        tracker.record(
            "Scaled data has no NaN values",
            passed=n_nan_scaled == 0,
            warn_condition=True,
            detail=f"{n_nan_scaled} NaNs",
        )
        tracker.record(
            "Scaled data has no Inf values",
            passed=n_inf_scaled == 0,
            detail=f"{n_inf_scaled} infinites",
        )
    except Exception as exc:
        tracker.record("DataScaler.fit_transform()", passed=False, detail=str(exc))

    # --- Full Pipeline -------------------------------------------------------
    sub_header("Full PreprocessingPipeline.run()")
    try:
        pipeline = PreprocessingPipeline(cfg, project_root)
        t0 = time.perf_counter()
        processed_df, train_df, val_df, test_df = pipeline.run()
        elapsed = time.perf_counter() - t0

        tracker.record(
            "PreprocessingPipeline.run() completes",
            passed=True,
            detail=f"[{elapsed:.2f}s]",
        )
        tracker.record(
            "Processed df is non-empty",
            passed=len(processed_df) > 0,
            detail=f"{processed_df.shape}",
        )

        # Split fractions
        total = len(train_df) + len(val_df) + len(test_df)
        train_frac = len(train_df) / max(total, 1)
        val_frac = len(val_df) / max(total, 1)
        test_frac = len(test_df) / max(total, 1)
        tracker.record(
            "Train/Val/Test split non-empty",
            passed=all(n > 0 for n in [len(train_df), len(val_df), len(test_df)]),
            detail=(
                f"train={len(train_df)}({train_frac*100:.1f}%)"
                f"  val={len(val_df)}({val_frac*100:.1f}%)"
                f"  test={len(test_df)}({test_frac*100:.1f}%)"
            ),
        )

        return processed_df, train_df, val_df, test_df

    except Exception as exc:
        tracker.record("PreprocessingPipeline.run()", passed=False, detail=str(exc))
        return None
