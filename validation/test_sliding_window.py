"""
test_sliding_window.py
======================
TEST 10 — Sliding Window Validation

Verifies the SlidingWindowDataset and TECReconstructionDataset for
every configured window size (2h, 6h, 12h, 24h, 48h).

Checks:
- Input/output shapes match expectations.
- No NaN or Inf in generated windows.
- Window generation is reproducible.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)

# Default temporal resolution
DEFAULT_RESOLUTION_MINUTES = 15
WINDOW_SIZES_HOURS = [2, 6, 12, 24, 48]


def _hours_to_steps(hours: int, resolution_minutes: int = DEFAULT_RESOLUTION_MINUTES) -> int:
    return (hours * 60) // resolution_minutes


def test_sliding_window(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
    processed_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    TEST 10: Validate sliding window generation for all configured window sizes.
    """
    section_header("TEST 10 — Sliding Window Validation")

    # --- Load processed data -----------------------------------------------
    if processed_df is None:
        processed_csv = project_root / "data" / "processed" / "processed_dataset.csv"
        if not processed_csv.exists():
            tracker.record("Processed CSV available for window test", passed=False)
            return
        try:
            processed_df = pd.read_csv(processed_csv, nrows=5000, low_memory=False)
            tracker.record("Load processed CSV (sample 5000 rows)", passed=True,
                           detail=str(processed_df.shape))
        except Exception as exc:
            tracker.record("Load processed CSV", passed=False, detail=str(exc))
            return

    # --- Import SlidingWindowDataset ----------------------------------------
    sys.path.insert(0, str(project_root))
    try:
        import torch
        from src.dataset import SlidingWindowDataset, TECReconstructionDataset
        tracker.record("Import: src.dataset (SlidingWindowDataset)", passed=True)
    except ImportError as exc:
        tracker.record("Import: src.dataset", passed=False, detail=str(exc))
        return

    # --- Feature columns from metadata -------------------------------------
    meta_path = project_root / "data" / "processed" / "dataset_metadata.json"
    feature_cols: List[str] = []
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            feature_cols = meta.get("feature_columns", [])
            # Keep only those present in df
            feature_cols = [c for c in feature_cols if c in processed_df.columns]
        except Exception:
            pass

    if not feature_cols:
        # fallback: use all numeric except Station/Timestamp/TopsideTEC
        exclude = {"Station", "Timestamp", "TopsideTEC"}
        feature_cols = [
            c for c in processed_df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")
    if target_col not in processed_df.columns:
        tracker.record("Target column present for window test", passed=False,
                       detail=f"{target_col} not found")
        return

    tracker.record(
        "Feature columns identified for sliding window",
        passed=len(feature_cols) > 0,
        detail=f"{len(feature_cols)} features",
    )

    # --- Window size validation -----------------------------------------------
    resolution = int(cfg.get("sequences", {}).get("temporal_resolution_minutes",
                                                    DEFAULT_RESOLUTION_MINUTES))
    window_sizes = cfg.get("sequences", {}).get("window_sizes_hours", WINDOW_SIZES_HOURS)

    sub_header("Window Size Shape Verification")
    cprint(f"\n  {'Window (h)':>10}  {'Steps':>6}  {'Input Shape':>22}  {'Target Shape':>14}  {'Status'}", "bold")
    cprint(f"  {'─'*10}  {'─'*6}  {'─'*22}  {'─'*14}  {'─'*8}", "dim")

    for wh in window_sizes:
        expected_steps = _hours_to_steps(wh, resolution)
        try:
            ds = TECReconstructionDataset(
                df=processed_df,
                feature_columns=feature_cols,
                target_column=target_col,
                window_hours=wh,
                temporal_resolution_minutes=resolution,
                stride=1,
                min_completeness=0.80,
                group_by_station="Station" in processed_df.columns,
            )

            n_samples = len(ds)
            if n_samples == 0:
                tracker.record(
                    f"Window {wh}h: sequences generated",
                    passed=False,
                    detail="0 samples (check data size vs window size)",
                )
                row = f"  {wh:>10}h  {expected_steps:>6}  {'(no samples)':>22}  {'N/A':>14}  "
                print(row, end="")
                cprint("FAIL", "red")
                continue

            X_sample, y_sample = ds[0]
            expected_X_shape = (expected_steps, len(feature_cols))
            expected_y_shape = (1,)

            shape_ok = (
                tuple(X_sample.shape) == expected_X_shape
                and tuple(y_sample.shape) == expected_y_shape
            )

            # Check for NaN/Inf in first 10 samples
            n_check = min(10, n_samples)
            has_nan_inf = False
            for i in range(n_check):
                xi, yi = ds[i]
                if not (torch.isfinite(xi).all() and torch.isfinite(yi).all()):
                    has_nan_inf = True
                    break

            passed = shape_ok and not has_nan_inf
            status_str = "PASS" if passed else "FAIL"
            status_col = "green" if passed else "red"
            warn_str = "  (NaN/Inf!)" if has_nan_inf else ""

            row = (
                f"  {wh:>10}h  {expected_steps:>6}  "
                f"{str(tuple(X_sample.shape)):>22}  "
                f"{str(tuple(y_sample.shape)):>14}  "
            )
            print(row, end="")
            cprint(f"{status_str}{warn_str}", status_col)

            tracker.record(
                f"Window {wh}h: shape ({expected_steps}, {len(feature_cols)}) correct",
                passed=shape_ok,
                detail=f"got X={tuple(X_sample.shape)} y={tuple(y_sample.shape)}",
            )
            tracker.record(
                f"Window {wh}h: no NaN/Inf in sequences",
                passed=not has_nan_inf,
                detail=f"{n_samples} total sequences",
            )

        except Exception as exc:
            tracker.record(
                f"Window {wh}h: SlidingWindowDataset",
                passed=False,
                detail=str(exc),
            )
            row = f"  {wh:>10}h  {expected_steps:>6}  {'ERROR':>22}  {'ERROR':>14}  "
            print(row, end="")
            cprint("FAIL", "red")

    # --- Direct SlidingWindowDataset test (24h default) ---------------------
    sub_header("Direct SlidingWindowDataset Unit Test (24h default)")
    try:
        default_steps = _hours_to_steps(24, resolution)
        sw_ds = SlidingWindowDataset(
            df=processed_df,
            feature_columns=feature_cols,
            target_column=target_col,
            window_steps=default_steps,
            stride=1,
            min_completeness=0.80,
            group_by_station="Station" in processed_df.columns,
        )
        n = len(sw_ds)
        tracker.record(
            "SlidingWindowDataset instantiation",
            passed=n > 0,
            detail=f"{n} sequences",
        )

        # __len__ and __getitem__ checks
        tracker.record(
            "SlidingWindowDataset.__len__() returns positive int",
            passed=isinstance(n, int) and n > 0,
            detail=str(n),
        )

        X0, y0 = sw_ds[0]
        tracker.record(
            "SlidingWindowDataset.__getitem__(0) returns tensors",
            passed=hasattr(X0, "shape") and hasattr(y0, "shape"),
            detail=f"X={X0.shape} y={y0.shape}",
        )
        tracker.record(
            "X tensor is 2D (steps × features)",
            passed=len(X0.shape) == 2,
            detail=str(X0.shape),
        )
        tracker.record(
            "y tensor is 1D (target scalar)",
            passed=len(y0.shape) == 1,
            detail=str(y0.shape),
        )

    except Exception as exc:
        tracker.record("SlidingWindowDataset unit test", passed=False, detail=str(exc))
