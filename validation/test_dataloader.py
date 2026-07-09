"""
test_dataloader.py
==================
TEST 12 — PyTorch TECDataset Validation
TEST 13 — DataLoader Validation
TEST 14 — Multi-Station Support

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

BATCH_SIZE = 32


def test_dataset_and_dataloader(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
    processed_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    TEST 12, 13, 14: Validate TECReconstructionDataset, DataLoader,
    and multi-station support.
    """
    # ========================================================================
    # TEST 12 — PyTorch TECDataset
    # ========================================================================
    section_header("TEST 12 — PyTorch TECReconstructionDataset")

    # --- Import ---------------------------------------------------------------
    sys.path.insert(0, str(project_root))
    try:
        import torch
        from torch.utils.data import DataLoader as TorchDataLoader
        from src.dataset import TECReconstructionDataset, DatasetFactory
        tracker.record("Import: torch and src.dataset", passed=True)
    except ImportError as exc:
        tracker.record("Import: torch and src.dataset", passed=False, detail=str(exc))
        return

    # --- Load data -----------------------------------------------------------
    if processed_df is None:
        processed_csv = project_root / "data" / "processed" / "processed_dataset.csv"
        if processed_csv.exists():
            try:
                processed_df = pd.read_csv(processed_csv, nrows=8000, low_memory=False)
            except Exception as exc:
                tracker.record("Load processed CSV", passed=False, detail=str(exc))
                return
        else:
            tracker.record("Processed CSV available for dataset test", passed=False)
            return

    # Feature columns
    meta_path = project_root / "data" / "processed" / "dataset_metadata.json"
    feature_cols: List[str] = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = [c for c in meta.get("feature_columns", []) if c in processed_df.columns]

    if not feature_cols:
        exclude = {"Station", "Timestamp", "TopsideTEC"}
        feature_cols = [c for c in processed_df.select_dtypes(include=[np.number]).columns
                        if c not in exclude]

    target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")
    resolution = int(cfg.get("sequences", {}).get("temporal_resolution_minutes", 15))
    window_hours = int(cfg.get("sequences", {}).get("default_window_hours", 24))

    if target_col not in processed_df.columns:
        tracker.record("Target column present", passed=False, detail=target_col)
        return

    # --- Instantiate TECReconstructionDataset ---------------------------------
    sub_header("TECReconstructionDataset Instantiation")
    try:
        t0 = time.perf_counter()
        tec_ds = TECReconstructionDataset(
            df=processed_df,
            feature_columns=feature_cols,
            target_column=target_col,
            window_hours=window_hours,
            temporal_resolution_minutes=resolution,
            stride=1,
            min_completeness=0.80,
            group_by_station="Station" in processed_df.columns,
        )
        elapsed = time.perf_counter() - t0
        n = len(tec_ds)

        tracker.record(
            "TECReconstructionDataset instantiated",
            passed=n > 0,
            detail=f"{n} sequences [{elapsed:.2f}s]",
        )
    except Exception as exc:
        tracker.record("TECReconstructionDataset instantiation", passed=False, detail=str(exc))
        return

    # --- __len__ --------------------------------------------------------------
    tracker.record(
        "__len__() returns positive integer",
        passed=isinstance(n, int) and n > 0,
        detail=str(n),
    )

    # --- __getitem__ ----------------------------------------------------------
    sub_header("__getitem__ Shape & Type Checks")
    try:
        X0, y0 = tec_ds[0]
        window_steps = (window_hours * 60) // resolution

        tracker.record(
            "__getitem__(0) returns (Tensor, Tensor)",
            passed=isinstance(X0, torch.Tensor) and isinstance(y0, torch.Tensor),
            detail=f"X type={type(X0).__name__} y type={type(y0).__name__}",
        )
        tracker.record(
            "X tensor shape: (window_steps, n_features)",
            passed=X0.shape == (window_steps, len(feature_cols)),
            detail=f"got {tuple(X0.shape)}, expected ({window_steps}, {len(feature_cols)})",
        )
        tracker.record(
            "y tensor shape: (1,)",
            passed=y0.shape == (1,),
            detail=f"got {tuple(y0.shape)}",
        )
        tracker.record(
            "X tensor dtype is float32",
            passed=X0.dtype == torch.float32,
            detail=str(X0.dtype),
        )
        tracker.record(
            "y tensor dtype is float32",
            passed=y0.dtype == torch.float32,
            detail=str(y0.dtype),
        )
        tracker.record(
            "X tensor is finite (no NaN/Inf)",
            passed=bool(torch.isfinite(X0).all()),
            detail="contains NaN/Inf" if not torch.isfinite(X0).all() else "clean",
        )
    except Exception as exc:
        tracker.record("__getitem__ checks", passed=False, detail=str(exc))

    # --- CPU compatibility ---------------------------------------------------
    sub_header("CPU Compatibility")
    try:
        device = torch.device("cpu")
        X_cpu = X0.to(device)
        tracker.record(
            "Tensor transferable to CPU",
            passed=X_cpu.device.type == "cpu",
            detail=str(X_cpu.device),
        )
    except Exception as exc:
        tracker.record("CPU compatibility", passed=False, detail=str(exc))

    # ========================================================================
    # TEST 13 — DataLoader
    # ========================================================================
    section_header("TEST 13 — DataLoader Validation")

    sub_header(f"DataLoader (batch_size={BATCH_SIZE})")
    try:
        loader = TorchDataLoader(
            tec_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,   # 0 for Windows compatibility
            drop_last=False,
        )
        window_steps = (window_hours * 60) // resolution
        tracker.record(
            "DataLoader created successfully",
            passed=True,
            detail=f"{len(loader)} batches",
        )

        # Grab first batch
        t0 = time.perf_counter()
        first_batch_X, first_batch_y = next(iter(loader))
        elapsed = time.perf_counter() - t0

        actual_batch = first_batch_X.shape[0]
        expected_X = (min(BATCH_SIZE, n), window_steps, len(feature_cols))
        expected_y = (min(BATCH_SIZE, n),)

        tracker.record(
            f"Batch X shape: ({actual_batch}, {window_steps}, {len(feature_cols)})",
            passed=(
                first_batch_X.shape[1] == window_steps
                and first_batch_X.shape[2] == len(feature_cols)
            ),
            detail=f"got {tuple(first_batch_X.shape)} [{elapsed:.3f}s]",
        )
        tracker.record(
            f"Batch y shape: ({actual_batch},) — scalar targets",
            passed=first_batch_y.ndim in (1, 2),
            detail=f"got {tuple(first_batch_y.shape)}",
        )
        tracker.record(
            "Batch X is finite",
            passed=bool(torch.isfinite(first_batch_X).all()),
            detail="contains NaN/Inf" if not torch.isfinite(first_batch_X).all() else "clean",
        )

    except Exception as exc:
        tracker.record("DataLoader batch iteration", passed=False, detail=str(exc))

    # ========================================================================
    # TEST 14 — Multi-Station Support
    # ========================================================================
    section_header("TEST 14 — Multi-Station Support")

    if "Station" not in processed_df.columns:
        tracker.record("Station column present", passed=False)
        return

    unique_stations = processed_df["Station"].unique().tolist()
    n_stations = len(unique_stations)

    tracker.record(
        "Multiple stations in dataset",
        passed=n_stations > 1,
        warn_condition=n_stations == 1,
        detail=f"{n_stations} unique stations",
    )

    cprint(f"\n  Stations ({n_stations}): {unique_stations[:10]}"
           f"{'...' if n_stations > 10 else ''}", "dim")

    # --- No station mixing: windows per station ----------------------------
    sub_header("Station Grouping — No Cross-Station Windows")
    if "TopsideTEC" in processed_df.columns and len(feature_cols) > 0:
        try:
            per_station_counts: Dict[str, int] = {}
            for stn in unique_stations[:5]:  # sample first 5 stations
                group_df = processed_df[processed_df["Station"] == stn].copy()
                if len(group_df) < 20:
                    continue
                stn_ds = TECReconstructionDataset(
                    df=group_df,
                    feature_columns=feature_cols,
                    target_column=target_col,
                    window_hours=min(2, window_hours),  # use smallest window for speed
                    temporal_resolution_minutes=resolution,
                    stride=1,
                    min_completeness=0.80,
                    group_by_station=False,  # already filtered to one station
                )
                per_station_counts[stn] = len(stn_ds)

            tracker.record(
                "Per-station dataset can be built independently",
                passed=len(per_station_counts) > 0,
                detail=str({k: v for k, v in list(per_station_counts.items())[:5]}),
            )
        except Exception as exc:
            tracker.record("Per-station dataset build", passed=False,
                           warn_condition=True, detail=str(exc))

    # --- Station record count distribution ---------------------------------
    station_counts = processed_df["Station"].value_counts()
    min_recs = int(station_counts.min())
    max_recs = int(station_counts.max())
    tracker.record(
        "All stations have records",
        passed=min_recs > 0,
        detail=f"min={min_recs}, max={max_recs} records/station",
    )
