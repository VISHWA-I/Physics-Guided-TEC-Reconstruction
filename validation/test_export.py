"""
test_export.py
==============
TEST 16 — Export Validation

Verifies that all processed output files exist, can be opened,
and contain non-empty, valid data:

  data/processed/processed_dataset.csv
  data/processed/numpy/train_X.npy, train_y.npy, val_X.npy, val_y.npy, test_X.npy, test_y.npy
  data/processed/tensors/train_X.pt, train_y.pt, val_X.pt, val_y.pt, test_X.pt, test_y.pt
  data/processed/dataset_metadata.json
  data/processed/feature_names.json

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)


def test_export(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
) -> None:
    """TEST 16: Validate all exported data files."""
    section_header("TEST 16 — Export File Validation")

    proc_dir = project_root / "data" / "processed"
    numpy_dir = proc_dir / "numpy"
    tensor_dir = proc_dir / "tensors"

    # -----------------------------------------------------------------------
    # Processed CSV
    # -----------------------------------------------------------------------
    sub_header("Processed CSV")
    proc_csv = proc_dir / "processed_dataset.csv"
    _check_file_exists_and_readable(proc_csv, "processed_dataset.csv", tracker)
    if proc_csv.exists() and proc_csv.stat().st_size > 0:
        try:
            df = pd.read_csv(proc_csv, nrows=5, low_memory=False)
            tracker.record(
                "processed_dataset.csv readable as CSV",
                passed=True,
                detail=f"{df.shape[1]} columns",
            )
        except Exception as exc:
            tracker.record("processed_dataset.csv readable as CSV", passed=False, detail=str(exc))

    # Train/val/test CSVs
    for split in ("train", "val", "test"):
        csv_path = proc_dir / f"{split}.csv"
        _check_file_exists_and_readable(csv_path, f"{split}.csv", tracker)

    # -----------------------------------------------------------------------
    # NumPy arrays
    # -----------------------------------------------------------------------
    sub_header("NumPy Arrays (.npy)")
    numpy_files = {
        "train_X.npy": ("train", "X"),
        "train_y.npy": ("train", "y"),
        "val_X.npy":   ("val",   "X"),
        "val_y.npy":   ("val",   "y"),
        "test_X.npy":  ("test",  "X"),
        "test_y.npy":  ("test",  "y"),
    }
    numpy_shapes: Dict[str, tuple] = {}
    for fname, (split, kind) in numpy_files.items():
        fpath = numpy_dir / fname
        exists = fpath.exists()
        tracker.record(
            f"NumPy: {fname} exists",
            passed=exists,
            detail=f"{fpath.stat().st_size / 1024:.1f} KB" if exists else "MISSING",
        )
        if not exists:
            continue
        try:
            arr = np.load(fpath, allow_pickle=False)
            tracker.record(
                f"NumPy: {fname} loadable",
                passed=True,
                detail=f"shape={arr.shape} dtype={arr.dtype}",
            )
            tracker.record(
                f"NumPy: {fname} non-empty",
                passed=arr.size > 0,
                detail=f"{arr.size} elements",
            )
            # No NaN in feature arrays
            if kind == "X":
                n_nan = int(np.isnan(arr).sum())
                n_inf = int(np.isinf(arr).sum())
                tracker.record(
                    f"NumPy: {fname} finite values",
                    passed=n_nan == 0 and n_inf == 0,
                    warn_condition=True,
                    detail=f"nan={n_nan}, inf={n_inf}",
                )
                # Expected 3D: (N, W, F)
                tracker.record(
                    f"NumPy: {fname} is 3D (N, W, F)",
                    passed=arr.ndim == 3,
                    detail=f"ndim={arr.ndim}, shape={arr.shape}",
                )
            elif kind == "y":
                # Expected 1D or 2D: (N,) or (N, 1)
                tracker.record(
                    f"NumPy: {fname} is 1D or 2D",
                    passed=arr.ndim in (1, 2),
                    detail=f"shape={arr.shape}",
                )
            numpy_shapes[fname] = arr.shape
        except Exception as exc:
            tracker.record(f"NumPy: {fname} loadable", passed=False, detail=str(exc))

    # Consistency: train_X rows == train_y rows
    sub_header("NumPy Shape Consistency Checks")
    for split in ("train", "val", "test"):
        x_key = f"{split}_X.npy"
        y_key = f"{split}_y.npy"
        if x_key in numpy_shapes and y_key in numpy_shapes:
            n_x = numpy_shapes[x_key][0]
            n_y = numpy_shapes[y_key][0]
            tracker.record(
                f"NumPy: {split} X and y row counts match",
                passed=n_x == n_y,
                detail=f"X={n_x}, y={n_y}",
            )

    # -----------------------------------------------------------------------
    # PyTorch Tensors
    # -----------------------------------------------------------------------
    sub_header("PyTorch Tensors (.pt)")
    try:
        import torch
        pt_files = {
            "train_X.pt": ("train", "X"),
            "train_y.pt": ("train", "y"),
            "val_X.pt":   ("val",   "X"),
            "val_y.pt":   ("val",   "y"),
            "test_X.pt":  ("test",  "X"),
            "test_y.pt":  ("test",  "y"),
        }
        pt_shapes: Dict[str, tuple] = {}
        for fname, (split, kind) in pt_files.items():
            fpath = tensor_dir / fname
            exists = fpath.exists()
            tracker.record(
                f"Tensor: {fname} exists",
                passed=exists,
                detail=f"{fpath.stat().st_size / 1024:.1f} KB" if exists else "MISSING",
            )
            if not exists:
                continue
            try:
                tensor = torch.load(fpath, map_location="cpu", weights_only=True)
                tracker.record(
                    f"Tensor: {fname} loadable",
                    passed=True,
                    detail=f"shape={tuple(tensor.shape)} dtype={tensor.dtype}",
                )
                tracker.record(
                    f"Tensor: {fname} finite",
                    passed=bool(torch.isfinite(tensor).all()),
                    warn_condition=True,
                    detail="contains NaN/Inf" if not torch.isfinite(tensor).all() else "clean",
                )
                pt_shapes[fname] = tuple(tensor.shape)
            except Exception as exc:
                tracker.record(f"Tensor: {fname} loadable", passed=False, detail=str(exc))

        # PT / NPY shape consistency
        sub_header("PT vs NPY Shape Consistency")
        for split in ("train", "val", "test"):
            for kind in ("X", "y"):
                npy_key = f"{split}_{kind}.npy"
                pt_key = f"{split}_{kind}.pt"
                if npy_key in numpy_shapes and pt_key in pt_shapes:
                    tracker.record(
                        f"{split}_{kind}: PT shape matches NPY shape",
                        passed=pt_shapes[pt_key] == numpy_shapes[npy_key],
                        detail=f"PT={pt_shapes[pt_key]}, NPY={numpy_shapes[npy_key]}",
                    )

    except ImportError:
        tracker.record("PyTorch available for tensor loading", passed=False,
                       warn_condition=True, detail="torch not installed")

    # -----------------------------------------------------------------------
    # JSON Metadata
    # -----------------------------------------------------------------------
    sub_header("JSON Metadata Files")
    for json_file in ("dataset_metadata.json", "feature_names.json"):
        json_path = proc_dir / json_file
        exists = json_path.exists()
        tracker.record(
            f"JSON: {json_file} exists",
            passed=exists,
            detail=str(json_path) if not exists else "found",
        )
        if exists:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                tracker.record(
                    f"JSON: {json_file} valid JSON",
                    passed=isinstance(data, (dict, list)),
                    detail=f"type={type(data).__name__}",
                )
                if json_file == "dataset_metadata.json":
                    # Check for key fields
                    for key in ("n_features", "feature_columns", "target_column"):
                        tracker.record(
                            f"Metadata: '{key}' key present",
                            passed=key in data,
                            detail=str(data.get(key, "MISSING"))[:60],
                        )
            except Exception as exc:
                tracker.record(f"JSON: {json_file} valid", passed=False, detail=str(exc))


def _check_file_exists_and_readable(
    path: Path,
    label: str,
    tracker: ResultTracker,
) -> None:
    """Helper: check file exists and is non-empty."""
    exists = path.exists()
    tracker.record(
        f"{label} exists",
        passed=exists,
        detail=str(path) if not exists else f"{path.stat().st_size / 1024:.1f} KB",
    )
    if exists:
        tracker.record(
            f"{label} is non-empty",
            passed=path.stat().st_size > 0,
            detail=f"{path.stat().st_size} bytes",
        )
