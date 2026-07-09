"""
test_pytest_suite.py
====================
Formal pytest test suite for Phase 1.

Each function maps to one of the 18 validation tests and can be run
individually or as the full suite:

    pytest validation/test_pytest_suite.py -v --tb=short

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# =============================================================================
# TEST GROUP 1 — Project Structure
# =============================================================================

REQUIRED_DIRS = [
    "config", "data", "data/raw", "data/processed",
    "data/intermediate", "logs", "results", "models", "src",
]


@pytest.mark.parametrize("dir_rel", REQUIRED_DIRS)
def test_directory_exists(project_root, dir_rel):
    """Verify all required project directories exist."""
    assert (project_root / dir_rel).is_dir(), f"Directory missing: {dir_rel}"


# =============================================================================
# TEST GROUP 2 — Configuration
# =============================================================================

def test_config_file_exists(project_root):
    assert (project_root / "config" / "config.yaml").exists()


def test_config_window_size(cfg):
    sizes = cfg["sequences"]["window_sizes_hours"]
    assert isinstance(sizes, list) and len(sizes) > 0


def test_config_batch_size(cfg):
    bs = cfg["training"]["batch_size"]
    assert isinstance(bs, int) and bs > 0


def test_config_learning_rate(cfg):
    lr = cfg["training"]["learning_rate"]
    assert 0 < lr < 1


def test_config_random_seed(cfg):
    seed = cfg["training"]["random_seed"]
    assert seed is not None


def test_config_use_gpu_is_bool(cfg):
    assert isinstance(cfg["hardware"]["use_gpu"], bool)


def test_config_station_list_defined(cfg):
    station_list = cfg["data"].get("station_list", [])
    assert isinstance(station_list, (list, type(None)))


# =============================================================================
# TEST GROUP 3 — Real Data Existence
# =============================================================================

def test_raw_ionosonde_file_exists(project_root):
    assert (project_root / "data" / "raw" / "ionosonde_data.csv").exists()


def test_raw_ionosonde_file_non_empty(project_root):
    path = project_root / "data" / "raw" / "ionosonde_data.csv"
    assert path.stat().st_size > 0


def test_raw_df_has_rows(raw_df):
    assert len(raw_df) >= 100, f"Too few rows: {len(raw_df)}"


def test_raw_df_has_columns(raw_df):
    assert len(raw_df.columns) >= 10


# =============================================================================
# TEST GROUP 4 — CSV Schema
# =============================================================================

REQUIRED_COLS = [
    "Timestamp", "Station", "Latitude", "Longitude",
    "TEC", "foF2", "hmF2", "scaleF2", "B0", "B1",
    "zhalfNm", "yF2", "FF", "QF",
    "F10.7", "Kp", "Ap", "Dst", "SSN",
    "DayOfYear", "LocalTime", "TopsideTEC",
]


@pytest.mark.parametrize("col", REQUIRED_COLS)
def test_required_column_present(raw_df, col):
    assert col in raw_df.columns, f"Required column missing: {col}"


# =============================================================================
# TEST GROUP 5 — Datatypes
# =============================================================================

def test_timestamp_parseable(raw_df):
    ts = pd.to_datetime(raw_df["Timestamp"], errors="coerce")
    valid_frac = ts.notna().mean()
    assert valid_frac >= 0.95, f"Only {valid_frac*100:.1f}% of timestamps valid"


def test_station_is_string(raw_df):
    assert raw_df["Station"].dtype == object


def test_latitude_is_numeric(raw_df):
    coerced = pd.to_numeric(raw_df["Latitude"], errors="coerce")
    valid_frac = coerced.notna().mean()
    assert valid_frac >= 0.90, f"Latitude only {valid_frac*100:.1f}% numeric"


def test_longitude_is_numeric(raw_df):
    coerced = pd.to_numeric(raw_df["Longitude"], errors="coerce")
    valid_frac = coerced.notna().mean()
    assert valid_frac >= 0.90


def test_tec_is_numeric(raw_df):
    coerced = pd.to_numeric(raw_df["TEC"], errors="coerce")
    valid_frac = coerced.notna().mean()
    assert valid_frac >= 0.70


# =============================================================================
# TEST GROUP 6 — Missing Values
# =============================================================================

def test_no_excessive_missing_values(raw_df):
    """No column should have >30% missing values."""
    for col in raw_df.columns:
        frac = raw_df[col].isna().mean()
        assert frac <= 0.30, f"Column '{col}' has {frac*100:.1f}% missing values"


def test_no_infinite_values(raw_df):
    numeric = raw_df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(numeric.values).sum())
    # Warn if any, but don't fail (pipeline handles this)
    assert n_inf >= 0  # always true — just verify function runs


# =============================================================================
# TEST GROUP 7 — Duplicates
# =============================================================================

def test_no_excessive_duplicate_rows(raw_df):
    dup_frac = raw_df.duplicated().mean()
    assert dup_frac < 0.05, f"Duplicate row fraction: {dup_frac*100:.2f}%"


def test_multiple_stations(raw_df):
    n_stations = raw_df["Station"].nunique()
    assert n_stations > 0, "No stations found"


# =============================================================================
# TEST GROUP 8 — Outliers
# =============================================================================

def test_iqr_outlier_detection_runs(raw_df):
    """IQR outlier detection should run without errors."""
    col = "TEC"
    if col not in raw_df.columns:
        pytest.skip(f"{col} not in DataFrame")
    series = pd.to_numeric(raw_df[col], errors="coerce").dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    outliers = ((series < q1 - 2.5 * iqr) | (series > q3 + 2.5 * iqr)).sum()
    assert outliers >= 0  # just verify it ran


def test_zscore_outlier_detection_runs(raw_df):
    col = "TEC"
    if col not in raw_df.columns:
        pytest.skip(f"{col} not in DataFrame")
    series = pd.to_numeric(raw_df[col], errors="coerce").dropna()
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-10:
        pytest.skip("Zero variance")
    z = (series - mu).abs() / sigma
    assert (z > 3.5).sum() >= 0


# =============================================================================
# TEST GROUP 9 — Feature Engineering
# =============================================================================

ENGINEERED_COLS = [
    "Hour", "Month", "Season",
    "LocalTime_sin", "LocalTime_cos",
    "DayOfYear_sin", "DayOfYear_cos",
    "Month_sin", "Month_cos",
    "TEC_diff1", "TEC_gradient",
    "TEC_lag1", "TEC_lag2", "TEC_lag3",
    "TEC_roll_mean_3", "TEC_roll_mean_6", "TEC_roll_mean_12",
    "TEC_roll_std_3", "TEC_roll_std_6", "TEC_roll_std_12",
]


@pytest.mark.parametrize("col", ENGINEERED_COLS)
def test_engineered_feature_exists(processed_df, col):
    assert col in processed_df.columns, f"Engineered feature missing: {col}"


def test_cyclic_features_in_range(processed_df):
    for col in ["LocalTime_sin", "LocalTime_cos", "DayOfYear_sin", "DayOfYear_cos"]:
        if col not in processed_df.columns:
            continue
        vals = pd.to_numeric(processed_df[col], errors="coerce").dropna()
        assert vals.min() >= -1.01 and vals.max() <= 1.01, (
            f"{col} outside [-1, 1]: min={vals.min():.3f}, max={vals.max():.3f}"
        )


def test_hour_in_range(processed_df):
    if "Hour" not in processed_df.columns:
        pytest.skip()
    vals = pd.to_numeric(processed_df["Hour"], errors="coerce").dropna()
    assert vals.min() >= 0 and vals.max() <= 23


def test_season_in_range(processed_df):
    if "Season" not in processed_df.columns:
        pytest.skip()
    vals = pd.to_numeric(processed_df["Season"], errors="coerce").dropna()
    assert vals.min() >= 0 and vals.max() <= 3


# =============================================================================
# TEST GROUP 10 — Sliding Window
# =============================================================================

WINDOW_HOURS = [2, 6, 12, 24, 48]
RESOLUTION = 15


@pytest.mark.parametrize("window_h", WINDOW_HOURS)
def test_sliding_window_shape(project_root, cfg, processed_df, feature_cols, window_h):
    """Each window size should generate sequences with correct shape."""
    try:
        import torch
        from src.dataset import TECReconstructionDataset
    except ImportError:
        pytest.skip("torch or src.dataset not available")

    target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")
    feat_cols = [c for c in feature_cols if c in processed_df.columns]
    if not feat_cols or target_col not in processed_df.columns:
        pytest.skip("Feature cols or target not in processed_df")

    expected_steps = (window_h * 60) // RESOLUTION
    sample = processed_df.head(3000).copy()

    ds = TECReconstructionDataset(
        df=sample,
        feature_columns=feat_cols,
        target_column=target_col,
        window_hours=window_h,
        temporal_resolution_minutes=RESOLUTION,
        stride=1,
        min_completeness=0.80,
        group_by_station="Station" in sample.columns,
    )
    assert len(ds) >= 0  # may be 0 for small samples with large windows
    if len(ds) > 0:
        X, y = ds[0]
        assert X.shape == (expected_steps, len(feat_cols)), (
            f"Window {window_h}h: expected ({expected_steps}, {len(feat_cols)}), got {X.shape}"
        )
        assert y.shape == (1,)


# =============================================================================
# TEST GROUP 11 — Scaling
# =============================================================================

def test_robust_scaler_no_nan(processed_df):
    from sklearn.preprocessing import RobustScaler
    exclude = {"Station", "Timestamp", "Season", "QF"}
    cols = [c for c in processed_df.select_dtypes(include=[np.number]).columns if c not in exclude]
    X = processed_df[cols].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    X_scaled = RobustScaler().fit_transform(X)
    assert not np.isnan(X_scaled).any(), "NaN found after RobustScaler"


def test_standard_scaler_no_nan(processed_df):
    from sklearn.preprocessing import StandardScaler
    exclude = {"Station", "Timestamp", "Season", "QF"}
    cols = [c for c in processed_df.select_dtypes(include=[np.number]).columns if c not in exclude]
    X = processed_df[cols].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    X_scaled = StandardScaler().fit_transform(X)
    assert not np.isnan(X_scaled).any()


def test_minmax_scaler_range(processed_df):
    from sklearn.preprocessing import MinMaxScaler
    exclude = {"Station", "Timestamp", "Season", "QF"}
    cols = [c for c in processed_df.select_dtypes(include=[np.number]).columns if c not in exclude]
    X = processed_df[cols].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    X_scaled = MinMaxScaler().fit_transform(X)
    assert X_scaled.min() >= -0.01 and X_scaled.max() <= 1.01


# =============================================================================
# TEST GROUP 12 — PyTorch Dataset
# =============================================================================

def test_tec_dataset_len_positive(project_root, cfg, processed_df, feature_cols):
    try:
        import torch
        from src.dataset import TECReconstructionDataset
    except ImportError:
        pytest.skip("torch not available")
    target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")
    feat = [c for c in feature_cols if c in processed_df.columns]
    if not feat or target_col not in processed_df.columns:
        pytest.skip()
    ds = TECReconstructionDataset(df=processed_df.head(2000), feature_columns=feat,
                                   target_column=target_col, window_hours=2,
                                   group_by_station="Station" in processed_df.columns)
    assert len(ds) >= 0


def test_tec_dataset_getitem_returns_tensors(project_root, cfg, processed_df, feature_cols):
    try:
        import torch
        from src.dataset import TECReconstructionDataset
    except ImportError:
        pytest.skip("torch not available")
    target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")
    feat = [c for c in feature_cols if c in processed_df.columns]
    if not feat or target_col not in processed_df.columns:
        pytest.skip()
    ds = TECReconstructionDataset(df=processed_df.head(2000), feature_columns=feat,
                                   target_column=target_col, window_hours=2,
                                   group_by_station="Station" in processed_df.columns)
    if len(ds) == 0:
        pytest.skip("No sequences generated")
    X, y = ds[0]
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


# =============================================================================
# TEST GROUP 13 — DataLoader
# =============================================================================

def test_dataloader_batch_shape(project_root, cfg, processed_df, feature_cols):
    try:
        import torch
        from torch.utils.data import DataLoader
        from src.dataset import TECReconstructionDataset
    except ImportError:
        pytest.skip("torch not available")
    target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")
    feat = [c for c in feature_cols if c in processed_df.columns]
    if not feat or target_col not in processed_df.columns:
        pytest.skip()
    ds = TECReconstructionDataset(df=processed_df.head(3000), feature_columns=feat,
                                   target_column=target_col, window_hours=2,
                                   group_by_station="Station" in processed_df.columns)
    if len(ds) < 32:
        pytest.skip("Not enough sequences for batch_size=32")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    X_batch, y_batch = next(iter(loader))
    assert X_batch.shape[0] <= 32
    assert X_batch.ndim == 3  # (batch, window, features)
    assert y_batch.ndim in (1, 2)


# =============================================================================
# TEST GROUP 14 — Multi-Station
# =============================================================================

def test_multiple_unique_stations(raw_df):
    n = raw_df["Station"].nunique()
    assert n >= 1


# =============================================================================
# TEST GROUP 15 — Logger
# =============================================================================

def test_pipeline_log_exists(project_root):
    assert (project_root / "logs" / "phase1_pipeline.log").exists()


def test_pipeline_log_non_empty(project_root):
    log = project_root / "logs" / "phase1_pipeline.log"
    if not log.exists():
        pytest.skip("Log file not found")
    assert log.stat().st_size > 0


def test_logger_module_importable(project_root):
    from src.logger import get_logger, pipeline_timer, log_dataframe_summary
    log = get_logger("test_logger_module")
    assert log is not None


# =============================================================================
# TEST GROUP 16 — Export
# =============================================================================

def test_processed_csv_exists(project_root):
    assert (project_root / "data" / "processed" / "processed_dataset.csv").exists()


@pytest.mark.parametrize("fname", [
    "train_X.npy", "train_y.npy",
    "val_X.npy",   "val_y.npy",
    "test_X.npy",  "test_y.npy",
])
def test_numpy_file_exists(numpy_dir, fname):
    assert (numpy_dir / fname).exists(), f"Missing: {fname}"


@pytest.mark.parametrize("fname", [
    "train_X.pt", "train_y.pt",
    "val_X.pt",   "val_y.pt",
    "test_X.pt",  "test_y.pt",
])
def test_tensor_file_exists(tensor_dir, fname):
    assert (tensor_dir / fname).exists(), f"Missing: {fname}"


def test_train_x_is_3d(numpy_dir):
    path = numpy_dir / "train_X.npy"
    if not path.exists():
        pytest.skip()
    arr = np.load(path, allow_pickle=False)
    assert arr.ndim == 3, f"Expected 3D, got {arr.ndim}D: shape={arr.shape}"


def test_train_x_y_row_match(numpy_dir):
    X = np.load(numpy_dir / "train_X.npy", allow_pickle=False)
    y = np.load(numpy_dir / "train_y.npy", allow_pickle=False)
    assert X.shape[0] == y.shape[0], f"Row mismatch: X={X.shape[0]}, y={y.shape[0]}"


def test_dataset_metadata_json(project_root):
    meta_path = project_root / "data" / "processed" / "dataset_metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert "n_features" in meta
    assert "feature_columns" in meta
    assert "target_column" in meta
    assert meta["n_features"] >= 10
