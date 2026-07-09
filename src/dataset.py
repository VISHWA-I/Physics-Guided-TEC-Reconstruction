"""
dataset.py
==========
PyTorch Dataset classes for the Mamba-TKAN TEC Reconstruction project.

Classes
-------
* ``SlidingWindowDataset``   - Generates overlapping temporal sequences from
                               a processed DataFrame using a sliding window.
* ``TECReconstructionDataset`` - High-level dataset that wraps a split-specific
                               DataFrame (train / val / test) and exposes a
                               PyTorch-compatible ``__getitem__`` / ``__len__``.
* ``DatasetFactory``         - Builds train, val, and test datasets + DataLoaders
                               from a fully-processed and feature-engineered
                               DataFrame in a single call.

Sequence shape convention
-------------------------
    X : (window_steps, n_features)   - input feature sequence
    y : (1,)                         - scalar TopsideTEC target

Supports
--------
* GPU tensor allocation (``pin_memory`` for efficient host->device transfer).
* CPU or CUDA device placement.
* Lazy loading (X/y stored as float32 NumPy arrays; tensor cast on __getitem__).
* Multiple window sizes (configured via ``config.yaml``).

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Lazy PyTorch import to allow use on machines without torch for pre-processing
try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.constants import (
    COLUMN_STATION,
    COLUMN_TIMESTAMP,
    COLUMN_TOPSIDE_TEC,
    DEFAULT_TEMPORAL_RESOLUTION_MINUTES,
    WINDOW_SIZE_HOURS,
)
from src.logger import get_logger, log_dataframe_summary, pipeline_timer
from src.utils import check_torch_device, compute_window_steps

_log = get_logger(__name__)


def _require_torch() -> None:
    """Raise ImportError with a helpful message if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for dataset creation. "
            "Install it with: pip install torch"
        )


# =============================================================================
# SlidingWindowDataset
# =============================================================================


class SlidingWindowDataset:
    """
    Converts a time-series DataFrame into overlapping fixed-length sequences
    using a sliding window approach.

    The dataset stores windows as NumPy arrays (float32) for memory efficiency.
    Conversion to ``torch.Tensor`` is done lazily in ``__getitem__``.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed and feature-engineered DataFrame. Must contain the target
        column (``TopsideTEC``).
    feature_columns : list[str]
        Ordered list of input feature column names.
    target_column : str
        Name of the target column.
    window_steps : int
        Number of time steps per input sequence.
    stride : int
        Step size between consecutive windows (default 1 = fully overlapping).
    min_completeness : float
        Minimum fraction of non-NaN values required in a window for it to be
        included (default 0.80).
    device : torch.device | None
        Target device.  If None, uses CPU.
    group_by_station : bool
        If True, sliding windows are generated independently per station to
        avoid temporal leakage across disjoint geographic locations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = COLUMN_TOPSIDE_TEC,
        window_steps: int = 96,          # 24h at 15-min resolution
        stride: int = 1,
        min_completeness: float = 0.80,
        device: Optional[Any] = None,
        group_by_station: bool = True,
    ) -> None:
        _require_torch()

        self._feature_cols: List[str] = feature_columns
        self._target_col: str = target_column
        self._window_steps: int = window_steps
        self._stride: int = max(1, stride)
        self._min_completeness: float = min_completeness
        self._device: Any = device or torch.device("cpu")
        self._group_by_station: bool = group_by_station

        # Build the index array and raw data arrays
        with pipeline_timer("SlidingWindowDataset.__init__", _log):
            self._X, self._y = self._build_sequences(df)

        _log.info(
            "SlidingWindowDataset ready: n_sequences=%d  window_steps=%d  n_features=%d",
            len(self._X),
            self._window_steps,
            len(self._feature_cols),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the total number of sequences."""
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple["Tensor", "Tensor"]:
        """
        Retrieve a single (X, y) pair as PyTorch tensors.

        Parameters
        ----------
        idx : int
            Sequence index.

        Returns
        -------
        tuple[Tensor, Tensor]
            X of shape (window_steps, n_features), y of shape (1,).
        """
        x_np: np.ndarray = self._X[idx]   # (window_steps, n_features)
        y_np: np.ndarray = self._y[idx]   # (1,)

        x_tensor = torch.from_numpy(x_np).float()
        y_tensor = torch.from_numpy(y_np).float()

        return x_tensor, y_tensor

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        """Number of input features per time step."""
        return len(self._feature_cols)

    @property
    def window_steps(self) -> int:
        """Number of time steps in each sequence."""
        return self._window_steps

    @property
    def feature_columns(self) -> List[str]:
        """Ordered list of feature column names."""
        return list(self._feature_cols)

    # ------------------------------------------------------------------
    def _build_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate all valid sliding window sequences from the DataFrame.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            (X_list, y_list) where each element is a NumPy array.
        """
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        # Validate that all feature columns exist
        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature columns missing from DataFrame: {missing}"
            )
        if self._target_col not in df.columns:
            raise ValueError(f"Target column '{self._target_col}' not in DataFrame.")

        if self._group_by_station and COLUMN_STATION in df.columns:
            for station, grp in df.groupby(COLUMN_STATION, sort=False):
                _log.debug("Building sequences for station: %s  (n=%d)", station, len(grp))
                xs, ys = self._slide(grp.reset_index(drop=True))
                X_list.extend(xs)
                y_list.extend(ys)
        else:
            xs, ys = self._slide(df.reset_index(drop=True))
            X_list.extend(xs)
            y_list.extend(ys)

        return X_list, y_list

    def _slide(
        self, df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply the sliding window to a single-station (or global) DataFrame.

        Returns
        -------
        tuple[list[ndarray], list[ndarray]]
        """
        feat_vals = df[self._feature_cols].values.astype("float32")
        tgt_vals = df[self._target_col].values.astype("float32")

        n = len(df)
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for start in range(0, n - self._window_steps, self._stride):
            end = start + self._window_steps
            window_X = feat_vals[start:end]              # (W, F)
            target_y = tgt_vals[end - 1]                 # scalar (predict last step)

            # Completeness check
            n_valid = np.sum(np.isfinite(window_X))
            completeness = n_valid / max(window_X.size, 1)
            if completeness < self._min_completeness:
                continue

            # Replace any remaining NaN/Inf in window with 0 (safety net)
            window_X = np.nan_to_num(window_X, nan=0.0, posinf=0.0, neginf=0.0)

            if not np.isfinite(target_y):
                continue

            X_list.append(window_X)
            y_list.append(np.array([target_y], dtype="float32"))

        return X_list, y_list


# =============================================================================
# TECReconstructionDataset - PyTorch Dataset wrapper
# =============================================================================


class TECReconstructionDataset:
    """
    High-level PyTorch-compatible ``Dataset`` for the TEC Reconstruction task.

    Wraps ``SlidingWindowDataset`` to provide a clean ``Dataset`` interface
    compatible with ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (a train, val, or test split).
    feature_columns : list[str]
        Input feature columns (in order).
    target_column : str
        Target column name.
    window_hours : int
        Window size in hours (converted to steps internally).
    temporal_resolution_minutes : int
        Temporal resolution of the data.
    stride : int
        Sliding window stride.
    min_completeness : float
        Minimum window completeness to accept.
    device : torch.device | None
        Target device.
    group_by_station : bool
        Whether to build sequences per-station.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = COLUMN_TOPSIDE_TEC,
        window_hours: int = 24,
        temporal_resolution_minutes: int = DEFAULT_TEMPORAL_RESOLUTION_MINUTES,
        stride: int = 1,
        min_completeness: float = 0.80,
        device: Optional[Any] = None,
        group_by_station: bool = True,
    ) -> None:
        _require_torch()
        window_steps = compute_window_steps(window_hours, temporal_resolution_minutes)
        _log.info(
            "TECReconstructionDataset: window_hours=%dh -> window_steps=%d  "
            "n_features=%d  n_rows=%d",
            window_hours, window_steps, len(feature_columns), len(df),
        )

        self._inner = SlidingWindowDataset(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            window_steps=window_steps,
            stride=stride,
            min_completeness=min_completeness,
            device=device,
            group_by_station=group_by_station,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Tuple["Tensor", "Tensor"]:
        return self._inner[idx]

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self._inner.n_features

    @property
    def window_steps(self) -> int:
        """Sequence length in time steps."""
        return self._inner.window_steps

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names."""
        return self._inner.feature_columns

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export all sequences as stacked NumPy arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X of shape (N, W, F), y of shape (N, 1).
        """
        if len(self._inner) == 0:
            return np.empty((0,)), np.empty((0,))
        X_np = np.stack(self._inner._X, axis=0)  # (N, W, F)
        y_np = np.stack(self._inner._y, axis=0)  # (N, 1)
        return X_np, y_np

    def to_tensors(self) -> Tuple["Tensor", "Tensor"]:
        """
        Export all sequences as PyTorch tensors.

        Returns
        -------
        tuple[Tensor, Tensor]
            X of shape (N, W, F), y of shape (N, 1).
        """
        X_np, y_np = self.to_numpy()
        return torch.from_numpy(X_np).float(), torch.from_numpy(y_np).float()


# =============================================================================
# DatasetFactory
# =============================================================================


class DatasetFactory:
    """
    Constructs train, validation, and test datasets and DataLoaders from
    the processed and feature-engineered DataFrames.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.
    feature_columns : list[str]
        Ordered input feature column names.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        feature_columns: List[str],
    ) -> None:
        _require_torch()
        self._cfg = cfg
        self._feature_cols = feature_columns
        self._target_col: str = cfg.get("data", {}).get("target_column", COLUMN_TOPSIDE_TEC)
        self._seq_cfg = cfg.get("sequences", {})
        self._train_cfg = cfg.get("training", {})
        self._hw_cfg = cfg.get("hardware", {})

        self._window_hours: int = int(
            self._seq_cfg.get("default_window_hours", 24)
        )
        self._resolution: int = int(
            self._seq_cfg.get("temporal_resolution_minutes", DEFAULT_TEMPORAL_RESOLUTION_MINUTES)
        )
        self._stride: int = int(self._seq_cfg.get("stride", 1))
        self._min_completeness: float = float(
            self._seq_cfg.get("min_completeness", 0.80)
        )

        # Device
        use_gpu: bool = self._hw_cfg.get("use_gpu", True)
        gpu_id: int = int(self._hw_cfg.get("gpu_id", 0))
        self._device = check_torch_device(use_gpu=use_gpu, gpu_id=gpu_id)

        _log.info("DatasetFactory initialised. device=%s", self._device)

    # ------------------------------------------------------------------
    def build_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[
        "TECReconstructionDataset",
        "TECReconstructionDataset",
        "TECReconstructionDataset",
    ]:
        """
        Build train, validation, and test datasets.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split.
        val_df : pd.DataFrame
            Validation split.
        test_df : pd.DataFrame
            Test split.

        Returns
        -------
        tuple[TECReconstructionDataset, TECReconstructionDataset, TECReconstructionDataset]
            (train_dataset, val_dataset, test_dataset)
        """
        kwargs = dict(
            feature_columns=self._feature_cols,
            target_column=self._target_col,
            window_hours=self._window_hours,
            temporal_resolution_minutes=self._resolution,
            stride=self._stride,
            min_completeness=self._min_completeness,
            device=self._device,
        )

        _log.info("Building training dataset ...")
        train_ds = TECReconstructionDataset(df=train_df, **kwargs)  # type: ignore[arg-type]
        _log.info("Building validation dataset ...")
        val_ds = TECReconstructionDataset(df=val_df, **kwargs)      # type: ignore[arg-type]
        _log.info("Building test dataset ...")
        test_ds = TECReconstructionDataset(df=test_df, **kwargs)    # type: ignore[arg-type]

        _log.info(
            "Datasets built: train=%d  val=%d  test=%d sequences.",
            len(train_ds), len(val_ds), len(test_ds),
        )
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    def build_dataloaders(
        self,
        train_ds: "TECReconstructionDataset",
        val_ds: "TECReconstructionDataset",
        test_ds: "TECReconstructionDataset",
    ) -> Tuple["DataLoader", "DataLoader", "DataLoader"]:
        """
        Wrap datasets in DataLoaders.

        Parameters
        ----------
        train_ds, val_ds, test_ds : TECReconstructionDataset
            Pre-built datasets.

        Returns
        -------
        tuple[DataLoader, DataLoader, DataLoader]
        """
        batch_size: int = int(self._train_cfg.get("batch_size", 64))
        num_workers: int = int(self._train_cfg.get("num_workers", 0))
        pin_memory: bool = (
            self._hw_cfg.get("pin_memory", True)
            and str(self._device) != "cpu"
        )

        _log.info(
            "Building DataLoaders: batch_size=%d  num_workers=%d  pin_memory=%s",
            batch_size, num_workers, pin_memory,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        _log.info(
            "DataLoaders ready: train_batches=%d  val_batches=%d  test_batches=%d",
            len(train_loader), len(val_loader), len(test_loader),
        )
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    def build_multi_window_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[int, Tuple[
        "TECReconstructionDataset",
        "TECReconstructionDataset",
        "TECReconstructionDataset",
    ]]:
        """
        Build datasets for every configured window size.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            Split DataFrames.

        Returns
        -------
        dict[int, tuple[TECReconstructionDataset, ...]]
            Keys are window sizes in hours.
        """
        window_sizes: List[int] = self._seq_cfg.get("window_sizes_hours", WINDOW_SIZE_HOURS)
        result: Dict[int, Any] = {}

        for wh in window_sizes:
            _log.info("--- Window: %d hours ---", wh)
            kwargs = dict(
                feature_columns=self._feature_cols,
                target_column=self._target_col,
                window_hours=wh,
                temporal_resolution_minutes=self._resolution,
                stride=self._stride,
                min_completeness=self._min_completeness,
                device=self._device,
            )
            result[wh] = (
                TECReconstructionDataset(df=train_df, **kwargs),  # type: ignore[arg-type]
                TECReconstructionDataset(df=val_df, **kwargs),    # type: ignore[arg-type]
                TECReconstructionDataset(df=test_df, **kwargs),   # type: ignore[arg-type]
            )

        _log.info("Multi-window datasets built for windows: %s hours.", window_sizes)
        return result
