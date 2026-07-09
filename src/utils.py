"""
utils.py
========
Shared utility functions used across all pipeline modules.

Utilities cover
---------------
* YAML configuration loading and schema validation.
* Path management (directory creation, path resolution).
* Scaler persistence (save / load).
* Data export helpers (CSV, NumPy, PyTorch tensors).
* Seed / reproducibility helpers.
* Generic decorators (timing, retry).
* Miscellaneous data helpers.

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import functools
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

if TYPE_CHECKING:
    # torch is imported lazily at runtime; this import is only for the type checker.
    import torch

import numpy as np
import pandas as pd
import yaml

from src.logger import get_logger

_log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Configuration Utilities
# =============================================================================


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and return the project YAML configuration as a nested dictionary.

    Parameters
    ----------
    config_path : str | Path
        Absolute or relative path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    yaml.YAMLError
        If the file contains invalid YAML syntax.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    _log.info("Configuration loaded from: %s", path)
    return cfg


def get_nested(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from a deeply nested dictionary.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Root configuration dictionary.
    *keys : str
        Key path (e.g. ``"data", "input_file"`` resolves ``cfg["data"]["input_file"]``).
    default : Any
        Value to return when any key is absent.

    Returns
    -------
    Any
    """
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_path(base: str | Path, relative: str) -> Path:
    """
    Resolve *relative* against *base*, returning an absolute Path.

    Parameters
    ----------
    base : str | Path
        Base directory (e.g., project root).
    relative : str
        Relative path string from config.

    Returns
    -------
    Path
    """
    return (Path(base).resolve() / relative).resolve()


# =============================================================================
# Directory Utilities
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory (and parents) if it does not already exist.

    Parameters
    ----------
    path : str | Path
        Directory path to create.

    Returns
    -------
    Path
        Resolved absolute path of the created / existing directory.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs_from_config(cfg: Dict[str, Any], project_root: str | Path) -> None:
    """
    Read the ``paths`` section of config and ensure all directories exist.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Full project config dictionary.
    project_root : str | Path
        Root directory of the project.
    """
    paths_cfg = cfg.get("paths", {})
    for key, rel_path in paths_cfg.items():
        if isinstance(rel_path, str):
            full_path = resolve_path(project_root, rel_path)
            ensure_dir(full_path)
            _log.debug("Ensured directory [%s]: %s", key, full_path)


# =============================================================================
# Reproducibility
# =============================================================================


def set_random_seed(seed: int = 42) -> None:
    """
    Fix all relevant random seeds for full reproducibility.

    Covers Python ``random``, NumPy, and PyTorch (if installed).

    Parameters
    ----------
    seed : int
        Seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        _log.debug("PyTorch seeds fixed at %d", seed)
    except ImportError:
        _log.debug("PyTorch not installed - skipping torch seed.")

    _log.info("Global random seed set to %d", seed)


# =============================================================================
# Scaler Persistence
# =============================================================================


def save_scaler(scaler: Any, path: str | Path) -> None:
    """
    Persist a fitted scikit-learn scaler to disk using pickle.

    Parameters
    ----------
    scaler : Any
        A fitted scaler object (e.g. ``RobustScaler``).
    path : str | Path
        Destination file path (``*.pkl`` recommended).
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(scaler, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _log.info("Scaler saved -> %s", p)


def load_scaler(path: str | Path) -> Any:
    """
    Load a previously persisted scaler from disk.

    Parameters
    ----------
    path : str | Path
        Path to the pickle file.

    Returns
    -------
    Any
        Deserialised scaler object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Scaler file not found: {p}")
    with p.open("rb") as fh:
        scaler = pickle.load(fh)
    _log.info("Scaler loaded ← %s", p)
    return scaler


# =============================================================================
# Data Export Helpers
# =============================================================================


def save_dataframe_csv(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    float_format: str = "%.6f",
) -> None:
    """
    Save a DataFrame to CSV with sensible defaults.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    path : str | Path
        Output file path.
    index : bool
        Whether to write the index column.
    float_format : str
        Format string for floating-point columns.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, float_format=float_format)
    _log.info("CSV saved -> %s  (rows=%d, cols=%d)", p, len(df), len(df.columns))


def save_numpy_arrays(
    arrays: Dict[str, np.ndarray],
    directory: str | Path,
    prefix: str = "",
) -> None:
    """
    Save a dictionary of named NumPy arrays to a directory.

    Parameters
    ----------
    arrays : Dict[str, np.ndarray]
        Mapping of filename stem to array.
    directory : str | Path
        Target directory.
    prefix : str
        Optional filename prefix.
    """
    out_dir = ensure_dir(directory)
    for name, arr in arrays.items():
        fname = f"{prefix}{name}.npy" if prefix else f"{name}.npy"
        p = out_dir / fname
        np.save(str(p), arr)
        _log.info("NumPy array saved -> %s  shape=%s  dtype=%s", p, arr.shape, arr.dtype)


def save_torch_tensors(
    tensors: Dict[str, Any],
    directory: str | Path,
    prefix: str = "",
) -> None:
    """
    Save a dictionary of named PyTorch tensors to a directory.

    Parameters
    ----------
    tensors : Dict[str, Any]
        Mapping of filename stem to tensor.
    directory : str | Path
        Target directory.
    prefix : str
        Optional filename prefix.

    Notes
    -----
    Imports PyTorch lazily so that the module can be used on machines
    without a GPU / torch installation for the preprocessing steps.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        _log.error("PyTorch is not installed - cannot save tensors.")
        raise

    out_dir = ensure_dir(directory)
    for name, tensor in tensors.items():
        fname = f"{prefix}{name}.pt" if prefix else f"{name}.pt"
        p = out_dir / fname
        torch.save(tensor, str(p))
        _log.info("Tensor saved -> %s  shape=%s  dtype=%s", p, tensor.shape, tensor.dtype)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Serialise an object to a JSON file.

    Parameters
    ----------
    data : Any
        JSON-serialisable object.
    path : str | Path
        Destination path.
    indent : int
        JSON indent level.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
    _log.info("JSON saved -> %s", p)


def load_json(path: str | Path) -> Any:
    """
    Load a JSON file and return the parsed Python object.

    Parameters
    ----------
    path : str | Path
        Source path.

    Returns
    -------
    Any
        Parsed JSON content.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# Decorators
# =============================================================================


def timeit(func: F) -> F:
    """
    Decorator that logs the wall-clock execution time of a function at INFO
    level using the module-level logger.

    Parameters
    ----------
    func : Callable
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            _log.info("[TIMER]  %s completed in %.3f s", func.__qualname__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            _log.error("[ERR] %s failed after %.3f s", func.__qualname__, elapsed)
            raise
    return wrapper  # type: ignore[return-value]


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator factory that retries a function on failure.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts before re-raising.
    delay : float
        Seconds to sleep between attempts.
    exceptions : tuple[type[Exception], ...]
        Exception types that trigger a retry.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    _log.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator  # type: ignore[return-value]


# =============================================================================
# Miscellaneous Data Utilities
# =============================================================================


def get_latitude_zone(lat: float) -> str:
    """
    Return the latitude zone label for a given latitude.

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees.

    Returns
    -------
    str
        Zone label (e.g. ``"mid_north"``, ``"equatorial_south"``).
    """
    from src.constants import LATITUDE_ZONES  # local import to avoid circular
    for zone, (lo, hi) in LATITUDE_ZONES.items():
        if lo <= lat <= hi:
            return zone
    return "unknown"


def month_to_season(month: int) -> str:
    """
    Map a calendar month (1-12) to its meteorological season name.

    Parameters
    ----------
    month : int
        Month number (1 = January, 12 = December).

    Returns
    -------
    str
        One of ``"winter"``, ``"spring"``, ``"summer"``, ``"autumn"``.
    """
    from src.constants import MONTH_TO_SEASON
    return MONTH_TO_SEASON.get(month, "unknown")


def compute_window_steps(
    window_hours: int, resolution_minutes: int = 15
) -> int:
    """
    Convert a time window in hours to the equivalent number of discrete
    time steps.

    Parameters
    ----------
    window_hours : int
        Window size in hours.
    resolution_minutes : int
        Temporal resolution of the dataset in minutes.

    Returns
    -------
    int
        Number of time steps in the window.
    """
    return (window_hours * 60) // resolution_minutes


def split_dataframe_temporal(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    timestamp_col: str = "Timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame chronologically into train / validation / test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame sorted by *timestamp_col*.
    train_frac : float
        Fraction of rows assigned to the training set.
    val_frac : float
        Fraction of rows assigned to the validation set.
    timestamp_col : str
        Column used for sorting before splitting.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train: n_train + n_val]
    test_df = df_sorted.iloc[n_train + n_val:]

    _log.info(
        "Temporal split -> train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def filter_stations(
    df: pd.DataFrame,
    station_list: Optional[List[str]],
    station_col: str = "Station",
) -> pd.DataFrame:
    """
    Filter a DataFrame to include only rows from the specified stations.

    If *station_list* is empty or None, all stations are retained.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    station_list : list[str] | None
        List of station identifiers to keep.
    station_col : str
        Name of the column containing station identifiers.

    Returns
    -------
    pd.DataFrame
    """
    if not station_list:
        _log.info("No station filter applied - using all %d stations.",
                  df[station_col].nunique())
        return df
    before = len(df)
    df_filtered = df[df[station_col].isin(station_list)].copy()
    _log.info(
        "Station filter: kept %d / %d rows for stations %s",
        len(df_filtered), before, station_list,
    )
    return df_filtered


def describe_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str] | None
        Subset of columns.  If None, all numeric columns are used.

    Returns
    -------
    pd.DataFrame
        Statistics (count, mean, std, min, quartiles, max).
    """
    cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    return df[cols].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]).T


def check_torch_device(use_gpu: bool = True, gpu_id: int = 0) -> "torch.device":  # noqa: F821
    """
    Determine the best available PyTorch device.

    Parameters
    ----------
    use_gpu : bool
        If True, prefer CUDA GPU if available.
    gpu_id : int
        GPU device index.

    Returns
    -------
    torch.device
    """
    try:
        import torch  # type: ignore
        if use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            _log.info(
                "GPU device selected: %s  (%s)",
                device,
                torch.cuda.get_device_name(gpu_id),
            )
        else:
            device = torch.device("cpu")
            if use_gpu:
                _log.warning("GPU requested but CUDA is not available - using CPU.")
            else:
                _log.info("CPU device selected.")
        return device
    except ImportError:
        _log.error("PyTorch not installed - cannot determine device.")
        raise
