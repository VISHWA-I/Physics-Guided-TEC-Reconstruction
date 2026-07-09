"""
preprocessing.py
================
Full preprocessing pipeline for ionosonde data including:

1.  **Data Loading** - robust CSV ingest with dtype coercion.
2.  **Station Filtering** - configurable station whitelist.
3.  **Missing Value Handling** - forward-fill, backward-fill, interpolation,
    median, mean, and drop strategies (per-station or global).
4.  **Infinite Value Replacement** - replace ±inf with NaN then handle.
5.  **Outlier Detection & Treatment** - IQR, Z-score, and Isolation Forest.
6.  **Scaling** - RobustScaler, StandardScaler, or MinMaxScaler.
7.  **Train/Val/Test Splitting** - temporal, random, or station-based.

Each step is independently callable so it can be used from notebooks or
integrated into the full ``run_pipeline`` orchestrator.

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.constants import (
    COLUMN_LATITUDE,
    COLUMN_LONGITUDE,
    COLUMN_STATION,
    COLUMN_TIMESTAMP,
    COLUMN_TOPSIDE_TEC,
    NUMERIC_FEATURES,
    OUTLIER_ACTION_CLIP,
    OUTLIER_ACTION_DROP,
    OUTLIER_ACTION_FLAG,
    OUTLIER_ACTION_NAN,
    OUTLIER_FLAG_SUFFIX,
    OUTLIER_IQR,
    OUTLIER_ISOLATION_FOREST,
    OUTLIER_ZSCORE,
    SCALER_MINMAX,
    SCALER_ROBUST,
    SCALER_STANDARD,
    STRATEGY_BACKWARD_FILL,
    STRATEGY_DROP,
    STRATEGY_FORWARD_FILL,
    STRATEGY_INTERPOLATE,
    STRATEGY_MEAN,
    STRATEGY_MEDIAN,
)
from src.logger import get_logger, log_dataframe_summary, pipeline_timer
from src.utils import filter_stations, split_dataframe_temporal

_log = get_logger(__name__)

# Suppress pandas chained-assignment warnings in this module
# SettingWithCopyWarning was removed in pandas 3.0 (pd.options.mode.copy_on_write=True is default)
try:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # pandas >= 3.0 - warning no longer exists


# =============================================================================
# Data Loader
# =============================================================================


class DataLoader:
    """
    Robust CSV loader for ionosonde station data.

    Handles
    -------
    * Multi-file directories (loads and concatenates all CSV files).
    * Single CSV file loading.
    * Automatic dtype coercion (numeric columns -> float64).
    * Timestamp parsing.
    * Station whitelisting.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.
    project_root : str | Path
        Root path of the project (used to resolve relative config paths).
    """

    def __init__(self, cfg: Dict[str, Any], project_root: str | Path = ".") -> None:
        self._cfg = cfg
        self._data_cfg = cfg.get("data", {})
        self._project_root = Path(project_root).resolve()
        self._timestamp_col: str = self._data_cfg.get("timestamp_column", COLUMN_TIMESTAMP)
        self._timestamp_fmt: Optional[str] = self._data_cfg.get("timestamp_format")
        self._station_col: str = self._data_cfg.get("station_column", COLUMN_STATION)
        self._station_list: List[str] = self._data_cfg.get("station_list", []) or []

    # ------------------------------------------------------------------
    def load(self, input_path: Optional[str | Path] = None) -> pd.DataFrame:
        """
        Load ionosonde data from a CSV file or directory.

        Parameters
        ----------
        input_path : str | Path | None
            Override the path from config.  If None, uses
            ``config.data.input_file``.

        Returns
        -------
        pd.DataFrame
            Loaded, type-coerced DataFrame.
        """
        src = Path(input_path or self._data_cfg.get("input_file", ""))
        if not src.is_absolute():
            src = self._project_root / src

        if not src.exists():
            raise FileNotFoundError(f"Input data not found: {src}")

        with pipeline_timer("DataLoader.load", _log):
            if src.is_dir():
                df = self._load_directory(src)
            else:
                df = self._load_single_file(src)

        _log.info("Loaded %d rows × %d cols from: %s", len(df), len(df.columns), src)
        log_dataframe_summary(df, tag="Raw Load", logger=_log)

        # Station filtering
        df = filter_stations(df, self._station_list, self._station_col)

        # Parse timestamp
        df = self._parse_timestamp(df)

        return df

    # ------------------------------------------------------------------
    def _load_single_file(self, path: Path) -> pd.DataFrame:
        """Read a single CSV into a DataFrame with safe type coercion."""
        _log.info("Reading CSV: %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = self._coerce_numeric_columns(df)
        return df

    def _load_directory(self, directory: Path) -> pd.DataFrame:
        """Load all CSV files in a directory and concatenate them."""
        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {directory}")
        _log.info("Found %d CSV files in %s", len(csv_files), directory)
        frames: List[pd.DataFrame] = []
        for fp in csv_files:
            _log.debug("Loading: %s", fp.name)
            frame = pd.read_csv(fp, low_memory=False)
            frame = self._coerce_numeric_columns(frame)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        _log.info("Concatenated %d files -> %d rows", len(csv_files), len(df))
        return df

    @staticmethod
    def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to cast object columns to numeric where possible."""
        for col in df.columns:
            if df[col].dtype == object and col not in (COLUMN_TIMESTAMP, COLUMN_STATION):
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only keep conversion if it doesn't destroy too much info
                non_null_before = df[col].notna().sum()
                non_null_after = converted.notna().sum()
                if non_null_after >= 0.80 * non_null_before:
                    df[col] = converted
        return df

    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the timestamp column to datetime64."""
        if self._timestamp_col not in df.columns:
            _log.warning("Timestamp column '%s' not found; skipping parse.", self._timestamp_col)
            return df
        try:
            if self._timestamp_fmt:
                df[self._timestamp_col] = pd.to_datetime(
                    df[self._timestamp_col], format=self._timestamp_fmt, errors="coerce"
                )
            else:
                df[self._timestamp_col] = pd.to_datetime(
                    df[self._timestamp_col], infer_datetime_format=True, errors="coerce"
                )
            n_null = int(df[self._timestamp_col].isna().sum())
            if n_null > 0:
                _log.warning("Timestamp parsing: %d nulls after conversion.", n_null)
        except Exception as exc:
            _log.error("Timestamp parsing failed: %s", exc)
        return df


# =============================================================================
# Missing Value Handler
# =============================================================================


class MissingValueHandler:
    """
    Implements configurable strategies for handling missing values.

    Strategies
    ----------
    * ``forward_fill``  - propagate last valid observation forward.
    * ``backward_fill`` - propagate next valid observation backward.
    * ``interpolate``   - linear / cubic / spline interpolation.
    * ``median``        - fill with column median.
    * ``mean``          - fill with column mean.
    * ``drop``          - remove rows with any NaN.

    When ``group_by_station=True`` all strategies are applied per-station
    group to avoid leakage across observation sites.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        mv_cfg = cfg.get("missing_values", {})
        self._strategy: str = mv_cfg.get("strategy", STRATEGY_INTERPOLATE)
        self._interp_method: str = mv_cfg.get("interpolation_method", "linear")
        self._interp_limit: int = int(mv_cfg.get("interpolation_limit", 6))
        self._fallback: str = mv_cfg.get("fallback_strategy", STRATEGY_MEDIAN)
        self._group_by_station: bool = mv_cfg.get("group_by_station", True)
        _log.info(
            "MissingValueHandler configured: strategy=%s  interp=%s  limit=%d  "
            "group_by_station=%s",
            self._strategy, self._interp_method, self._interp_limit, self._group_by_station,
        )

    # ------------------------------------------------------------------
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the configured missing value strategy to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame potentially containing NaN values.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.
        """
        n_nan_before = int(df.isna().sum().sum())
        _log.info("Missing value handling - NaN count before: %d", n_nan_before)

        with pipeline_timer("MissingValueHandler.handle", _log):
            if self._group_by_station and COLUMN_STATION in df.columns:
                df = self._apply_grouped(df)
            else:
                df = self._apply_strategy(df, self._strategy)

        # Fallback for any remaining NaNs
        remaining = int(df.isna().sum().sum())
        if remaining > 0 and self._strategy != self._fallback:
            _log.info("Applying fallback strategy '%s' for remaining %d NaNs.",
                      self._fallback, remaining)
            df = self._apply_strategy(df, self._fallback)

        n_nan_after = int(df.isna().sum().sum())
        _log.info(
            "Missing value handling complete: %d -> %d NaN values.",
            n_nan_before, n_nan_after,
        )
        return df

    # ------------------------------------------------------------------
    def _apply_grouped(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the strategy independently per station group."""
        groups = []
        for station, group in df.groupby(COLUMN_STATION, sort=False):
            _log.debug("Handling missing values for station: %s", station)
            groups.append(self._apply_strategy(group.copy(), self._strategy))
        return pd.concat(groups).sort_index()

    def _apply_strategy(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply a single named strategy to the DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if strategy == STRATEGY_FORWARD_FILL:
            df[numeric_cols] = df[numeric_cols].ffill(limit=self._interp_limit)

        elif strategy == STRATEGY_BACKWARD_FILL:
            df[numeric_cols] = df[numeric_cols].bfill(limit=self._interp_limit)

        elif strategy == STRATEGY_INTERPOLATE:
            df[numeric_cols] = df[numeric_cols].interpolate(
                method=self._interp_method,
                limit=self._interp_limit,
                limit_direction="both",
            )

        elif strategy == STRATEGY_MEDIAN:
            medians = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(medians)

        elif strategy == STRATEGY_MEAN:
            means = df[numeric_cols].mean()
            df[numeric_cols] = df[numeric_cols].fillna(means)

        elif strategy == STRATEGY_DROP:
            before = len(df)
            df = df.dropna(subset=numeric_cols)
            _log.info("Dropped %d rows with NaN values.", before - len(df))

        else:
            _log.warning("Unknown missing value strategy '%s'. No action taken.", strategy)

        return df


# =============================================================================
# Infinite Value Handler
# =============================================================================


class InfiniteValueHandler:
    """
    Replaces ±inf values in numeric columns with NaN so they can be treated
    by the MissingValueHandler.

    Parameters
    ----------
    None (stateless utility class).
    """

    @staticmethod
    def handle(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace all ±inf entries with NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_inf = int(np.isinf(df[numeric_cols].values).sum())
        if n_inf > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            _log.warning("Replaced %d infinite values with NaN.", n_inf)
        else:
            _log.info("[OK] No infinite values found.")
        return df


# =============================================================================
# Outlier Detector
# =============================================================================


class OutlierDetector:
    """
    Detects and treats outliers using IQR, Z-score, and Isolation Forest.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.

    Notes
    -----
    Multiple methods are applied sequentially.  An outlier flagged by
    **any** method receives the configured treatment (clip / nan / drop / flag).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        out_cfg = cfg.get("outliers", {})
        self._methods: List[str] = out_cfg.get("methods", [OUTLIER_IQR])
        self._action: str = out_cfg.get("action", OUTLIER_ACTION_CLIP)
        self._iqr_cfg: Dict[str, Any] = out_cfg.get("iqr", {})
        self._zscore_cfg: Dict[str, Any] = out_cfg.get("zscore", {})
        self._if_cfg: Dict[str, Any] = out_cfg.get("isolation_forest", {})

        _log.info(
            "OutlierDetector configured: methods=%s  action=%s",
            self._methods, self._action,
        )

    # ------------------------------------------------------------------
    def detect_and_treat(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all configured outlier detection methods and apply treatment.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            * Treated DataFrame.
            * Boolean outlier mask DataFrame (True = outlier).
        """
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        with pipeline_timer("OutlierDetector.detect_and_treat", _log):
            for method in self._methods:
                if method == OUTLIER_IQR:
                    mask = self._detect_iqr(df)
                elif method == OUTLIER_ZSCORE:
                    mask = self._detect_zscore(df)
                elif method == OUTLIER_ISOLATION_FOREST:
                    mask = self._detect_isolation_forest(df)
                else:
                    _log.warning("Unknown outlier method '%s'. Skipping.", method)
                    continue
                # Union of outlier masks
                for col in mask.columns:
                    if col in outlier_mask.columns:
                        outlier_mask[col] = outlier_mask[col] | mask[col]

            n_outliers = int(outlier_mask.any(axis=1).sum())
            _log.info("Total rows flagged as outliers: %d (%.2f%%)",
                      n_outliers, 100.0 * n_outliers / max(len(df), 1))

            df = self._apply_action(df, outlier_mask)

        return df, outlier_mask

    # ------------------------------------------------------------------
    def _get_target_columns(self, sub_cfg: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Resolve the list of columns to apply a specific method to."""
        configured = sub_cfg.get("columns", [])
        if configured:
            return [c for c in configured if c in df.columns]
        # Default: all numeric columns
        return df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    def _detect_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IQR-based outlier detection.

        A value x is an outlier if:
            x < Q1 - k * IQR  or  x > Q3 + k * IQR
        where k = ``multiplier`` (default 2.5).
        """
        multiplier: float = float(self._iqr_cfg.get("multiplier", 2.5))
        target_cols = self._get_target_columns(self._iqr_cfg, df)
        mask = pd.DataFrame(False, index=df.index, columns=target_cols)

        for col in target_cols:
            series = df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - multiplier * iqr
            hi = q3 + multiplier * iqr
            col_mask = (df[col] < lo) | (df[col] > hi)
            mask[col] = col_mask.fillna(False)
            n = int(col_mask.sum())
            if n > 0:
                _log.debug("IQR - '%s': %d outliers (lo=%.3f, hi=%.3f)", col, n, lo, hi)

        return mask

    # ------------------------------------------------------------------
    def _detect_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score based outlier detection.

        A value x is an outlier if:  |z| = |(x - μ) / σ| > threshold
        """
        threshold: float = float(self._zscore_cfg.get("threshold", 3.5))
        target_cols = self._get_target_columns(self._zscore_cfg, df)
        mask = pd.DataFrame(False, index=df.index, columns=target_cols)

        for col in target_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            mu, sigma = series.mean(), series.std()
            if sigma < 1e-10:
                continue
            z = (series - mu).abs() / sigma
            col_mask = z > threshold
            mask[col] = col_mask.fillna(False)
            n = int(col_mask.sum())
            if n > 0:
                _log.debug("Z-score - '%s': %d outliers (threshold=%.1f)", col, n, threshold)

        return mask

    # ------------------------------------------------------------------
    def _detect_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Isolation Forest multivariate outlier detection.

        Fits a single model on all target columns jointly.
        """
        contamination: float = float(self._if_cfg.get("contamination", 0.05))
        n_estimators: int = int(self._if_cfg.get("n_estimators", 100))
        random_state: int = int(self._if_cfg.get("random_state", 42))
        target_cols = self._get_target_columns(self._if_cfg, df)

        # Use only rows where all target columns are non-NaN
        sub = df[target_cols].dropna()
        if len(sub) < 10:
            _log.warning("Isolation Forest: insufficient non-NaN rows (%d). Skipping.", len(sub))
            return pd.DataFrame(False, index=df.index, columns=target_cols)

        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        preds = clf.fit_predict(sub.values)   # 1 = inlier, -1 = outlier
        outlier_indices = sub.index[preds == -1]

        mask = pd.DataFrame(False, index=df.index, columns=target_cols)
        mask.loc[outlier_indices, target_cols] = True
        _log.debug(
            "Isolation Forest: %d / %d samples flagged as outliers.",
            len(outlier_indices), len(sub),
        )
        return mask

    # ------------------------------------------------------------------
    def _apply_action(
        self, df: pd.DataFrame, outlier_mask: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the configured action to detected outliers."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        shared_cols = [c for c in outlier_mask.columns if c in numeric_cols]

        if self._action == OUTLIER_ACTION_CLIP:
            # Clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for each column
            for col in shared_cols:
                col_outliers = outlier_mask[col] if col in outlier_mask else None
                if col_outliers is None or not col_outliers.any():
                    continue
                lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lo, upper=hi)

        elif self._action == OUTLIER_ACTION_NAN:
            for col in shared_cols:
                if col in outlier_mask.columns:
                    df.loc[outlier_mask[col], col] = np.nan

        elif self._action == OUTLIER_ACTION_DROP:
            row_mask = outlier_mask[shared_cols].any(axis=1)
            before = len(df)
            df = df[~row_mask].copy()
            _log.info("Dropped %d outlier rows.", before - len(df))

        elif self._action == OUTLIER_ACTION_FLAG:
            for col in shared_cols:
                if col in outlier_mask.columns:
                    df[f"{col}{OUTLIER_FLAG_SUFFIX}"] = outlier_mask[col].astype(int)

        return df


# =============================================================================
# Scaler
# =============================================================================


class DataScaler:
    """
    Wraps scikit-learn scalers with project-specific logic.

    Supports
    --------
    * ``robust``   - RobustScaler (recommended for data with outliers).
    * ``standard`` - StandardScaler (zero mean, unit variance).
    * ``minmax``   - MinMaxScaler (scales to [0, 1]).

    The scaler is fitted **only** on training data to prevent data leakage.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.
    """

    _SCALER_MAP = {
        SCALER_ROBUST: RobustScaler,
        SCALER_STANDARD: StandardScaler,
        SCALER_MINMAX: MinMaxScaler,
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        scale_cfg = cfg.get("scaling", {})
        scaler_type: str = scale_cfg.get("scaler_type", SCALER_ROBUST)
        if scaler_type not in self._SCALER_MAP:
            _log.warning(
                "Unknown scaler '%s'; defaulting to '%s'.", scaler_type, SCALER_ROBUST
            )
            scaler_type = SCALER_ROBUST

        self._scaler_type: str = scaler_type
        self._exclude: List[str] = scale_cfg.get("exclude_columns", [])
        self._scaler = self._SCALER_MAP[scaler_type]()
        self._fit_columns: Optional[List[str]] = None
        self._is_fitted: bool = False

        _log.info("DataScaler initialised with scaler_type='%s'.", scaler_type)

    # ------------------------------------------------------------------
    @property
    def scaler(self):
        """Return the underlying sklearn scaler object."""
        return self._scaler

    @property
    def is_fitted(self) -> bool:
        """True if the scaler has been fitted."""
        return self._is_fitted

    @property
    def fit_columns(self) -> Optional[List[str]]:
        """Columns the scaler was fitted on."""
        return self._fit_columns

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "DataScaler":
        """
        Fit the scaler on a DataFrame (should be training data only).

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        DataScaler
            Self (for chaining).
        """
        cols = self._get_scale_columns(df)
        _log.info("Fitting %s on %d columns.", self._scaler_type, len(cols))
        self._scaler.fit(df[cols].values)
        self._fit_columns = cols
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scaler to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to scale.

        Returns
        -------
        pd.DataFrame
            Scaled DataFrame.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        df = df.copy()
        cols = [c for c in self._fit_columns if c in df.columns]  # type: ignore[union-attr]
        df[cols] = self._scaler.transform(df[cols].values)
        _log.info("Scaled %d columns using %s.", len(cols), self._scaler_type)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in a single call (for training set)."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the scaling transformation.

        Parameters
        ----------
        df : pd.DataFrame
            Scaled DataFrame.

        Returns
        -------
        pd.DataFrame
            Unscaled DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform().")
        df = df.copy()
        cols = [c for c in self._fit_columns if c in df.columns]  # type: ignore[union-attr]
        df[cols] = self._scaler.inverse_transform(df[cols].values)
        return df

    # ------------------------------------------------------------------
    def _get_scale_columns(self, df: pd.DataFrame) -> List[str]:
        """Return numeric columns eligible for scaling, respecting exclusion list."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in self._exclude]


# =============================================================================
# Full Preprocessing Pipeline
# =============================================================================


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing sequence:

    1. Load raw data.
    2. Replace infinite values with NaN.
    3. Handle missing values.
    4. Detect and treat outliers.
    5. Temporal / random / station split.
    6. Fit scaler on train, transform all splits.

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.
    project_root : str | Path
        Project root directory.
    """

    def __init__(self, cfg: Dict[str, Any], project_root: str | Path = ".") -> None:
        self._cfg = cfg
        self._project_root = Path(project_root).resolve()
        self._loader = DataLoader(cfg, project_root)
        self._inf_handler = InfiniteValueHandler()
        self._missing_handler = MissingValueHandler(cfg)
        self._outlier_detector = OutlierDetector(cfg)
        self._scaler = DataScaler(cfg)

    # ------------------------------------------------------------------
    @property
    def scaler(self) -> DataScaler:
        """Access the fitted DataScaler instance."""
        return self._scaler

    # ------------------------------------------------------------------
    def run(
        self,
        input_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.

        Parameters
        ----------
        input_path : str | Path | None
            Optional path override for the raw data file.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (processed_full, train_df, val_df, test_df)
        """
        _log.info("=" * 60)
        _log.info("Starting Preprocessing Pipeline")
        _log.info("=" * 60)

        # Step 1: Load
        df = self._loader.load(input_path)

        # Step 2: Infinite values -> NaN
        df = self._inf_handler.handle(df)

        # Step 3: Missing values
        df = self._missing_handler.handle(df)

        # Step 4: Outlier detection and treatment
        df, _outlier_mask = self._outlier_detector.detect_and_treat(df)

        # Step 5: Sort by timestamp before splitting
        timestamp_col = self._cfg.get("data", {}).get("timestamp_column", COLUMN_TIMESTAMP)
        if timestamp_col in df.columns:
            df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Step 6: Split
        splits_cfg = self._cfg.get("splits", {})
        strategy: str = splits_cfg.get("strategy", "temporal")
        train_frac: float = float(splits_cfg.get("train", 0.70))
        val_frac: float = float(splits_cfg.get("validation", 0.15))

        if strategy == "temporal":
            train_df, val_df, test_df = split_dataframe_temporal(
                df, train_frac=train_frac, val_frac=val_frac, timestamp_col=timestamp_col
            )
        elif strategy == "random":
            random_seed = self._cfg.get("training", {}).get("random_seed", 42)
            df_shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
            train_df, val_df, test_df = split_dataframe_temporal(
                df_shuffled, train_frac=train_frac, val_frac=val_frac, timestamp_col=timestamp_col
            )
        else:
            _log.warning("Split strategy '%s' not implemented; defaulting to temporal.", strategy)
            train_df, val_df, test_df = split_dataframe_temporal(
                df, train_frac=train_frac, val_frac=val_frac, timestamp_col=timestamp_col
            )

        # Step 7: Fit scaler on train, transform all
        train_df = self._scaler.fit_transform(train_df)
        val_df = self._scaler.transform(val_df)
        test_df = self._scaler.transform(test_df)

        _log.info(
            "Preprocessing complete: train=%d  val=%d  test=%d",
            len(train_df), len(val_df), len(test_df),
        )
        return df, train_df, val_df, test_df
