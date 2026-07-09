"""
feature_engineering.py
======================
All feature engineering transformations for the Mamba-TKAN TEC Reconstruction
pipeline.

Features generated
------------------
**Temporal features**
* Hour (0-23) extracted from Timestamp.
* Month (1-12) extracted from Timestamp.
* Season (winter / spring / summer / autumn) as integer label.

**Cyclic encodings**  (sin/cos to avoid ordinal discontinuity)
* sin(LocalTime), cos(LocalTime)  - period = 24 h
* sin(DayOfYear), cos(DayOfYear)  - period = 365.25 days
* sin(Month),     cos(Month)      - period = 12 months

**TEC derived features**
* TEC_diff1       - first-order temporal difference.
* TEC_gradient    - numerical gradient (central-difference scheme).
* TEC_lag1/2/3   - lagged values by 1, 2, 3 steps.
* TEC_roll_mean_{3,6,12}  - rolling mean over 3, 6, 12 steps.
* TEC_roll_std_{3,6,12}   - rolling standard deviation.

All operations are performed **per station** (when ``group_by_station=True``)
to prevent cross-site data leakage.

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.constants import (
    COLUMN_DAY_OF_YEAR,
    COLUMN_DOY_COS,
    COLUMN_DOY_SIN,
    COLUMN_HOUR,
    COLUMN_LOCAL_TIME,
    COLUMN_LOCAL_TIME_COS,
    COLUMN_LOCAL_TIME_SIN,
    COLUMN_MONTH,
    COLUMN_MONTH_COS,
    COLUMN_MONTH_SIN,
    COLUMN_SEASON,
    COLUMN_STATION,
    COLUMN_TEC,
    COLUMN_TEC_DIFF,
    COLUMN_TEC_GRADIENT,
    COLUMN_TEC_LAG1,
    COLUMN_TEC_LAG2,
    COLUMN_TEC_LAG3,
    COLUMN_TEC_ROLL_MEAN_12,
    COLUMN_TEC_ROLL_MEAN_3,
    COLUMN_TEC_ROLL_MEAN_6,
    COLUMN_TEC_ROLL_STD_12,
    COLUMN_TEC_ROLL_STD_3,
    COLUMN_TEC_ROLL_STD_6,
    COLUMN_TIMESTAMP,
    DAYS_PER_YEAR,
    HOURS_PER_DAY,
    MONTHS_PER_YEAR,
    MONTH_TO_SEASON,
    SEASON_ENCODING,
)
from src.logger import get_logger, pipeline_timer

_log = get_logger(__name__)

try:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # pandas >= 3.0 - warning removed



# =============================================================================
# Individual Transformer Classes
# =============================================================================


class TemporalFeatureExtractor:
    """
    Extracts calendar-based features (Hour, Month, Season) from the Timestamp
    column.

    Parameters
    ----------
    timestamp_col : str
        Name of the datetime column.
    """

    def __init__(self, timestamp_col: str = COLUMN_TIMESTAMP) -> None:
        self._ts_col = timestamp_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Hour, Month, and Season columns derived from Timestamp.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a parsed datetime Timestamp column.

        Returns
        -------
        pd.DataFrame
            DataFrame with three additional columns.
        """
        if self._ts_col not in df.columns:
            _log.warning(
                "Timestamp column '%s' not found. Skipping temporal features.", self._ts_col
            )
            return df

        ts = pd.to_datetime(df[self._ts_col], errors="coerce")

        # Hour (0-23)
        df[COLUMN_HOUR] = ts.dt.hour.astype("float32")
        _log.debug("Added column: %s", COLUMN_HOUR)

        # Month (1-12)
        df[COLUMN_MONTH] = ts.dt.month.astype("float32")
        _log.debug("Added column: %s", COLUMN_MONTH)

        # Season (integer encoded)
        df[COLUMN_SEASON] = (
            ts.dt.month
            .map(MONTH_TO_SEASON)
            .map(SEASON_ENCODING)
            .astype("float32")
        )
        _log.debug("Added column: %s", COLUMN_SEASON)

        return df


# ---------------------------------------------------------------------------


class CyclicEncoder:
    """
    Encodes periodic variables using sine-cosine transformation to avoid
    the artificial discontinuity at the wrap-around point.

    For a variable x with period P:
        sin_enc = sin(2π * x / P)
        cos_enc = cos(2π * x / P)

    Parameters
    ----------
    encodings : list[tuple[str, str, str, float]]
        Each tuple is (source_column, sin_column_name, cos_column_name, period).
    """

    def __init__(
        self,
        encodings: Optional[List[Tuple[str, str, str, float]]] = None,
    ) -> None:
        # Default encodings matching the project spec
        self._encodings: List[Tuple[str, str, str, float]] = encodings or [
            (COLUMN_LOCAL_TIME, COLUMN_LOCAL_TIME_SIN, COLUMN_LOCAL_TIME_COS, float(HOURS_PER_DAY)),
            (COLUMN_DAY_OF_YEAR, COLUMN_DOY_SIN, COLUMN_DOY_COS, DAYS_PER_YEAR),
            (COLUMN_MONTH, COLUMN_MONTH_SIN, COLUMN_MONTH_COS, float(MONTHS_PER_YEAR)),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sin/cos encodings for all configured periodic variables.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with sin/cos columns added.
        """
        for src_col, sin_col, cos_col, period in self._encodings:
            if src_col not in df.columns:
                _log.warning("Cyclic encoding: column '%s' not found. Skipping.", src_col)
                continue
            vals = pd.to_numeric(df[src_col], errors="coerce")
            angle = 2.0 * np.pi * vals / period
            df[sin_col] = np.sin(angle).astype("float32")
            df[cos_col] = np.cos(angle).astype("float32")
            _log.debug("Added cyclic columns: %s, %s  (period=%.2f)", sin_col, cos_col, period)

        return df


# ---------------------------------------------------------------------------


class TECDerivedFeatures:
    """
    Generates a rich set of derived features from the TEC column.

    Features generated
    ------------------
    * First-order difference (temporal rate of change).
    * Numerical gradient.
    * Lag features (1, 2, 3 steps).
    * Rolling mean (windows 3, 6, 12 steps).
    * Rolling standard deviation (windows 3, 6, 12 steps).

    All rolling and lag operations are applied **per station** to prevent
    cross-site data leakage.

    Parameters
    ----------
    tec_col : str
        Name of the TEC column.
    station_col : str
        Name of the station identifier column.
    group_by_station : bool
        Whether to apply transforms independently per station.
    lags : list[int]
        Lag step sizes.
    roll_windows : list[int]
        Rolling window sizes in steps.
    """

    def __init__(
        self,
        tec_col: str = COLUMN_TEC,
        station_col: str = COLUMN_STATION,
        group_by_station: bool = True,
        lags: Optional[List[int]] = None,
        roll_windows: Optional[List[int]] = None,
    ) -> None:
        self._tec = tec_col
        self._station = station_col
        self._group = group_by_station
        self._lags: List[int] = lags or [1, 2, 3]
        self._windows: List[int] = roll_windows or [3, 6, 12]

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and attach all TEC-derived features.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame enriched with derived TEC features.
        """
        if self._tec not in df.columns:
            _log.warning("TEC column '%s' not found. Skipping derived features.", self._tec)
            return df

        if self._group and self._station in df.columns:
            # Apply per-station to avoid cross-site leakage
            groups = []
            for station, grp in df.groupby(self._station, sort=False):
                grp = grp.copy()
                grp = self._compute_features(grp)
                groups.append(grp)
            df = pd.concat(groups).sort_index()
        else:
            df = self._compute_features(df)

        return df

    # ------------------------------------------------------------------
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all TEC-derived features on a (station) sub-DataFrame."""
        tec = df[self._tec]

        # --- First-order difference (rate of change) -------------------
        df[COLUMN_TEC_DIFF] = tec.diff(periods=1).astype("float32")
        _log.debug("Added: %s", COLUMN_TEC_DIFF)

        # --- Gradient (numpy central-difference) -----------------------
        finite = tec.dropna()
        grad = np.zeros(len(tec), dtype="float32")
        if len(finite) >= 2:
            grad_vals = np.gradient(finite.values.astype("float64")).astype("float32")
            grad[finite.index.map(lambda i: df.index.get_loc(i))] = grad_vals
        df[COLUMN_TEC_GRADIENT] = grad
        _log.debug("Added: %s", COLUMN_TEC_GRADIENT)

        # --- Lag features -----------------------------------------------
        lag_map = {1: COLUMN_TEC_LAG1, 2: COLUMN_TEC_LAG2, 3: COLUMN_TEC_LAG3}
        for lag in self._lags:
            col_name = lag_map.get(lag, f"TEC_lag{lag}")
            df[col_name] = tec.shift(lag).astype("float32")
            _log.debug("Added: %s  (lag=%d)", col_name, lag)

        # --- Rolling mean -----------------------------------------------
        roll_mean_map = {3: COLUMN_TEC_ROLL_MEAN_3, 6: COLUMN_TEC_ROLL_MEAN_6, 12: COLUMN_TEC_ROLL_MEAN_12}
        for win in self._windows:
            col_name = roll_mean_map.get(win, f"TEC_roll_mean_{win}")
            df[col_name] = (
                tec.rolling(window=win, min_periods=max(1, win // 2))
                .mean()
                .astype("float32")
            )
            _log.debug("Added: %s  (window=%d)", col_name, win)

        # --- Rolling std ------------------------------------------------
        roll_std_map = {3: COLUMN_TEC_ROLL_STD_3, 6: COLUMN_TEC_ROLL_STD_6, 12: COLUMN_TEC_ROLL_STD_12}
        for win in self._windows:
            col_name = roll_std_map.get(win, f"TEC_roll_std_{win}")
            df[col_name] = (
                tec.rolling(window=win, min_periods=max(2, win // 2))
                .std()
                .astype("float32")
            )
            _log.debug("Added: %s  (window=%d)", col_name, win)

        return df


# =============================================================================
# Feature Engineering Pipeline (Orchestrator)
# =============================================================================


class FeatureEngineeringPipeline:
    """
    Orchestrates all feature engineering steps in the correct order.

    Steps
    -----
    1. Extract temporal features (Hour, Month, Season).
    2. Compute cyclic encodings (LocalTime, DayOfYear, Month).
    3. Generate TEC-derived features (diff, gradient, lags, rolling stats).

    Parameters
    ----------
    cfg : dict
        Full project configuration dictionary.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        feat_cfg = cfg.get("features", {}).get("engineered_features", {})

        # Temporal
        ts_col: str = cfg.get("data", {}).get("timestamp_column", COLUMN_TIMESTAMP)
        self._temporal = TemporalFeatureExtractor(timestamp_col=ts_col)

        # Cyclic - build from config if provided
        cyclic_cfg = feat_cfg.get("cyclic", [])
        if cyclic_cfg:
            period_map = {
                "LocalTime": float(HOURS_PER_DAY),
                "DayOfYear": DAYS_PER_YEAR,
                "Month": float(MONTHS_PER_YEAR),
            }
            encodings: List[Tuple[str, str, str, float]] = []
            for item in cyclic_cfg:
                name = item["name"]
                period = float(item.get("period", period_map.get(name, 1.0)))
                sin_col = f"{name}_sin"
                cos_col = f"{name}_cos"
                encodings.append((name, sin_col, cos_col, period))
            self._cyclic = CyclicEncoder(encodings=encodings)
        else:
            self._cyclic = CyclicEncoder()  # defaults

        # TEC derived
        lag_cfg = feat_cfg.get("lag_features", {})
        roll_cfg = feat_cfg.get("rolling_features", {})
        lags: List[int] = lag_cfg.get("lags", [1, 2, 3])
        windows: List[int] = roll_cfg.get("windows", [3, 6, 12])
        group_by_station: bool = cfg.get("missing_values", {}).get("group_by_station", True)

        self._tec_features = TECDerivedFeatures(
            tec_col=COLUMN_TEC,
            station_col=COLUMN_STATION,
            group_by_station=group_by_station,
            lags=lags,
            roll_windows=windows,
        )

        _log.info("FeatureEngineeringPipeline initialised.")

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all feature engineering steps on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-processed DataFrame.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame.
        """
        _log.info("=" * 60)
        _log.info("Starting Feature Engineering Pipeline")
        _log.info("=" * 60)
        _log.info("Input shape: %s", df.shape)

        with pipeline_timer("FeatureEngineering.temporal", _log):
            df = self._temporal.transform(df)

        with pipeline_timer("FeatureEngineering.cyclic", _log):
            df = self._cyclic.transform(df)

        with pipeline_timer("FeatureEngineering.tec_derived", _log):
            df = self._tec_features.transform(df)

        new_cols = [
            COLUMN_HOUR, COLUMN_MONTH, COLUMN_SEASON,
            COLUMN_LOCAL_TIME_SIN, COLUMN_LOCAL_TIME_COS,
            COLUMN_DOY_SIN, COLUMN_DOY_COS,
            COLUMN_MONTH_SIN, COLUMN_MONTH_COS,
            COLUMN_TEC_DIFF, COLUMN_TEC_GRADIENT,
            COLUMN_TEC_LAG1, COLUMN_TEC_LAG2, COLUMN_TEC_LAG3,
            COLUMN_TEC_ROLL_MEAN_3, COLUMN_TEC_ROLL_MEAN_6, COLUMN_TEC_ROLL_MEAN_12,
            COLUMN_TEC_ROLL_STD_3, COLUMN_TEC_ROLL_STD_6, COLUMN_TEC_ROLL_STD_12,
        ]
        existing_new = [c for c in new_cols if c in df.columns]
        _log.info(
            "Feature engineering complete: output shape=%s  new_features=%d",
            df.shape, len(existing_new),
        )
        return df

    # ------------------------------------------------------------------
    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Return a categorised dictionary of all feature column names present
        in *df* after running the pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame.

        Returns
        -------
        dict[str, list[str]]
            Keys: ``"ionosonde"``, ``"space_weather"``, ``"geophysical"``,
                  ``"temporal"``, ``"cyclic"``, ``"tec_derived"``, ``"target"``.
        """
        from src.constants import (
            COLUMN_TOPSIDE_TEC,
            IONOSONDE_FEATURES,
            SPACE_WEATHER_FEATURES,
            GEOPHYSICAL_FEATURES,
        )

        all_cols = set(df.columns)
        return {
            "ionosonde": [c for c in IONOSONDE_FEATURES if c in all_cols],
            "space_weather": [c for c in SPACE_WEATHER_FEATURES if c in all_cols],
            "geophysical": [c for c in GEOPHYSICAL_FEATURES if c in all_cols],
            "temporal": [
                c for c in [COLUMN_HOUR, COLUMN_MONTH, COLUMN_SEASON] if c in all_cols
            ],
            "cyclic": [
                c for c in [
                    COLUMN_LOCAL_TIME_SIN, COLUMN_LOCAL_TIME_COS,
                    COLUMN_DOY_SIN, COLUMN_DOY_COS,
                    COLUMN_MONTH_SIN, COLUMN_MONTH_COS,
                ] if c in all_cols
            ],
            "tec_derived": [
                c for c in [
                    COLUMN_TEC_DIFF, COLUMN_TEC_GRADIENT,
                    COLUMN_TEC_LAG1, COLUMN_TEC_LAG2, COLUMN_TEC_LAG3,
                    COLUMN_TEC_ROLL_MEAN_3, COLUMN_TEC_ROLL_MEAN_6, COLUMN_TEC_ROLL_MEAN_12,
                    COLUMN_TEC_ROLL_STD_3, COLUMN_TEC_ROLL_STD_6, COLUMN_TEC_ROLL_STD_12,
                ] if c in all_cols
            ],
            "target": [COLUMN_TOPSIDE_TEC] if COLUMN_TOPSIDE_TEC in all_cols else [],
        }
