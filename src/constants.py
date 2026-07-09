"""
constants.py
============
Project-wide constants for the Physics-Guided Hybrid Mamba-TKAN Framework.

All physical constants, column definitions, season mappings, and feature
group definitions are centralised here to ensure single-source-of-truth
across every module in the project.

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

from typing import Final, List, Dict, Tuple

# =============================================================================
# Project Metadata
# =============================================================================

PROJECT_NAME: Final[str] = "Mamba-TKAN-TEC-Reconstruction"
PROJECT_VERSION: Final[str] = "1.0.0"
PROJECT_PHASE: Final[int] = 1

# =============================================================================
# Physical Constants
# =============================================================================

EARTH_RADIUS_KM: Final[float] = 6371.0          # Mean Earth radius [km]
IONO_BASE_HEIGHT_KM: Final[float] = 60.0        # Base of ionosphere [km]
PLASMASPHERE_TOP_KM: Final[float] = 20000.0     # Approximate top of plasmasphere

# Topside ionosphere base (roughly F2 peak height upper bound)
TOPSIDE_BASE_HMF2_KM: Final[float] = 200.0

# Speed of light [m/s]
SPEED_OF_LIGHT_MS: Final[float] = 2.998e8

# TEC unit conversion factor (1 TECU = 1e16 electrons/m²)
TECU_FACTOR: Final[float] = 1e16

# Seconds per day
SECONDS_PER_DAY: Final[int] = 86400

# Hours per day
HOURS_PER_DAY: Final[int] = 24

# Days per year (tropical)
DAYS_PER_YEAR: Final[float] = 365.25

# Months per year
MONTHS_PER_YEAR: Final[int] = 12

# =============================================================================
# Required CSV Columns
# =============================================================================

# Identifier / temporal columns
COLUMN_TIMESTAMP: Final[str] = "Timestamp"
COLUMN_STATION: Final[str] = "Station"
COLUMN_LATITUDE: Final[str] = "Latitude"
COLUMN_LONGITUDE: Final[str] = "Longitude"

# Ionosonde parameters
COLUMN_TEC: Final[str] = "TEC"
COLUMN_FOF2: Final[str] = "foF2"
COLUMN_HMF2: Final[str] = "hmF2"
COLUMN_SCALEF2: Final[str] = "scaleF2"
COLUMN_B0: Final[str] = "B0"
COLUMN_B1: Final[str] = "B1"
COLUMN_ZHALF_NM: Final[str] = "zhalfNm"
COLUMN_YF2: Final[str] = "yF2"
COLUMN_FF: Final[str] = "FF"
COLUMN_QF: Final[str] = "QF"

# Space weather parameters
COLUMN_F107: Final[str] = "F10.7"
COLUMN_KP: Final[str] = "Kp"
COLUMN_AP: Final[str] = "Ap"
COLUMN_DST: Final[str] = "Dst"
COLUMN_SSN: Final[str] = "SSN"

# Geophysical parameters
COLUMN_DAY_OF_YEAR: Final[str] = "DayOfYear"
COLUMN_LOCAL_TIME: Final[str] = "LocalTime"

# Target
COLUMN_TOPSIDE_TEC: Final[str] = "TopsideTEC"

# All required columns in the input CSV
REQUIRED_COLUMNS: Final[List[str]] = [
    COLUMN_TIMESTAMP,
    COLUMN_STATION,
    COLUMN_LATITUDE,
    COLUMN_LONGITUDE,
    COLUMN_TEC,
    COLUMN_FOF2,
    COLUMN_HMF2,
    COLUMN_SCALEF2,
    COLUMN_B0,
    COLUMN_B1,
    COLUMN_ZHALF_NM,
    COLUMN_YF2,
    COLUMN_FF,
    COLUMN_QF,
    COLUMN_F107,
    COLUMN_KP,
    COLUMN_AP,
    COLUMN_DST,
    COLUMN_SSN,
    COLUMN_DAY_OF_YEAR,
    COLUMN_LOCAL_TIME,
    COLUMN_TOPSIDE_TEC,
]

# =============================================================================
# Feature Groups
# =============================================================================

IONOSONDE_FEATURES: Final[List[str]] = [
    COLUMN_TEC,
    COLUMN_FOF2,
    COLUMN_HMF2,
    COLUMN_SCALEF2,
    COLUMN_B0,
    COLUMN_B1,
    COLUMN_ZHALF_NM,
    COLUMN_YF2,
    COLUMN_FF,
    COLUMN_QF,
]

SPACE_WEATHER_FEATURES: Final[List[str]] = [
    COLUMN_F107,
    COLUMN_KP,
    COLUMN_AP,
    COLUMN_DST,
    COLUMN_SSN,
]

GEOPHYSICAL_FEATURES: Final[List[str]] = [
    COLUMN_LATITUDE,
    COLUMN_LONGITUDE,
    COLUMN_DAY_OF_YEAR,
    COLUMN_LOCAL_TIME,
]

# Numeric features eligible for scaling
NUMERIC_FEATURES: Final[List[str]] = (
    IONOSONDE_FEATURES + SPACE_WEATHER_FEATURES + GEOPHYSICAL_FEATURES
)

# =============================================================================
# Engineered Feature Names
# =============================================================================

# Temporal
COLUMN_HOUR: Final[str] = "Hour"
COLUMN_MONTH: Final[str] = "Month"
COLUMN_SEASON: Final[str] = "Season"

# Cyclic encodings
COLUMN_LOCAL_TIME_SIN: Final[str] = "LocalTime_sin"
COLUMN_LOCAL_TIME_COS: Final[str] = "LocalTime_cos"
COLUMN_DOY_SIN: Final[str] = "DayOfYear_sin"
COLUMN_DOY_COS: Final[str] = "DayOfYear_cos"
COLUMN_MONTH_SIN: Final[str] = "Month_sin"
COLUMN_MONTH_COS: Final[str] = "Month_cos"

# TEC derived features
COLUMN_TEC_DIFF: Final[str] = "TEC_diff1"
COLUMN_TEC_GRADIENT: Final[str] = "TEC_gradient"
COLUMN_TEC_LAG1: Final[str] = "TEC_lag1"
COLUMN_TEC_LAG2: Final[str] = "TEC_lag2"
COLUMN_TEC_LAG3: Final[str] = "TEC_lag3"
COLUMN_TEC_ROLL_MEAN_3: Final[str] = "TEC_roll_mean_3"
COLUMN_TEC_ROLL_MEAN_6: Final[str] = "TEC_roll_mean_6"
COLUMN_TEC_ROLL_MEAN_12: Final[str] = "TEC_roll_mean_12"
COLUMN_TEC_ROLL_STD_3: Final[str] = "TEC_roll_std_3"
COLUMN_TEC_ROLL_STD_6: Final[str] = "TEC_roll_std_6"
COLUMN_TEC_ROLL_STD_12: Final[str] = "TEC_roll_std_12"

# Outlier flag column suffix
OUTLIER_FLAG_SUFFIX: Final[str] = "_outlier_flag"

# =============================================================================
# Physical Parameter Bounds (for validation)
# =============================================================================

PHYSICAL_BOUNDS: Final[Dict[str, Tuple[float, float]]] = {
    COLUMN_TEC:        (0.0, 1000.0),    # TECU
    COLUMN_FOF2:       (1.0, 25.0),      # MHz
    COLUMN_HMF2:       (80.0, 700.0),    # km
    COLUMN_SCALEF2:    (0.0, 200.0),     # km
    COLUMN_B0:         (10.0, 500.0),    # km
    COLUMN_B1:         (1.0, 10.0),      # dimensionless
    COLUMN_ZHALF_NM:   (50.0, 600.0),   # km
    COLUMN_YF2:        (10.0, 300.0),    # km
    COLUMN_FF:         (0.0, 10.0),      # dimensionless
    COLUMN_QF:         (0.0, 5.0),       # quality flag
    COLUMN_F107:       (60.0, 500.0),    # SFU
    COLUMN_KP:         (0.0, 9.0),       # dimensionless
    COLUMN_AP:         (0.0, 400.0),     # nT equivalent
    COLUMN_DST:        (-600.0, 100.0),  # nT
    COLUMN_SSN:        (0.0, 400.0),     # dimensionless
    COLUMN_LATITUDE:   (-90.0, 90.0),    # degrees
    COLUMN_LONGITUDE:  (-180.0, 180.0),  # degrees
    COLUMN_DAY_OF_YEAR:(1.0, 366.0),     # day of year
    COLUMN_LOCAL_TIME: (0.0, 24.0),      # hours
    COLUMN_TOPSIDE_TEC:(0.0, 1000.0),    # TECU
}

# =============================================================================
# Season Mapping  (Northern Hemisphere convention)
# =============================================================================

SEASON_MAP: Final[Dict[str, List[int]]] = {
    "winter":  [12, 1, 2],
    "spring":  [3, 4, 5],
    "summer":  [6, 7, 8],
    "autumn":  [9, 10, 11],
}

MONTH_TO_SEASON: Final[Dict[int, str]] = {}
for season, months in SEASON_MAP.items():
    for m in months:
        MONTH_TO_SEASON[m] = season

SEASON_ENCODING: Final[Dict[str, int]] = {
    "winter": 0,
    "spring": 1,
    "summer": 2,
    "autumn": 3,
}

# =============================================================================
# Latitude Zone Labels
# =============================================================================

LATITUDE_ZONES: Final[Dict[str, Tuple[float, float]]] = {
    "high_north":       (60.0,  90.0),
    "mid_north":        (30.0,  60.0),
    "low_north":        (15.0,  30.0),
    "equatorial_north": (0.0,   15.0),
    "equatorial_south": (-15.0,  0.0),
    "low_south":        (-30.0, -15.0),
    "mid_south":        (-60.0, -30.0),
    "high_south":       (-90.0, -60.0),
}

# =============================================================================
# Sliding Window Parameters
# =============================================================================

# Time resolution of raw data (minutes)
DEFAULT_TEMPORAL_RESOLUTION_MINUTES: Final[int] = 15

# Steps per window for each window size
WINDOW_SIZE_HOURS: Final[List[int]] = [2, 6, 12, 24, 48]

def hours_to_steps(hours: int, resolution_minutes: int = DEFAULT_TEMPORAL_RESOLUTION_MINUTES) -> int:
    """Convert a window size in hours to the equivalent number of time steps."""
    return (hours * 60) // resolution_minutes

# Pre-computed step counts (at 15-min resolution)
WINDOW_STEPS: Final[Dict[int, int]] = {
    h: hours_to_steps(h) for h in WINDOW_SIZE_HOURS
}  # {2:8, 6:24, 12:48, 24:96, 48:192}

# =============================================================================
# Scaler Names
# =============================================================================

SCALER_ROBUST: Final[str] = "robust"
SCALER_STANDARD: Final[str] = "standard"
SCALER_MINMAX: Final[str] = "minmax"

VALID_SCALERS: Final[List[str]] = [SCALER_ROBUST, SCALER_STANDARD, SCALER_MINMAX]

# =============================================================================
# Missing Value Strategies
# =============================================================================

STRATEGY_FORWARD_FILL: Final[str] = "forward_fill"
STRATEGY_BACKWARD_FILL: Final[str] = "backward_fill"
STRATEGY_INTERPOLATE: Final[str] = "interpolate"
STRATEGY_MEDIAN: Final[str] = "median"
STRATEGY_MEAN: Final[str] = "mean"
STRATEGY_DROP: Final[str] = "drop"

VALID_MISSING_STRATEGIES: Final[List[str]] = [
    STRATEGY_FORWARD_FILL,
    STRATEGY_BACKWARD_FILL,
    STRATEGY_INTERPOLATE,
    STRATEGY_MEDIAN,
    STRATEGY_MEAN,
    STRATEGY_DROP,
]

# =============================================================================
# Outlier Methods
# =============================================================================

OUTLIER_IQR: Final[str] = "iqr"
OUTLIER_ZSCORE: Final[str] = "zscore"
OUTLIER_ISOLATION_FOREST: Final[str] = "isolation_forest"

VALID_OUTLIER_METHODS: Final[List[str]] = [
    OUTLIER_IQR,
    OUTLIER_ZSCORE,
    OUTLIER_ISOLATION_FOREST,
]

OUTLIER_ACTION_CLIP: Final[str] = "clip"
OUTLIER_ACTION_NAN: Final[str] = "nan"
OUTLIER_ACTION_DROP: Final[str] = "drop"
OUTLIER_ACTION_FLAG: Final[str] = "flag"

VALID_OUTLIER_ACTIONS: Final[List[str]] = [
    OUTLIER_ACTION_CLIP,
    OUTLIER_ACTION_NAN,
    OUTLIER_ACTION_DROP,
    OUTLIER_ACTION_FLAG,
]

# =============================================================================
# Split Strategy
# =============================================================================

SPLIT_TEMPORAL: Final[str] = "temporal"
SPLIT_RANDOM: Final[str] = "random"
SPLIT_STATION: Final[str] = "station"

VALID_SPLIT_STRATEGIES: Final[List[str]] = [
    SPLIT_TEMPORAL,
    SPLIT_RANDOM,
    SPLIT_STATION,
]

# =============================================================================
# File Extensions & Naming Conventions
# =============================================================================

PROCESSED_CSV_SUFFIX: Final[str] = "_processed.csv"
NUMPY_FEATURE_SUFFIX: Final[str] = "_features.npy"
NUMPY_TARGET_SUFFIX: Final[str] = "_targets.npy"
TENSOR_FEATURE_SUFFIX: Final[str] = "_features.pt"
TENSOR_TARGET_SUFFIX: Final[str] = "_targets.pt"

SCALER_FILENAME: Final[str] = "feature_scaler.pkl"
FEATURE_NAMES_FILENAME: Final[str] = "feature_names.json"
DATASET_METADATA_FILENAME: Final[str] = "dataset_metadata.json"

# =============================================================================
# Logging Constants
# =============================================================================

LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_ROTATION: Final[str] = "midnight"
LOG_BACKUP_COUNT: Final[int] = 7
