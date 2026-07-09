"""
test_feature_engineering.py
============================
TEST 9 — Feature Engineering Validation

Verifies that all engineered features exist in the processed dataset
and that their values are physically reasonable.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations


import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)

# ---------------------------------------------------------------------------
# Expected engineered features and their valid value ranges
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # Temporal
    "Hour":          (0.0,  23.0),
    "Month":         (1.0,  12.0),
    "Season":        (0.0,   3.0),
    # Cyclic - sin/cos must be in [-1, 1]
    "LocalTime_sin": (-1.0,  1.0),
    "LocalTime_cos": (-1.0,  1.0),
    "DayOfYear_sin": (-1.0,  1.0),
    "DayOfYear_cos": (-1.0,  1.0),
    "Month_sin":     (-1.0,  1.0),
    "Month_cos":     (-1.0,  1.0),
    # TEC derived - no strict physical bounds, but should be finite
    "TEC_diff1":         (None, None),
    "TEC_gradient":      (None, None),
    "TEC_lag1":          (None, None),
    "TEC_lag2":          (None, None),
    "TEC_lag3":          (None, None),
    "TEC_roll_mean_3":   (None, None),
    "TEC_roll_mean_6":   (None, None),
    "TEC_roll_mean_12":  (None, None),
    "TEC_roll_std_3":    (None, None),
    "TEC_roll_std_6":    (None, None),
    "TEC_roll_std_12":   (None, None),
}


def test_feature_engineering(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
    processed_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    TEST 9: Verify all engineered features exist and have reasonable values.

    Checks:
    1. feature_names.json contains all expected engineered features.
    2. NumPy train_X has the correct number of features (39).
    3. Run FeatureEngineeringPipeline on a sample to confirm it adds features.
    4. Verify engineered features in the processed CSV (if FE was run inline).
    """
    section_header("TEST 9 — Feature Engineering Validation")

    # --- Primary check: feature_names.json --------------------------------
    feature_names_json = project_root / "data" / "processed" / "feature_names.json"
    metadata_json = project_root / "data" / "processed" / "dataset_metadata.json"
    numpy_dir = project_root / "data" / "processed" / "numpy"

    known_feature_cols: List[str] = []

    if feature_names_json.exists():
        try:
            with open(feature_names_json) as f:
                fn_data = json.load(f)
            known_feature_cols = fn_data.get("feature_columns", [])
            tracker.record(
                "feature_names.json loadable",
                passed=True,
                detail=f"{len(known_feature_cols)} features listed",
            )
        except Exception as exc:
            tracker.record("feature_names.json loadable", passed=False, detail=str(exc))
    elif metadata_json.exists():
        try:
            with open(metadata_json) as f:
                meta = json.load(f)
            known_feature_cols = meta.get("feature_columns", [])
        except Exception:
            pass

    # --- Check engineered features in feature_names.json ------------------
    sub_header("Engineered Feature Existence (from feature_names.json)")
    for feat in ENGINEERED_FEATURES:
        in_json = feat in known_feature_cols
        tracker.record(
            f"Feature in feature_names.json: {feat}",
            passed=in_json,
            detail="MISSING from JSON" if not in_json else "present",
        )

    # --- Check numpy arrays have correct feature count --------------------
    sub_header("NumPy Array Feature Count")
    numpy_x = numpy_dir / "train_X.npy"
    if numpy_x.exists():
        try:
            import numpy as np
            arr = np.load(numpy_x, allow_pickle=False)
            n_features_numpy = arr.shape[2] if arr.ndim == 3 else -1
            expected_n = len(known_feature_cols) if known_feature_cols else 39
            tracker.record(
                f"NumPy train_X feature count matches metadata",
                passed=n_features_numpy == expected_n,
                detail=f"got {n_features_numpy}, expected {expected_n}",
            )
            # Check engineered features are subset of feature names
            eng_feats_in_names = [f for f in ENGINEERED_FEATURES if f in known_feature_cols]
            tracker.record(
                f"Engineered features present in feature names ({len(eng_feats_in_names)}/{len(ENGINEERED_FEATURES)})",
                passed=len(eng_feats_in_names) >= 15,
                detail=f"{len(eng_feats_in_names)} of {len(ENGINEERED_FEATURES)} found in feature_names.json",
            )
        except Exception as exc:
            tracker.record("NumPy train_X feature count", passed=False, detail=str(exc))
    else:
        tracker.record("NumPy train_X.npy available for feature count check", passed=False,
                       warn_condition=True, detail="file not found")

    # --- Engineered Feature Value Ranges from NumPy arrays ---------------
    sub_header("Engineered Feature Values from NumPy (train_X sample)")
    numpy_x = project_root / "data" / "processed" / "numpy" / "train_X.npy"
    if numpy_x.exists() and known_feature_cols:
        try:
            arr = np.load(numpy_x, allow_pickle=False)  # (N, W, F)
            if arr.ndim == 3 and arr.shape[0] > 0:
                # Use last time step across all samples
                sample_vals = arr[:100, -1, :]  # (100, F)
                for feat, (lo, hi) in ENGINEERED_FEATURES.items():
                    if feat not in known_feature_cols:
                        continue
                    idx = known_feature_cols.index(feat)
                    if idx >= sample_vals.shape[1]:
                        continue
                    col_vals = sample_vals[:, idx]
                    col_vals_finite = col_vals[np.isfinite(col_vals)]
                    if len(col_vals_finite) == 0:
                        tracker.record(f"Values OK (NumPy): {feat}", passed=False, detail="all NaN/Inf")
                        continue
                    v_min, v_max = float(col_vals_finite.min()), float(col_vals_finite.max())
                    if lo is not None and hi is not None:
                        range_ok = (v_min >= lo - abs(lo)*0.10) and (v_max <= hi + abs(hi)*0.10)
                        tracker.record(
                            f"Values OK (NumPy): {feat}",
                            passed=range_ok,
                            detail=f"min={v_min:.3f}, max={v_max:.3f}  expected[{lo},{hi}]",
                        )
        except Exception as exc:
            tracker.record("Value range check from NumPy", passed=False, warn_condition=True, detail=str(exc))


    try:
        sys.path.insert(0, str(project_root))
        from src.feature_engineering import FeatureEngineeringPipeline

        # Use a small slice to keep it fast
        raw_csv = project_root / "data" / "raw" / "ionosonde_data.csv"
        if raw_csv.exists():
            sample = pd.read_csv(raw_csv, nrows=500, low_memory=False)
            sample["Timestamp"] = pd.to_datetime(sample["Timestamp"], errors="coerce")
            # Minimal numeric coerce
            for col in sample.select_dtypes(include="object").columns:
                if col not in ("Timestamp", "Station"):
                    sample[col] = pd.to_numeric(sample[col], errors="coerce")

            pipeline = FeatureEngineeringPipeline(cfg)
            result = pipeline.run(sample.copy())

            new_cols_added = [f for f in ENGINEERED_FEATURES if f in result.columns]
            tracker.record(
                "FeatureEngineeringPipeline.run() on sample data",
                passed=len(new_cols_added) >= 15,
                detail=f"{len(new_cols_added)}/{len(ENGINEERED_FEATURES)} features generated",
            )
        else:
            tracker.record(
                "FeatureEngineeringPipeline unit test (raw data unavailable)",
                passed=False,
                warn_condition=True,
                detail="raw CSV not found",
            )

    except Exception as exc:
        tracker.record(
            "FeatureEngineeringPipeline unit test",
            passed=False,
            warn_condition=True,
            detail=str(exc),
        )

    # --- Multi-station support check via FE pipeline output -----------------
    sub_header("Multi-Station Feature Isolation")
    try:
        sys.path.insert(0, str(project_root))
        from src.feature_engineering import FeatureEngineeringPipeline
        raw_csv = project_root / "data" / "raw" / "ionosonde_data.csv"
        if raw_csv.exists():
            raw_sample = pd.read_csv(raw_csv, nrows=2000, low_memory=False)
            raw_sample["Timestamp"] = pd.to_datetime(raw_sample["Timestamp"], errors="coerce")
            for col in raw_sample.select_dtypes(include="object").columns:
                if col not in ("Timestamp", "Station"):
                    raw_sample[col] = pd.to_numeric(raw_sample[col], errors="coerce")
            fe_out = FeatureEngineeringPipeline(cfg).run(raw_sample.copy())
            if "Station" in fe_out.columns and "TEC_lag1" in fe_out.columns:
                stations = fe_out["Station"].unique()
                cross_station_leakage = False
                for stn in stations[:5]:
                    group = fe_out[fe_out["Station"] == stn]
                    if len(group) < 5:
                        continue
                    first_lag = group["TEC_lag1"].iloc[0]
                    if not (pd.isna(first_lag) or first_lag == 0):
                        cross_station_leakage = True
                        break
                tracker.record(
                    "No cross-station TEC lag leakage detected",
                    passed=not cross_station_leakage,
                    warn_condition=cross_station_leakage,
                    detail="Lag isolated per station" if not cross_station_leakage else "Possible leakage",
                )
    except Exception as exc:
        tracker.record(
            "Multi-station feature isolation check",
            passed=False, warn_condition=True, detail=str(exc),
        )
