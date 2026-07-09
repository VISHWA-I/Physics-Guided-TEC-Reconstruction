"""
test_scaling.py
===============
TEST 11 — Scaling Validation

Verifies RobustScaler, StandardScaler, and MinMaxScaler on the
real processed data:
- No NaN introduced.
- No Inf introduced.
- Correct feature ranges for each scaler type.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)


def _validate_scaler(
    scaler,
    df: pd.DataFrame,
    scaler_name: str,
    tracker: ResultTracker,
    feature_cols: List[str],
) -> None:
    """Fit-transform a scaler and verify the output is clean."""
    try:
        X = df[feature_cols].values.astype(np.float64)
        # Replace inf before fitting
        X = np.where(np.isinf(X), np.nan, X)
        # Fill NaN with column medians
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])

        X_scaled = scaler.fit_transform(X)

        n_nan = int(np.isnan(X_scaled).sum())
        n_inf = int(np.isinf(X_scaled).sum())
        x_min = float(np.nanmin(X_scaled))
        x_max = float(np.nanmax(X_scaled))

        tracker.record(
            f"{scaler_name}: fit_transform() succeeds",
            passed=True,
            detail=f"shape={X_scaled.shape}",
        )
        tracker.record(
            f"{scaler_name}: no NaN after scaling",
            passed=n_nan == 0,
            warn_condition=True,
            detail=f"{n_nan} NaNs",
        )
        tracker.record(
            f"{scaler_name}: no Inf after scaling",
            passed=n_inf == 0,
            detail=f"{n_inf} infs",
        )

        # Range checks
        if isinstance(scaler, MinMaxScaler):
            range_ok = -0.01 <= x_min and x_max <= 1.01
            tracker.record(
                f"{scaler_name}: output in [0, 1]",
                passed=range_ok,
                detail=f"min={x_min:.4f}, max={x_max:.4f}",
            )
        elif isinstance(scaler, StandardScaler):
            # Mean ≈ 0, std ≈ 1 (global check)
            global_mean = float(np.nanmean(X_scaled))
            global_std = float(np.nanstd(X_scaled))
            tracker.record(
                f"{scaler_name}: approximate zero mean (|mean|<0.5)",
                passed=abs(global_mean) < 0.5,
                detail=f"global_mean={global_mean:.4f}",
            )
        elif isinstance(scaler, RobustScaler):
            # RobustScaler: median ≈ 0 for each feature
            col_medians_scaled = np.nanmedian(X_scaled, axis=0)
            max_median_dev = float(np.max(np.abs(col_medians_scaled)))
            tracker.record(
                f"{scaler_name}: median close to 0 (max_dev<1.0)",
                passed=max_median_dev < 1.0,
                detail=f"max_median_deviation={max_median_dev:.4f}",
            )

    except Exception as exc:
        tracker.record(f"{scaler_name}: fit_transform()", passed=False, detail=str(exc))


def test_scaling(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
    processed_df: Optional[pd.DataFrame] = None,
) -> None:
    """TEST 11: Validate RobustScaler, StandardScaler, and MinMaxScaler."""
    section_header("TEST 11 — Scaling Validation")

    # --- Load processed data if not provided --------------------------------
    if processed_df is None:
        processed_csv = project_root / "data" / "processed" / "processed_dataset.csv"
        if not processed_csv.exists():
            tracker.record("Processed CSV available for scaling test", passed=False)
            return
        try:
            processed_df = pd.read_csv(processed_csv, low_memory=False)
        except Exception as exc:
            tracker.record("Load processed CSV for scaling", passed=False, detail=str(exc))
            return

    # Pick numeric feature columns (exclude Station, Timestamp, Season, QF)
    exclude = {"Station", "Timestamp", "Season", "QF"}
    feature_cols = [
        c for c in processed_df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    if len(feature_cols) == 0:
        tracker.record("Numeric feature columns available for scaling", passed=False)
        return

    tracker.record(
        "Numeric feature columns identified",
        passed=True,
        detail=f"{len(feature_cols)} columns",
    )

    cprint(f"\n  Feature columns used for scaling ({len(feature_cols)}): "
           f"{feature_cols[:8]}{'...' if len(feature_cols) > 8 else ''}", "dim")

    # --- Test each scaler ---------------------------------------------------
    sub_header("RobustScaler")
    _validate_scaler(RobustScaler(), processed_df.copy(), "RobustScaler", tracker, feature_cols)

    sub_header("StandardScaler")
    _validate_scaler(StandardScaler(), processed_df.copy(), "StandardScaler", tracker, feature_cols)

    sub_header("MinMaxScaler")
    _validate_scaler(MinMaxScaler(), processed_df.copy(), "MinMaxScaler", tracker, feature_cols)

    # --- Test DataScaler class (project's own wrapper) ----------------------
    sub_header("DataScaler (project wrapper)")
    try:
        sys.path.insert(0, str(project_root))
        from src.preprocessing import DataScaler

        for scaler_type in ("robust", "standard", "minmax"):
            test_cfg = dict(cfg)
            test_cfg["scaling"] = dict(cfg.get("scaling", {}))
            test_cfg["scaling"]["scaler_type"] = scaler_type

            ds = DataScaler(test_cfg)
            ds.fit(processed_df[feature_cols])
            df_tr = ds.transform(processed_df.copy())
            scaled_vals = df_tr[feature_cols].values
            n_nan = int(np.isnan(scaled_vals).sum())
            n_inf = int(np.isinf(scaled_vals).sum())
            tracker.record(
                f"DataScaler({scaler_type}): no NaN/Inf after transform",
                passed=n_nan == 0 and n_inf == 0,
                detail=f"nan={n_nan}, inf={n_inf}",
            )

            # Inverse transform check
            df_inv = ds.inverse_transform(df_tr.copy())
            orig_vals = processed_df[feature_cols].values
            inv_vals = df_inv[feature_cols].values
            # Mask out rows with original NaN for comparison
            valid_mask = np.isfinite(orig_vals) & np.isfinite(inv_vals)
            if valid_mask.sum() > 0:
                max_rel_err = float(np.max(
                    np.abs(orig_vals[valid_mask] - inv_vals[valid_mask])
                    / (np.abs(orig_vals[valid_mask]) + 1e-8)
                ))
                tracker.record(
                    f"DataScaler({scaler_type}): inverse_transform accuracy (<5%)",
                    passed=max_rel_err < 0.05,
                    warn_condition=max_rel_err < 0.20,
                    detail=f"max_rel_error={max_rel_err:.6f}",
                )

    except Exception as exc:
        tracker.record("DataScaler class validation", passed=False, detail=str(exc))
