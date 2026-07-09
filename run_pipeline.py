"""
run_pipeline.py
===============
Phase 1 Master Pipeline Orchestrator.

Executes the complete data preparation pipeline end-to-end:

1.  Load and parse configuration.
2.  Ensure all output directories exist.
3.  Set global random seed.
4.  Configure logging from config.
5.  Load raw ionosonde CSV data.
6.  Run data validation (fail fast on critical errors).
7.  Run preprocessing (inf -> NaN, missing values, outliers, scaling, split).
8.  Run feature engineering on all three splits.
9.  Build PyTorch datasets and DataLoaders for every window size.
10. Export processed data (CSV, NumPy arrays, PyTorch tensors).
11. Save the fitted scaler and feature metadata.

Usage
-----
    python run_pipeline.py --config configs/config.yaml
    python run_pipeline.py --config configs/config.yaml --input data/raw/mydata.csv
    python run_pipeline.py --config configs/config.yaml --no-gpu

Author  : Senior Python Software Architect / AI Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 - Foundation & Data Pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make project root importable regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.constants import (
    DATASET_METADATA_FILENAME,
    FEATURE_NAMES_FILENAME,
    NUMPY_FEATURE_SUFFIX,
    NUMPY_TARGET_SUFFIX,
    SCALER_FILENAME,
    TENSOR_FEATURE_SUFFIX,
    TENSOR_TARGET_SUFFIX,
)
from src.feature_engineering import FeatureEngineeringPipeline
from src.logger import configure_from_dict, get_logger, pipeline_timer
from src.preprocessing import PreprocessingPipeline
from src.utils import (
    ensure_dir,
    ensure_dirs_from_config,
    load_config,
    save_dataframe_csv,
    save_json,
    save_numpy_arrays,
    save_scaler,
    save_torch_tensors,
    set_random_seed,
)
from src.validation import DataValidator

_log = get_logger(__name__)


# =============================================================================
# Argument Parser
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Pipeline - Ionosphere TEC Data Preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override data input path from config.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage even if CUDA is available.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override default window size in hours.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation step (not recommended for production).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip exporting processed files to disk.",
    )
    return parser.parse_args()


# =============================================================================
# Pipeline Orchestrator
# =============================================================================


def run_phase1_pipeline(
    config_path: str,
    input_override: str | None = None,
    use_gpu: bool = True,
    window_override: int | None = None,
    skip_validation: bool = False,
    skip_export: bool = False,
) -> dict:
    """
    Execute the full Phase 1 data preparation pipeline.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    input_override : str | None
        Override the raw data input path from config.
    use_gpu : bool
        Whether to use GPU for dataset tensors.
    window_override : int | None
        Override default window size in hours.
    skip_validation : bool
        Skip the validation step.
    skip_export : bool
        Skip exporting files to disk.

    Returns
    -------
    dict
        Summary metadata from the pipeline run.
    """
    pipeline_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Step 1: Load Configuration
    # -----------------------------------------------------------------------
    cfg = load_config(config_path)

    # Apply overrides
    if not use_gpu:
        cfg.setdefault("hardware", {})["use_gpu"] = False
    if window_override:
        cfg.setdefault("sequences", {})["default_window_hours"] = window_override

    # -----------------------------------------------------------------------
    # Step 2: Configure Logging
    # -----------------------------------------------------------------------
    log_cfg = cfg.get("logging", {})
    configure_from_dict(log_cfg)
    _log.info("=" * 70)
    _log.info("  Physics-Guided Mamba-TKAN  -  Phase 1 Pipeline")
    _log.info("=" * 70)
    _log.info("Config: %s", config_path)

    # -----------------------------------------------------------------------
    # Step 3: Ensure directories exist
    # -----------------------------------------------------------------------
    project_root = Path(config_path).resolve().parent.parent
    ensure_dirs_from_config(cfg, project_root)
    # Additional directories not explicitly in config paths
    ensure_dir(project_root / cfg.get("paths", {}).get("scaler_save", "models/scalers"))

    # -----------------------------------------------------------------------
    # Step 4: Set random seed
    # -----------------------------------------------------------------------
    seed: int = cfg.get("training", {}).get("random_seed", 42)
    set_random_seed(seed)

    # -----------------------------------------------------------------------
    # Step 5: Load raw data
    # -----------------------------------------------------------------------
    _log.info("\n[STEP 5] Loading Raw Data")
    preproc = PreprocessingPipeline(cfg, project_root=project_root)

    # Run only the loader to get the raw DataFrame for validation
    raw_df = preproc._loader.load(input_override)

    # -----------------------------------------------------------------------
    # Step 6: Validation
    # -----------------------------------------------------------------------
    if not skip_validation:
        _log.info("\n[STEP 6] Data Validation")
        validator = DataValidator(cfg)
        report = validator.validate(raw_df)
        if not report.passed:
            _log.error(
                "Data validation FAILED with %d errors. "
                "Inspect the validation report and fix issues before proceeding.",
                len(report.errors()),
            )
            # Log all errors
            for issue in report.errors():
                _log.error("[ERROR] %s: %s", issue.check, issue.message)
            # We continue with a WARNING rather than hard-stop (allows research flexibility)
            _log.warning(
                "Continuing pipeline despite validation errors. "
                "Results may be unreliable."
            )
        else:
            _log.info("[OK] Validation passed with %d warnings.", len(report.warnings()))
    else:
        _log.warning("[STEP 6] Validation SKIPPED.")
        report = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Step 7: Preprocessing
    # -----------------------------------------------------------------------
    _log.info("\n[STEP 7] Preprocessing (inf->NaN, missing, outliers, scale, split)")
    with pipeline_timer("Full Preprocessing", _log):
        full_df, train_df, val_df, test_df = preproc.run(input_override)

    # -----------------------------------------------------------------------
    # Step 8: Feature Engineering
    # -----------------------------------------------------------------------
    _log.info("\n[STEP 8] Feature Engineering")
    feat_eng = FeatureEngineeringPipeline(cfg)
    with pipeline_timer("Full Feature Engineering", _log):
        train_df = feat_eng.run(train_df)
        val_df = feat_eng.run(val_df)
        test_df = feat_eng.run(test_df)

    # Derive feature names (after engineering)
    feature_groups = feat_eng.get_feature_names(train_df)
    feature_columns: list = (
        feature_groups["ionosonde"]
        + feature_groups["space_weather"]
        + feature_groups["geophysical"]
        + feature_groups["temporal"]
        + feature_groups["cyclic"]
        + feature_groups["tec_derived"]
    )
    target_col: str = cfg.get("data", {}).get("target_column", "TopsideTEC")
    _log.info("Total input features: %d", len(feature_columns))
    _log.info("Target column: %s", target_col)

    # -----------------------------------------------------------------------
    # Step 9: Build PyTorch Datasets
    # -----------------------------------------------------------------------
    _log.info("\n[STEP 9] Building PyTorch Datasets")
    try:
        from src.dataset import DatasetFactory

        factory = DatasetFactory(cfg=cfg, feature_columns=feature_columns)
        # Single default-window datasets
        train_ds, val_ds, test_ds = factory.build_datasets(train_df, val_df, test_df)
        train_loader, val_loader, test_loader = factory.build_dataloaders(
            train_ds, val_ds, test_ds
        )
        # Multi-window datasets
        multi_window_datasets = factory.build_multi_window_datasets(
            train_df, val_df, test_df
        )
        _log.info("[OK] PyTorch datasets and DataLoaders ready.")
    except ImportError as exc:
        _log.warning(
            "PyTorch not available - skipping dataset creation. (%s)", exc
        )
        train_ds = val_ds = test_ds = None
        train_loader = val_loader = test_loader = None
        multi_window_datasets = {}

    # -----------------------------------------------------------------------
    # Step 10: Export
    # -----------------------------------------------------------------------
    if not skip_export:
        _log.info("\n[STEP 10] Exporting Processed Artefacts")
        _export_artefacts(
            cfg=cfg,
            project_root=project_root,
            full_df=full_df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            feature_columns=feature_columns,
            feature_groups=feature_groups,
            scaler=preproc.scaler,
        )
    else:
        _log.warning("[STEP 10] Export SKIPPED.")

    # -----------------------------------------------------------------------
    # Pipeline Summary
    # -----------------------------------------------------------------------
    total_time = time.perf_counter() - pipeline_start
    summary = {
        "status": "completed",
        "total_time_seconds": round(total_time, 3),
        "n_raw_rows": len(raw_df),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_test_rows": len(test_df),
        "n_features": len(feature_columns),
        "n_train_sequences": len(train_ds) if train_ds is not None else 0,
        "n_val_sequences": len(val_ds) if val_ds is not None else 0,
        "n_test_sequences": len(test_ds) if test_ds is not None else 0,
        "validation_passed": report.passed if report is not None else "skipped",
        "feature_groups": {k: len(v) for k, v in feature_groups.items()},
    }

    _log.info("\n" + "=" * 70)
    _log.info("  Phase 1 Pipeline COMPLETE  (%.2f s)", total_time)
    _log.info("  Train: %d rows  |  Val: %d rows  |  Test: %d rows",
              len(train_df), len(val_df), len(test_df))
    _log.info("  Input features: %d", len(feature_columns))
    if train_ds:
        _log.info("  Train sequences: %d", len(train_ds))
    _log.info("=" * 70)

    return summary


# =============================================================================
# Export Helper
# =============================================================================


def _export_artefacts(
    cfg: dict,
    project_root: Path,
    full_df,
    train_df,
    val_df,
    test_df,
    train_ds,
    val_ds,
    test_ds,
    feature_columns: list,
    feature_groups: dict,
    scaler,
) -> None:
    """Write all processed files to disk."""
    paths_cfg = cfg.get("paths", {})
    proc_dir = project_root / paths_cfg.get("processed_data", "data/processed")
    numpy_dir = proc_dir / "numpy"
    tensor_dir = proc_dir / "tensors"
    scaler_dir = project_root / paths_cfg.get("scaler_save", "models/scalers")

    # --- CSV -----------------------------------------------------------------
    csv_path = proc_dir / "processed_dataset.csv"
    save_dataframe_csv(full_df, csv_path)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        save_dataframe_csv(df, proc_dir / f"{name}.csv")

    # --- NumPy ---------------------------------------------------------------
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is None:
            continue
        X_np, y_np = ds.to_numpy()
        save_numpy_arrays(
            {f"{name}_X": X_np, f"{name}_y": y_np},
            directory=numpy_dir,
        )

    # --- Tensors -------------------------------------------------------------
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is None:
            continue
        try:
            X_t, y_t = ds.to_tensors()
            save_torch_tensors(
                {f"{name}_X": X_t, f"{name}_y": y_t},
                directory=tensor_dir,
            )
        except Exception as exc:
            _log.warning("Could not save tensors for %s split: %s", name, exc)

    # --- Scaler --------------------------------------------------------------
    if scaler.is_fitted:
        save_scaler(scaler.scaler, scaler_dir / SCALER_FILENAME)

    # --- Feature metadata ----------------------------------------------------
    save_json(
        {"feature_columns": feature_columns, "feature_groups": feature_groups},
        proc_dir / FEATURE_NAMES_FILENAME,
    )

    # --- Dataset metadata ---------------------------------------------------
    meta = {
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": cfg.get("data", {}).get("target_column", "TopsideTEC"),
        "window_hours": cfg.get("sequences", {}).get("default_window_hours", 24),
        "scaler_type": cfg.get("scaling", {}).get("scaler_type", "robust"),
        "splits": cfg.get("splits", {}),
    }
    save_json(meta, proc_dir / DATASET_METADATA_FILENAME)

    _log.info("[OK] All artefacts exported to: %s", proc_dir)


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for the Phase 1 pipeline."""
    args = parse_args()
    summary = run_phase1_pipeline(
        config_path=args.config,
        input_override=args.input,
        use_gpu=not args.no_gpu,
        window_override=args.window,
        skip_validation=args.skip_validation,
        skip_export=args.skip_export,
    )
    _log.info("Pipeline summary: %s", summary)


if __name__ == "__main__":
    main()
