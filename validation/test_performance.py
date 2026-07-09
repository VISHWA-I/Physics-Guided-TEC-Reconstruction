"""
test_performance.py
===================
TEST 17 — Performance Benchmarks
TEST 18 — Final Dataset Validation

Measures timing and memory for each pipeline stage and validates
the final dataset is ready for Phase 2 model development.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from validation.validation_utils import (
    ResultTracker,
    section_header, sub_header, cprint,
)


# ---------------------------------------------------------------------------
# TEST 17 — Performance Benchmarks
# ---------------------------------------------------------------------------

def _memory_mb(obj: Any) -> float:
    """Estimate memory usage in MB using sys.getsizeof (rough estimate)."""
    try:
        if hasattr(obj, "memory_usage"):
            return obj.memory_usage(deep=True).sum() / 1e6
        return sys.getsizeof(obj) / 1e6
    except Exception:
        return 0.0


def test_performance(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
) -> None:
    """TEST 17: Measure data loading, preprocessing, FE, window, and memory."""
    section_header("TEST 17 — Performance Benchmarks")

    sys.path.insert(0, str(project_root))
    raw_path = project_root / "data" / "raw" / "ionosonde_data.csv"
    if not raw_path.exists():
        tracker.record("Raw data available for performance test", passed=False)
        return

    perf: Dict[str, Any] = {}

    cprint(f"\n  {'Stage':<35}  {'Time (s)':>10}  {'Memory (MB)':>14}", "bold")
    cprint(f"  {'─'*35}  {'─'*10}  {'─'*14}", "dim")

    # --- Data Loading -------------------------------------------------------
    try:
        from src.preprocessing import DataLoader
        loader = DataLoader(cfg, project_root)
        t0 = time.perf_counter()
        df = loader.load()
        elapsed = time.perf_counter() - t0
        mem = _memory_mb(df)
        perf["data_loading"] = {"time_s": round(elapsed, 3), "memory_mb": round(mem, 2)}
        cprint(f"  {'Data Loading':<35}  {elapsed:>10.3f}  {mem:>13.2f}", "green")
        tracker.record(
            "Data loading time acceptable (<60s)",
            passed=elapsed < 60,
            warn_condition=elapsed < 120,
            detail=f"{elapsed:.2f}s, {mem:.1f} MB",
        )
    except Exception as exc:
        tracker.record("Data loading benchmark", passed=False, detail=str(exc))
        return

    # --- Preprocessing ------------------------------------------------------
    try:
        from src.preprocessing import InfiniteValueHandler, MissingValueHandler, OutlierDetector
        t0 = time.perf_counter()
        df2 = InfiniteValueHandler.handle(df.copy())
        handler = MissingValueHandler(cfg)
        df2 = handler.handle(df2)
        elapsed = time.perf_counter() - t0
        mem = _memory_mb(df2)
        perf["preprocessing"] = {"time_s": round(elapsed, 3), "memory_mb": round(mem, 2)}
        cprint(f"  {'Preprocessing (inf+missing)':<35}  {elapsed:>10.3f}  {mem:>13.2f}", "green")
        tracker.record(
            "Preprocessing time acceptable (<120s)",
            passed=elapsed < 120,
            warn_condition=elapsed < 300,
            detail=f"{elapsed:.2f}s",
        )
    except Exception as exc:
        tracker.record("Preprocessing benchmark", passed=False, detail=str(exc))
        df2 = df.copy()

    # --- Feature Engineering ------------------------------------------------
    try:
        from src.feature_engineering import FeatureEngineeringPipeline
        fe_pipeline = FeatureEngineeringPipeline(cfg)
        t0 = time.perf_counter()
        df3 = fe_pipeline.run(df2.copy())
        elapsed = time.perf_counter() - t0
        mem = _memory_mb(df3)
        perf["feature_engineering"] = {"time_s": round(elapsed, 3), "memory_mb": round(mem, 2)}
        cprint(f"  {'Feature Engineering':<35}  {elapsed:>10.3f}  {mem:>13.2f}", "green")
        tracker.record(
            "Feature engineering time acceptable (<120s)",
            passed=elapsed < 120,
            warn_condition=elapsed < 300,
            detail=f"{elapsed:.2f}s",
        )
    except Exception as exc:
        tracker.record("Feature engineering benchmark", passed=False, detail=str(exc))
        df3 = df2.copy()

    # --- Sliding Window (subsample for speed) --------------------------------
    try:
        import torch
        from src.dataset import TECReconstructionDataset
        # Use subsample to measure window generation speed
        sample_df = df3.head(2000).copy() if len(df3) > 2000 else df3.copy()

        exclude = {"Station", "Timestamp", "TopsideTEC"}
        feature_cols = [c for c in sample_df.select_dtypes(include=[np.number]).columns
                        if c not in exclude and c in df3.columns]
        target_col = cfg.get("data", {}).get("target_column", "TopsideTEC")

        if feature_cols and target_col in sample_df.columns:
            resolution = int(cfg.get("sequences", {}).get("temporal_resolution_minutes", 15))
            t0 = time.perf_counter()
            ds = TECReconstructionDataset(
                df=sample_df,
                feature_columns=feature_cols,
                target_column=target_col,
                window_hours=24,
                temporal_resolution_minutes=resolution,
                stride=1,
                group_by_station="Station" in sample_df.columns,
            )
            elapsed = time.perf_counter() - t0
            mem_est = (len(ds) * (24 * 60 // resolution) * len(feature_cols) * 4) / 1e6  # float32
            perf["sliding_window"] = {"time_s": round(elapsed, 3), "memory_mb_est": round(mem_est, 2),
                                       "n_sequences": len(ds)}
            cprint(f"  {'Sliding Window (2000 rows)':<35}  {elapsed:>10.3f}  {mem_est:>13.2f} (est)", "green")
            tracker.record(
                "Sliding window generation time acceptable (<60s)",
                passed=elapsed < 60,
                warn_condition=elapsed < 180,
                detail=f"{elapsed:.2f}s for {len(ds)} sequences",
            )
    except Exception as exc:
        tracker.record("Sliding window benchmark", passed=False, detail=str(exc))

    # --- Memory Usage Summary -----------------------------------------------
    sub_header("Memory Usage Summary")
    for stage, info in perf.items():
        mem = info.get("memory_mb", info.get("memory_mb_est", 0))
        cprint(f"  {stage:<35}  {info['time_s']:>8.3f}s  {mem:>10.2f} MB", "dim")

    # Save performance results
    perf_path = project_root / "results" / "performance_benchmark.json"
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(perf_path, "w") as f:
            json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **perf}, f, indent=2)
        tracker.record("Performance results saved", passed=True, detail=str(perf_path))
    except Exception as exc:
        tracker.record("Performance results save", passed=False, warn_condition=True, detail=str(exc))


# ---------------------------------------------------------------------------
# TEST 18 — Final Dataset Validation
# ---------------------------------------------------------------------------

def test_final_dataset(
    project_root: Path,
    cfg: Dict[str, Any],
    tracker: ResultTracker,
) -> None:
    """TEST 18: Verify the final dataset dimensions and Phase 2 readiness."""
    section_header("TEST 18 — Final Dataset Readiness for Phase 2")

    numpy_dir = project_root / "data" / "processed" / "numpy"
    meta_path = project_root / "data" / "processed" / "dataset_metadata.json"

    # --- Load metadata -------------------------------------------------------
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        n_features = meta.get("n_features", 0)
        feature_cols = meta.get("feature_columns", [])
        target_col = meta.get("target_column", "TopsideTEC")
        window_hours = meta.get("window_hours", 24)
        scaler_type = meta.get("scaler_type", "unknown")

        tracker.record(
            "Dataset metadata.json readable",
            passed=True,
            detail=f"{n_features} features, window={window_hours}h, scaler={scaler_type}",
        )
        tracker.record(
            "Number of features is reasonable (≥10)",
            passed=n_features >= 10,
            detail=str(n_features),
        )
        tracker.record(
            "Target column specified",
            passed=bool(target_col),
            detail=target_col,
        )
    else:
        tracker.record("Dataset metadata.json exists", passed=False)
        meta = {}
        n_features = 0

    # --- NumPy final shapes -------------------------------------------------
    sub_header("Final Dataset Dimensions (NumPy)")
    total_samples = 0
    for split in ("train", "val", "test"):
        x_path = numpy_dir / f"{split}_X.npy"
        y_path = numpy_dir / f"{split}_y.npy"
        if x_path.exists() and y_path.exists():
            try:
                X = np.load(x_path, allow_pickle=False)
                y = np.load(y_path, allow_pickle=False)
                n = X.shape[0]
                total_samples += n
                cprint(
                    f"  {split:<8}  X={X.shape}  y={y.shape}"
                    f"  features={X.shape[-1] if X.ndim==3 else 'N/A'}",
                    "dim",
                )
                tracker.record(
                    f"{split} X shape: (N, W, F)",
                    passed=X.ndim == 3,
                    detail=str(X.shape),
                )
                tracker.record(
                    f"{split} has ≥1 sample",
                    passed=n >= 1,
                    detail=f"{n} samples",
                )
                if n_features > 0 and X.ndim == 3:
                    tracker.record(
                        f"{split} feature count matches metadata ({n_features})",
                        passed=X.shape[2] == n_features,
                        detail=f"X.shape[2]={X.shape[2]}, metadata.n_features={n_features}",
                    )
            except Exception as exc:
                tracker.record(f"{split} NumPy load", passed=False, detail=str(exc))

    tracker.record(
        "Total samples across all splits",
        passed=total_samples > 0,
        detail=f"{total_samples} total sequences",
    )

    # --- Phase 2 readiness gate -------------------------------------------
    sub_header("Phase 2 Readiness Gate")

    checks = {
        "Raw data exists":        (project_root / "data" / "raw" / "ionosonde_data.csv").exists(),
        "Processed CSV exists":   (project_root / "data" / "processed" / "processed_dataset.csv").exists(),
        "NumPy train_X exists":   (numpy_dir / "train_X.npy").exists(),
        "NumPy train_y exists":   (numpy_dir / "train_y.npy").exists(),
        "Tensors train_X.pt exists": (project_root / "data" / "processed" / "tensors" / "train_X.pt").exists(),
        "Metadata JSON exists":   meta_path.exists(),
        "Log file exists":        (project_root / "logs" / "phase1_pipeline.log").exists(),
    }

    for check_name, result in checks.items():
        tracker.record(
            f"Phase 2 Gate: {check_name}",
            passed=result,
            detail="✔" if result else "✘ MISSING",
        )

    # Final readiness summary
    all_gates_pass = all(checks.values())
    cprint(
        f"\n  {'─'*60}",
        "dim",
    )
    if all_gates_pass:
        cprint("  [OK] All Phase 2 readiness gates PASSED", "green")
        cprint("  [OK] Dataset is READY FOR PHASE 2 (Deep Learning Model Development)", "green")
    else:
        missing = [k for k, v in checks.items() if not v]
        cprint(f"  [FAIL] {len(missing)} gate(s) failed:", "red")
        for m in missing:
            cprint(f"       - {m}", "red")
