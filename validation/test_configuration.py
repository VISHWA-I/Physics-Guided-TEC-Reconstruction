"""
test_configuration.py
=====================
TEST 1  - Project Structure
TEST 2  - Configuration Validation

Verifies that all required directories exist and that config.yaml is
loadable and contains all mandatory keys.

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from validation.validation_utils import (
    ResultTracker,
    TestResult,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_WARN,
    section_header,
    sub_header,
    cprint,
)

# ---------------------------------------------------------------------------
# Required project directories (relative to project root)
# ---------------------------------------------------------------------------
REQUIRED_DIRS: List[str] = [
    "config",
    "data",
    "data/raw",
    "data/processed",
    "data/intermediate",
    "logs",
    "results",
    "models",
    "src",
    "utils",
]

# Optional directories (warn if missing, not fail)
OPTIONAL_DIRS: List[str] = [
    "notebooks",
    "results/plots",
    "models/scalers",
    "validation",
]

# ---------------------------------------------------------------------------
# Required config.yaml top-level keys and their mandatory sub-keys
# ---------------------------------------------------------------------------
REQUIRED_CONFIG_KEYS: Dict[str, List[str]] = {
    "project":  ["name", "version", "phase"],
    "paths":    ["data_root", "raw_data", "processed_data", "logs", "models", "results"],
    "data":     ["input_file", "target_column", "timestamp_column", "station_column"],
    "features": ["ionosonde_features", "space_weather_features"],
    "sequences": ["window_sizes_hours", "default_window_hours", "temporal_resolution_minutes"],
    "splits":   ["train", "validation", "test", "strategy"],
    "training": ["batch_size", "learning_rate", "random_seed", "max_epochs"],
    "hardware": ["use_gpu"],
    "logging":  ["level", "log_file"],
}


def test_project_structure(project_root: Path, tracker: ResultTracker) -> None:
    """TEST 1: Verify all required project directories exist."""
    section_header("TEST 1 — Project Structure")

    for dir_rel in REQUIRED_DIRS:
        dir_path = project_root / dir_rel
        exists = dir_path.is_dir()
        tracker.record(
            name=f"Directory: {dir_rel}",
            passed=exists,
            detail=str(dir_path) if not exists else "exists",
        )

    sub_header("Optional Directories")
    for dir_rel in OPTIONAL_DIRS:
        dir_path = project_root / dir_rel
        exists = dir_path.is_dir()
        tracker.record(
            name=f"[Optional] Directory: {dir_rel}",
            passed=exists,
            warn_condition=True,
            detail="missing (non-critical)" if not exists else "exists",
        )


def test_configuration(project_root: Path, tracker: ResultTracker) -> Dict[str, Any]:
    """
    TEST 2: Load config.yaml and verify all mandatory keys.

    Returns
    -------
    dict
        The loaded configuration dict (empty dict on failure).
    """
    section_header("TEST 2 — Configuration (config.yaml)")
    import yaml  # type: ignore[import]

    config_path = project_root / "config" / "config.yaml"

    # --- File existence -----------------------------------------------------
    if not config_path.exists():
        tracker.record("config.yaml exists", passed=False, detail=str(config_path))
        return {}

    tracker.record("config.yaml exists", passed=True, detail=str(config_path))

    # --- YAML parseable -----------------------------------------------------
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tracker.record("config.yaml is valid YAML", passed=True)
    except Exception as exc:
        tracker.record("config.yaml is valid YAML", passed=False, detail=str(exc))
        return {}

    if not isinstance(cfg, dict):
        tracker.record("config.yaml top-level is a dict", passed=False,
                       detail=f"Got {type(cfg).__name__}")
        return {}
    tracker.record("config.yaml top-level is a dict", passed=True)

    # --- Required top-level keys --------------------------------------------
    sub_header("Mandatory Config Keys")
    for section, sub_keys in REQUIRED_CONFIG_KEYS.items():
        section_exists = section in cfg
        tracker.record(
            f"Section [{section}] exists",
            passed=section_exists,
            detail="missing" if not section_exists else "",
        )
        if not section_exists:
            continue
        for sk in sub_keys:
            sk_exists = sk in cfg[section]
            tracker.record(
                f"  [{section}].{sk}",
                passed=sk_exists,
                detail="MISSING" if not sk_exists else str(cfg[section][sk])[:60],
            )

    # --- Specific value checks -----------------------------------------------
    sub_header("Critical Value Checks")

    # window size
    seq = cfg.get("sequences", {})
    win_sizes = seq.get("window_sizes_hours", [])
    tracker.record(
        "window_sizes_hours is a non-empty list",
        passed=isinstance(win_sizes, list) and len(win_sizes) > 0,
        detail=str(win_sizes),
    )

    # batch size
    bs = cfg.get("training", {}).get("batch_size", 0)
    tracker.record(
        "batch_size is a positive integer",
        passed=isinstance(bs, int) and bs > 0,
        detail=str(bs),
    )

    # learning rate
    lr = cfg.get("training", {}).get("learning_rate", 0)
    tracker.record(
        "learning_rate is in (0, 1)",
        passed=isinstance(lr, (int, float)) and 0 < lr < 1,
        detail=str(lr),
    )

    # random seed
    seed = cfg.get("training", {}).get("random_seed", None)
    tracker.record(
        "random_seed is set",
        passed=seed is not None,
        detail=str(seed),
    )

    # GPU flag
    gpu = cfg.get("hardware", {}).get("use_gpu", None)
    tracker.record(
        "use_gpu flag is a boolean",
        passed=isinstance(gpu, bool),
        detail=str(gpu),
    )

    # station list
    station_list = cfg.get("data", {}).get("station_list", "NOT_SET")
    tracker.record(
        "station_list is defined (may be empty)",
        passed=isinstance(station_list, (list, type(None))),
        detail=f"{len(station_list)} stations" if isinstance(station_list, list) else str(station_list),
    )

    # data paths
    data_paths = [
        ("data.input_file",              cfg.get("data", {}).get("input_file", "")),
        ("data.output_processed_csv",    cfg.get("data", {}).get("output_processed_csv", "")),
        ("data.output_numpy_dir",        cfg.get("data", {}).get("output_numpy_dir", "")),
        ("data.output_tensor_dir",       cfg.get("data", {}).get("output_tensor_dir", "")),
    ]
    for path_key, path_val in data_paths:
        tracker.record(
            f"Path configured: {path_key}",
            passed=bool(path_val),
            detail=str(path_val)[:80],
        )

    return cfg
