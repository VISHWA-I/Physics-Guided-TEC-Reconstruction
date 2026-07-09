"""
conftest.py
===========
Shared pytest fixtures for the Phase 1 Validation Suite.

Provides:
- project_root  : Path to the project root directory
- cfg           : Loaded config.yaml as a dict
- raw_df        : Raw ionosonde DataFrame (loaded once per session)
- processed_df  : Processed dataset DataFrame (loaded once per session)
- feature_cols  : List of feature column names from metadata

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import]

# Resolve project root from conftest.py location
_CONFTEST_DIR = Path(__file__).resolve().parent       # validation/
_PROJECT_ROOT = _CONFTEST_DIR.parent                  # project/

# Ensure project root is importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute project root path."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def cfg(project_root: Path) -> Dict[str, Any]:
    """Load and return the project configuration dictionary."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def raw_df(project_root: Path) -> pd.DataFrame:
    """Load the raw ionosonde CSV (loaded once per test session)."""
    csv_path = project_root / "data" / "raw" / "ionosonde_data.csv"
    if not csv_path.exists():
        pytest.skip(f"Raw data not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


@pytest.fixture(scope="session")
def processed_df(project_root: Path) -> pd.DataFrame:
    """Load the processed dataset CSV (loaded once per test session)."""
    csv_path = project_root / "data" / "processed" / "processed_dataset.csv"
    if not csv_path.exists():
        pytest.skip(f"Processed dataset not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    return df


@pytest.fixture(scope="session")
def feature_cols(project_root: Path) -> List[str]:
    """Return feature column names from dataset_metadata.json."""
    meta_path = project_root / "data" / "processed" / "dataset_metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("feature_columns", [])


@pytest.fixture(scope="session")
def numpy_dir(project_root: Path) -> Path:
    """Return path to the numpy output directory."""
    return project_root / "data" / "processed" / "numpy"


@pytest.fixture(scope="session")
def tensor_dir(project_root: Path) -> Path:
    """Return path to the tensor output directory."""
    return project_root / "data" / "processed" / "tensors"
