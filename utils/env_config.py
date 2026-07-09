"""
utils/env_config.py
===================
Central Environment Detection and Path Resolution

This is the **single source of truth** for all directory paths used by the project.
It automatically detects whether the code is running on Google Colab or a local
machine and routes all output paths accordingly.

Execution Modes
---------------
LOCAL  — Windows / Linux / macOS local machine.
         All paths are relative to the project working directory.

COLAB  — Google Colaboratory.
         All output paths are rooted under Google Drive so they persist
         across Colab session restarts.

Usage
-----
    from utils.env_config import get_paths, is_colab, is_cuda_available

    paths = get_paths()
    print(paths.checkpoints)    # → Path to checkpoint directory
    print(paths.logs)           # → Path to logs directory
    print(is_colab())           # → True if running on Colab
    print(is_cuda_available())  # → True if CUDA GPU is present
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Environment detection helpers
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Return True if running inside Google Colaboratory."""
    # Primary check: google.colab module is only available on Colab
    if "google.colab" in sys.modules:
        return True
    # Secondary check: COLAB_BACKEND_VERSION env var is set on Colab runtimes
    if os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    # Tertiary check: the /content directory is the Colab working directory
    if Path("/content").is_dir() and not Path("/etc/os-release").read_text(
        encoding="utf-8"
    ).startswith("NAME=\"Ubuntu\"") if Path("/etc/os-release").exists() else False:
        pass
    # Force override via env var (useful for testing)
    if os.environ.get("FORCE_COLAB", "0") == "1":
        return True
    return False


def is_cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Return the best available device string ('cuda', 'mps', or 'cpu')."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def get_amp_enabled() -> bool:
    """Return True only when CUDA is available (AMP requires CUDA on PyTorch CPU)."""
    return is_cuda_available()


# ---------------------------------------------------------------------------
# ProjectPaths dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProjectPaths:
    """
    Dataclass holding all project directory paths, resolved for the
    current execution environment (LOCAL or COLAB).

    All paths are :class:`pathlib.Path` objects so they are platform-independent
    and work identically on Windows, Linux, and macOS / Colab.
    """

    # Base directories
    project_root: Path = field(default_factory=Path.cwd)
    drive_root: Optional[Path] = None        # Only set in COLAB mode

    # Data directories (always local — data is not stored on Drive)
    data_root: Path = field(init=False)
    raw_data: Path = field(init=False)
    processed_data: Path = field(init=False)

    # Output directories (routed to Drive on Colab)
    checkpoints: Path = field(init=False)
    logs: Path = field(init=False)
    results: Path = field(init=False)
    experiments: Path = field(init=False)
    exports: Path = field(init=False)
    evaluation: Path = field(init=False)
    predictions: Path = field(init=False)
    tensorboard: Path = field(init=False)
    performance: Path = field(init=False)

    # Config directories (always local)
    configs: Path = field(init=False)
    config: Path = field(init=False)

    def __post_init__(self):
        pr = self.project_root

        # Data paths — always relative to project root
        self.data_root      = pr / "data"
        self.raw_data       = pr / "data" / "raw"
        self.processed_data = pr / "data" / "processed"

        # Config paths — always relative to project root
        self.configs = pr / "configs"
        self.config  = pr / "config"

        # Output root: Drive on Colab, project root on local
        out = self.drive_root if self.drive_root is not None else pr

        self.checkpoints  = out / "checkpoints"
        self.logs         = out / "logs"
        self.results      = out / "results"
        self.experiments  = out / "experiments"
        self.exports      = out / "exports"
        self.evaluation   = out / "evaluation"
        self.predictions  = out / "predictions"
        self.tensorboard  = out / "tensorboard_logs"
        self.performance  = out / "results" / "performance"

    def create_all(self) -> None:
        """Create all output directories (no-op if they already exist)."""
        dirs = [
            self.data_root, self.raw_data, self.processed_data,
            self.configs, self.config,
            self.checkpoints, self.logs, self.results, self.experiments,
            self.exports, self.evaluation, self.predictions,
            self.tensorboard, self.performance,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        """Return all paths as a plain string dictionary (JSON-serialisable)."""
        return {
            "project_root":   str(self.project_root),
            "drive_root":     str(self.drive_root) if self.drive_root else None,
            "data_root":      str(self.data_root),
            "raw_data":       str(self.raw_data),
            "processed_data": str(self.processed_data),
            "checkpoints":    str(self.checkpoints),
            "logs":           str(self.logs),
            "results":        str(self.results),
            "experiments":    str(self.experiments),
            "exports":        str(self.exports),
            "evaluation":     str(self.evaluation),
            "predictions":    str(self.predictions),
            "tensorboard":    str(self.tensorboard),
            "performance":    str(self.performance),
        }


# ---------------------------------------------------------------------------
# Singleton resolver
# ---------------------------------------------------------------------------

_PATHS_INSTANCE: Optional[ProjectPaths] = None


def configure(
    *,
    project_root: Optional[str] = None,
    drive_root: Optional[str] = None,
    create_dirs: bool = True,
) -> ProjectPaths:
    """
    Configure and return the singleton :class:`ProjectPaths` instance.

    Call this **once** at program startup (e.g., from ``train.py`` or the
    Colab notebook).  Subsequent calls to :func:`get_paths` return the
    already-configured instance.

    Parameters
    ----------
    project_root : str, optional
        Absolute path to the project root.  Defaults to the current working
        directory (``Path.cwd()``).
    drive_root : str, optional
        Absolute path to the Google Drive output folder.  If *None* and we
        are running on Colab, defaults to
        ``/content/drive/MyDrive/TEC_Project``.  Ignored on LOCAL.
    create_dirs : bool
        If *True* (default), create all directories immediately.

    Returns
    -------
    ProjectPaths
    """
    global _PATHS_INSTANCE

    pr = Path(project_root) if project_root else Path.cwd()

    # Resolve drive root
    resolved_drive: Optional[Path] = None
    if is_colab():
        if drive_root:
            resolved_drive = Path(drive_root)
        else:
            # Default Colab Drive location
            resolved_drive = Path("/content/drive/MyDrive/TEC_Project")
    # On local, drive_root is always ignored (paths stay within project_root)

    _PATHS_INSTANCE = ProjectPaths(project_root=pr, drive_root=resolved_drive)
    if create_dirs:
        _PATHS_INSTANCE.create_all()

    return _PATHS_INSTANCE


def get_paths() -> ProjectPaths:
    """
    Return the singleton :class:`ProjectPaths` instance, initialising it
    with defaults if :func:`configure` has not been called yet.
    """
    global _PATHS_INSTANCE
    if _PATHS_INSTANCE is None:
        configure()
    return _PATHS_INSTANCE


def print_environment_summary() -> None:
    """Print a formatted summary of the detected runtime environment."""
    paths = get_paths()
    device = get_device()
    env_name = "COLAB" if is_colab() else "LOCAL"

    print("=" * 60)
    print(f"  Runtime Environment : {env_name}")
    print(f"  Device              : {device.upper()}")
    print(f"  CUDA Available      : {is_cuda_available()}")
    print(f"  AMP Enabled         : {get_amp_enabled()}")
    print(f"  Project Root        : {paths.project_root}")
    if paths.drive_root:
        print(f"  Drive Root          : {paths.drive_root}")
    print(f"  Checkpoints         : {paths.checkpoints}")
    print(f"  Logs                : {paths.logs}")
    print(f"  Exports             : {paths.exports}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Colab-specific helpers
# ---------------------------------------------------------------------------

def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mount Google Drive inside a Colab runtime.

    Returns True if successfully mounted, False if not on Colab.
    Raises RuntimeError if mount fails.
    """
    if not is_colab():
        print("[env_config] Not on Colab — Drive mount skipped.")
        return False

    try:
        from google.colab import drive  # type: ignore[import]
        drive.mount(mount_point)
        print(f"[env_config] Google Drive mounted at {mount_point}")
        return True
    except Exception as exc:
        raise RuntimeError(f"Failed to mount Google Drive: {exc}") from exc


def install_missing_packages() -> None:
    """
    Install any packages that are not present in the Colab runtime.
    Only runs on Colab; no-op on local.
    """
    if not is_colab():
        return

    import subprocess
    packages = [
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.17.0",
        "psutil>=5.9.0",
    ]
    for pkg in packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=False,
        )
        print(f"[env_config] Checked/installed: {pkg}")
