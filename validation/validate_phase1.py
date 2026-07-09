"""
validate_phase1.py
==================
Main Phase 1 Validation Orchestrator

Runs all 18 test groups in sequence and generates:
  - Phase1_Validation_Report.txt
  - Phase1_Validation_Report.html
  - Phase1_Validation_Report.json

Usage
-----
From project root:

    python -m validation.validate_phase1

Or with explicit project root:

    python -m validation.validate_phase1 --root /path/to/project

Exit codes
----------
  0 : All tests PASS (no FAILs)
  1 : One or more tests FAIL
  2 : Critical setup error (cannot even load config)

Author  : Senior Python QA Engineer / Software Architect
Project : Physics-Guided Hybrid Mamba-TKAN Framework
          Global Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation Suite
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 on Windows to support Unicode output
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Bootstrap: resolve project root and add it to sys.path
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent  # validation/ -> project/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from validation.validation_utils import (
    ResultTracker,
    cprint,
    section_header,
)
from validation.report_generator import generate_all_reports


# ---------------------------------------------------------------------------
# Individual test module imports
# ---------------------------------------------------------------------------
from validation.test_configuration import (
    test_project_structure,
    test_configuration,
)
from validation.test_dataset import (
    test_real_data,
    test_schema,
    test_datatypes,
    test_missing_values,
    test_duplicates,
    test_outliers,
)
from validation.test_preprocessing import test_preprocessing_pipeline
from validation.test_feature_engineering import test_feature_engineering
from validation.test_scaling import test_scaling
from validation.test_sliding_window import test_sliding_window
from validation.test_dataloader import test_dataset_and_dataloader
from validation.test_logger import test_logger
from validation.test_export import test_export
from validation.test_performance import test_performance, test_final_dataset


# ---------------------------------------------------------------------------
# Validation Runner
# ---------------------------------------------------------------------------

def run_validation(project_root: Path, output_dir: Path) -> int:
    """
    Execute the complete Phase 1 validation suite.

    Parameters
    ----------
    project_root : Path
        Root directory of the project (contains src/, data/, configs/).
    output_dir : Path
        Directory where validation reports will be written.

    Returns
    -------
    int
        Exit code: 0 = all pass, 1 = failures exist, 2 = critical error.
    """
    t_start = time.perf_counter()
    tracker = ResultTracker()

    # ======================================================================
    cprint("\n" + "=" * 72, "cyan")
    cprint("  PHASE 1 VALIDATION SUITE", "bold")
    cprint("  Physics-Guided Hybrid Mamba-TKAN Framework", "cyan")
    cprint("  Global Topside Ionosphere-Plasmasphere TEC Reconstruction", "cyan")
    cprint("  " + time.strftime("%Y-%m-%d  %H:%M:%S"), "dim")
    cprint("=" * 72 + "\n", "cyan")
    cprint(f"  Project root : {project_root}", "dim")
    cprint(f"  Output dir   : {output_dir}", "dim")
    cprint(f"  Python       : {sys.version.split()[0]}", "dim")
    print()
    # ======================================================================

    cfg = {}

    try:
        # ------------------------------------------------------------------
        # TEST 1 — Project Structure
        # ------------------------------------------------------------------
        test_project_structure(project_root, tracker)

        # ------------------------------------------------------------------
        # TEST 2 — Configuration
        # ------------------------------------------------------------------
        cfg = test_configuration(project_root, tracker)
        if not cfg:
            cprint("\n  [CRITICAL] Cannot load config.yaml — aborting remaining tests.", "red")
            generate_all_reports(tracker, output_dir)
            tracker.print_summary_table()
            return 2

        # ------------------------------------------------------------------
        # TEST 3-8 — Raw Data Quality
        # ------------------------------------------------------------------
        raw_df = test_real_data(project_root, tracker)

        if raw_df is not None and len(raw_df) > 0:
            test_schema(raw_df, tracker)
            test_datatypes(raw_df, tracker)
            test_missing_values(raw_df, tracker)
            test_duplicates(raw_df, tracker)
            test_outliers(raw_df, tracker)
        else:
            cprint("\n  [WARNING] Raw DataFrame unavailable — skipping schema/dtype/missing/dup/outlier tests.", "yellow")

        # ------------------------------------------------------------------
        # TEST — Preprocessing Pipeline
        # ------------------------------------------------------------------
        pipeline_result = test_preprocessing_pipeline(project_root, cfg, tracker)
        processed_df = pipeline_result[0] if pipeline_result else None

        # ------------------------------------------------------------------
        # TEST 9 — Feature Engineering
        # ------------------------------------------------------------------
        test_feature_engineering(project_root, cfg, tracker, processed_df)

        # ------------------------------------------------------------------
        # TEST 11 — Scaling
        # ------------------------------------------------------------------
        test_scaling(project_root, cfg, tracker, processed_df)

        # ------------------------------------------------------------------
        # TEST 10 — Sliding Window
        # ------------------------------------------------------------------
        test_sliding_window(project_root, cfg, tracker, processed_df)

        # ------------------------------------------------------------------
        # TEST 12, 13, 14 — PyTorch Dataset, DataLoader, Multi-Station
        # ------------------------------------------------------------------
        test_dataset_and_dataloader(project_root, cfg, tracker, processed_df)

        # ------------------------------------------------------------------
        # TEST 15 — Logger
        # ------------------------------------------------------------------
        test_logger(project_root, cfg, tracker)

        # ------------------------------------------------------------------
        # TEST 16 — Export
        # ------------------------------------------------------------------
        test_export(project_root, cfg, tracker)

        # ------------------------------------------------------------------
        # TEST 17 — Performance
        # ------------------------------------------------------------------
        test_performance(project_root, cfg, tracker)

        # ------------------------------------------------------------------
        # TEST 18 — Final Dataset
        # ------------------------------------------------------------------
        test_final_dataset(project_root, cfg, tracker)

    except KeyboardInterrupt:
        cprint("\n\n  [INTERRUPTED] Validation stopped by user.", "yellow")
    except Exception as exc:
        cprint(f"\n  [CRITICAL ERROR] Unexpected exception: {exc}", "red")
        traceback.print_exc()

    # ======================================================================
    # Final Summary
    # ======================================================================
    elapsed_total = time.perf_counter() - t_start
    tracker.print_summary_table()
    cprint(f"  Total validation time: {elapsed_total:.1f}s", "dim")

    # Generate reports
    section_header("Generating Validation Reports")
    generate_all_reports(tracker, output_dir)

    exit_code = 0 if tracker.overall_pass else 1
    cprint(f"\n  Exit code: {exit_code}", "dim")
    return exit_code


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Validation Suite — Mamba-TKAN TEC Reconstruction"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(_PROJECT_ROOT),
        help="Path to the project root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory for validation reports (default: <root>/results/validation)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(args.root).resolve()

    if not project_root.exists():
        cprint(f"[ERROR] Project root not found: {project_root}", "red")
        sys.exit(2)

    output_dir = Path(args.output).resolve() if args.output else project_root / "results" / "validation"
    exit_code = run_validation(project_root, output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
