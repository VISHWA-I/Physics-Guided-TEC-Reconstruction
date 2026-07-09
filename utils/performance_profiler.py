"""
utils/performance_profiler.py
==============================
Performance Optimization Framework for the Hybrid Mamba-TKAN Training Pipeline.

Components
----------
* ``ExecutionMode``      - Enum controlling debug/validation overhead
* ``SectionTimer``       - Context-manager for named wall-clock section timing
* ``PerformanceProfiler``- Aggregates timings, tracks samples/sec, generates reports

Execution Modes
---------------
    DEVELOPMENT  - All debug logs, tensor validation, physics assertions enabled
    PRODUCTION   - Only forward / loss / backward / optimizer; all debug disabled
    BENCHMARK    - Like PRODUCTION but with fine-grained timing probes active

Reports Generated (results/performance/)
-----------------------------------------
    performance_report.csv   - Per-epoch, per-section timing table
    performance_summary.json - Aggregated stats (mean, std, min, max, samples/sec)
    epoch_timing.png         - Epoch wall-clock trend chart
    module_timing.png        - Stacked-bar chart of section time fractions

Usage
-----
    from utils.performance_profiler import PerformanceProfiler, ExecutionMode

    profiler = PerformanceProfiler(mode=ExecutionMode.BENCHMARK)
    with profiler.section("forward_pass"):
        output = model(...)
    profiler.record_epoch(epoch=1, n_samples=5760, n_batches=180)
    profiler.generate_report()

Author  : Performance Optimization Framework
Project : Topside Ionosphere TEC Reconstruction
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — safe on CPU-only servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from utils.logger import get_model_logger

logger = get_model_logger("PerformanceProfiler")


# =============================================================================
# Execution Mode
# =============================================================================

class ExecutionMode(str, Enum):
    """Controls the level of debug and validation overhead during training."""
    DEVELOPMENT = "development"   # All validation / debug / assertions ON
    PRODUCTION  = "production"    # Only forward + loss + backward + optimizer
    BENCHMARK   = "benchmark"     # Like production + fine-grained timing probes


# =============================================================================
# Section Timer
# =============================================================================

class SectionTimer:
    """
    Lightweight wall-clock timer for a named section.
    Not thread-safe (single-threaded training loop assumption).
    """

    def __init__(self, name: str):
        self.name = name
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> "SectionTimer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def __enter__(self) -> "SectionTimer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# =============================================================================
# PerformanceProfiler
# =============================================================================

@dataclass
class EpochRecord:
    """Stores all timing and throughput data for one epoch."""
    epoch: int
    wall_time_sec: float
    n_samples: int
    n_batches: int
    samples_per_sec: float
    batches_per_sec: float
    memory_mb: float
    cpu_percent: float
    section_times: Dict[str, float] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Aggregates section timings across batches and epochs.

    Parameters
    ----------
    mode : ExecutionMode
        Controls which features are active.
    output_dir : str
        Directory for all generated reports and charts.
    enabled : bool
        Master switch. When False the profiler is a no-op.
    """

    # Canonical section names (subset active per mode)
    ALL_SECTIONS: Tuple[str, ...] = (
        "dataset",
        "dataloader",
        "forward_pass",
        "physics_encoder",
        "cross_attention",
        "memory_bank",
        "tkan",
        "loss",
        "backward_pass",
        "optimizer",
        "validation",
        "checkpoint",
        "logging",
    )

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        output_dir: str = "results/performance",
        enabled: bool = True,
    ) -> None:
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.enabled = enabled and (mode == ExecutionMode.BENCHMARK)

        # Per-batch accumulators (reset each epoch)
        self._batch_accum: Dict[str, float] = defaultdict(float)

        # Epoch-level records
        self._epoch_records: List[EpochRecord] = []

        # Epoch wall-clock start
        self._epoch_start: Optional[float] = None

        # Process handle (for memory/CPU)
        self._process = psutil.Process() if _PSUTIL_AVAILABLE else None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "PerformanceProfiler ACTIVE — mode=%s  output=%s",
                mode.value, self.output_dir,
            )
        else:
            logger.info(
                "PerformanceProfiler mode=%s — timing probes disabled "
                "(enable BENCHMARK mode to collect per-section data).",
                mode.value,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """
        Context manager that times a named section.
        Is a **no-op** unless mode == BENCHMARK.

        Usage::

            with profiler.section("forward_pass"):
                output = model(...)
        """
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._batch_accum[name] += elapsed

    def start_epoch(self) -> None:
        """Call at the beginning of every epoch."""
        self._epoch_start = time.perf_counter()
        if self.enabled:
            self._batch_accum.clear()

    def record_epoch(
        self,
        epoch: int,
        n_samples: int,
        n_batches: int,
    ) -> EpochRecord:
        """
        Finalise timing for the completed epoch and persist a record.

        Parameters
        ----------
        epoch : int
            1-based epoch number.
        n_samples : int
            Total training samples seen this epoch.
        n_batches : int
            Total batches processed this epoch.

        Returns
        -------
        EpochRecord
            The completed record (also appended to internal history).
        """
        wall_time = time.perf_counter() - (self._epoch_start or time.perf_counter())
        sps = n_samples / max(wall_time, 1e-9)
        bps = n_batches / max(wall_time, 1e-9)

        mem_mb = 0.0
        cpu_pct = 0.0
        if self._process is not None:
            try:
                mem_mb = self._process.memory_info().rss / (1024 ** 2)
                cpu_pct = self._process.cpu_percent(interval=None)
            except Exception:
                pass

        record = EpochRecord(
            epoch=epoch,
            wall_time_sec=wall_time,
            n_samples=n_samples,
            n_batches=n_batches,
            samples_per_sec=sps,
            batches_per_sec=bps,
            memory_mb=mem_mb,
            cpu_percent=cpu_pct,
            section_times=dict(self._batch_accum) if self.enabled else {},
        )
        self._epoch_records.append(record)
        return record

    def get_speed_summary(self, epoch_record: EpochRecord) -> str:
        """Return a compact one-line speed summary for logging."""
        return (
            f"[Speed] {epoch_record.samples_per_sec:.1f} sps | "
            f"{epoch_record.batches_per_sec:.2f} bps | "
            f"Epoch: {epoch_record.wall_time_sec:.1f}s | "
            f"Mem: {epoch_record.memory_mb:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> None:
        """
        Generate all reports and charts.
        Writes to ``self.output_dir``.
        """
        if not self._epoch_records:
            logger.warning("No epoch records to report.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._write_csv()
        self._write_json()

        if _MPL_AVAILABLE:
            self._plot_epoch_timing()
            if self.enabled:
                self._plot_module_timing()
        else:
            logger.warning(
                "matplotlib not available — skipping chart generation. "
                "Install with: pip install matplotlib"
            )

        logger.info("Performance report saved to %s", self.output_dir)

    def _write_csv(self) -> None:
        """Write per-epoch timing table to performance_report.csv."""
        csv_path = self.output_dir / "performance_report.csv"

        # Collect all section keys that appeared across epochs
        all_sections = set()
        for r in self._epoch_records:
            all_sections.update(r.section_times.keys())
        all_sections_sorted = sorted(all_sections)

        fieldnames = [
            "epoch", "wall_time_sec", "n_samples", "n_batches",
            "samples_per_sec", "batches_per_sec", "memory_mb", "cpu_percent",
        ] + [f"section_{s}" for s in all_sections_sorted]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._epoch_records:
                row: Dict[str, object] = {
                    "epoch": r.epoch,
                    "wall_time_sec": f"{r.wall_time_sec:.4f}",
                    "n_samples": r.n_samples,
                    "n_batches": r.n_batches,
                    "samples_per_sec": f"{r.samples_per_sec:.2f}",
                    "batches_per_sec": f"{r.batches_per_sec:.4f}",
                    "memory_mb": f"{r.memory_mb:.1f}",
                    "cpu_percent": f"{r.cpu_percent:.1f}",
                }
                for s in all_sections_sorted:
                    row[f"section_{s}"] = f"{r.section_times.get(s, 0.0):.4f}"
                writer.writerow(row)

        logger.info("CSV report → %s", csv_path)

    def _write_json(self) -> None:
        """Write aggregated summary statistics to performance_summary.json."""
        json_path = self.output_dir / "performance_summary.json"

        if not self._epoch_records:
            return

        wall_times = [r.wall_time_sec for r in self._epoch_records]
        sps_vals   = [r.samples_per_sec for r in self._epoch_records]
        mem_vals   = [r.memory_mb for r in self._epoch_records]
        cpu_vals   = [r.cpu_percent for r in self._epoch_records]

        def _stats(vals: List[float]) -> Dict[str, float]:
            n = len(vals)
            mean = sum(vals) / n
            variance = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
            return {
                "mean": round(mean, 4),
                "std":  round(variance ** 0.5, 4),
                "min":  round(min(vals), 4),
                "max":  round(max(vals), 4),
            }

        summary = {
            "total_epochs": len(self._epoch_records),
            "total_samples_trained": sum(r.n_samples for r in self._epoch_records),
            "execution_mode": self.mode.value,
            "wall_time_sec": _stats(wall_times),
            "samples_per_sec": _stats(sps_vals),
            "memory_mb": _stats(mem_vals),
            "cpu_percent": _stats(cpu_vals),
        }

        # Section-level aggregation (only if timing probes were active)
        if self.enabled and self._epoch_records[0].section_times:
            all_sections: set = set()
            for r in self._epoch_records:
                all_sections.update(r.section_times.keys())
            section_summary: Dict[str, Dict] = {}
            for s in sorted(all_sections):
                vals = [r.section_times.get(s, 0.0) for r in self._epoch_records]
                section_summary[s] = _stats(vals)
            summary["section_times_sec"] = section_summary

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("JSON summary → %s", json_path)

    def _plot_epoch_timing(self) -> None:
        """Plot epoch wall-clock time trend."""
        png_path = self.output_dir / "epoch_timing.png"
        epochs   = [r.epoch for r in self._epoch_records]
        times    = [r.wall_time_sec for r in self._epoch_records]
        sps      = [r.samples_per_sec for r in self._epoch_records]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        color_time = "#4A90D9"
        color_sps  = "#E86A33"

        ax1.plot(epochs, times, color=color_time, linewidth=2, marker="o",
                 markersize=4, label="Epoch Time (s)")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Wall Time (s)", color=color_time, fontsize=12)
        ax1.tick_params(axis="y", labelcolor=color_time)
        ax1.set_facecolor("#F8F9FA")

        ax2 = ax1.twinx()
        ax2.plot(epochs, sps, color=color_sps, linewidth=2, linestyle="--",
                 marker="s", markersize=4, label="Samples/sec")
        ax2.set_ylabel("Samples / sec", color=color_sps, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=color_sps)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                   framealpha=0.9)

        fig.suptitle("Training Speed — Epoch Timing", fontsize=14, fontweight="bold")
        fig.patch.set_facecolor("#FFFFFF")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Epoch timing chart → %s", png_path)

    def _plot_module_timing(self) -> None:
        """Plot stacked-bar chart of section time fractions (last epoch)."""
        if not self._epoch_records:
            return

        last = self._epoch_records[-1]
        if not last.section_times:
            return

        png_path = self.output_dir / "module_timing.png"

        sections = list(last.section_times.keys())
        times    = [last.section_times[s] for s in sections]
        total    = sum(times) or 1.0
        fracs    = [t / total * 100 for t in times]

        # Sort by fraction descending
        pairs = sorted(zip(fracs, sections), reverse=True)
        fracs, sections = zip(*pairs) if pairs else ([], [])

        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(sections, fracs, color=[colors[i % 10] for i in range(len(sections))],
                       edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, frac, t in zip(bars, fracs, times):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{frac:.1f}%  ({t:.2f}s)",
                va="center", ha="left", fontsize=9,
            )

        ax.set_xlabel("% of Total Section Time (last epoch)", fontsize=12)
        ax.set_title(
            f"Module Timing Breakdown — Epoch {last.epoch}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlim(0, max(fracs, default=100) * 1.3)
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("#FFFFFF")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Module timing chart → %s", png_path)

    # ------------------------------------------------------------------
    # Execution mode helpers (used by Trainer / Validator)
    # ------------------------------------------------------------------

    @property
    def is_development(self) -> bool:
        return self.mode == ExecutionMode.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.mode == ExecutionMode.PRODUCTION

    @property
    def is_benchmark(self) -> bool:
        return self.mode == ExecutionMode.BENCHMARK

    def should_validate_tensors(self) -> bool:
        """Tensor shape validation only in development mode."""
        return self.mode == ExecutionMode.DEVELOPMENT

    def should_run_physics_assertions(self) -> bool:
        """Physics assertions only in development mode."""
        return self.mode == ExecutionMode.DEVELOPMENT

    def should_run_batch_validation(self) -> bool:
        """Batch interface validation only in development mode."""
        return self.mode == ExecutionMode.DEVELOPMENT

    def should_print_debug(self) -> bool:
        """Debug prints only in development mode."""
        return self.mode == ExecutionMode.DEVELOPMENT
