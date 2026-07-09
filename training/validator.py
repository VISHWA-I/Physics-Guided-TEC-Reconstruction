"""
training/validator.py
=====================
Validation loop for the Hybrid Mamba-TKAN model.

All batch access is performed exclusively through BatchManager.
No direct batch dictionary indexing is allowed in this module.

Performance Optimization Framework
-----------------------------------
* Execution-mode guard: ``BatchManager.validate_batch()`` is called only in
  DEVELOPMENT mode and only on the very first validation call.  Subsequent
  calls skip it entirely (``_batch_validated`` flag).
* In PRODUCTION / BENCHMARK mode the batch validation step is suppressed
  automatically — no code change needed in Trainer.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from training.metrics import MetricsCalculator
from training.batch_manager import BatchManager

# Import ExecutionMode for mode-aware validation.
# Falls back gracefully if the profiler module is unavailable.
try:
    from utils.performance_profiler import ExecutionMode
    _PROFILER_AVAILABLE = True
except ImportError:
    _PROFILER_AVAILABLE = False
    ExecutionMode = None  # type: ignore[assignment,misc]

from utils.logger import get_model_logger

logger = get_model_logger("Validator")


class Validator:
    """
    Encapsulates the validation loop without tracking gradients.

    Batch Access Policy
    -------------------
    All batch reads go through BatchManager static methods.
    This guarantees compatibility with both the nested and flat
    batch formats produced by different dataset implementations.

    Parameters
    ----------
    model : nn.Module
        The model to validate.
    device : torch.device
        Target device.
    execution_mode : str | ExecutionMode | None
        Controls debug overhead. "development" enables batch validation
        diagnostics; "production" / "benchmark" suppress them entirely.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        execution_mode: str = "production",
    ):
        self.model  = model
        self.device = device
        self.metrics_calc = MetricsCalculator()

        # Resolve execution mode
        if _PROFILER_AVAILABLE and ExecutionMode is not None:
            try:
                self._exec_mode = ExecutionMode(execution_mode)
            except ValueError:
                self._exec_mode = ExecutionMode.PRODUCTION
        else:
            self._exec_mode = execution_mode  # type: ignore[assignment]

        # One-time flag: print schema diagnostics on first batch only in development mode.
        self._batch_validated: bool = False

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Run the full validation loop.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader. Each batch must conform to the
            canonical batch schema (see BatchManager.print_expected_schema).

        Returns
        -------
        dict
            Validation metrics keyed with 'val_' prefix, plus 'val_loss'.
        """
        self.model.eval()

        total_loss = 0.0
        all_topside_preds   = []
        all_topside_targets = []

        criterion = nn.MSELoss()

        for batch in val_loader:

            # ── Diagnostics (first batch, development mode only) ──────
            if not self._batch_validated:
                _is_dev = (
                    self._exec_mode == ExecutionMode.DEVELOPMENT
                    if _PROFILER_AVAILABLE and ExecutionMode is not None
                    else str(self._exec_mode) == "development"
                )
                if _is_dev:
                    logger.info("[Validator] Inspecting first validation batch structure...")
                    BatchManager.validate_batch(batch, verbose=True)
                self._batch_validated = True

            # ── Extract inputs via BatchManager ──────────────────────
            inputs = BatchManager.get_inputs(batch, self.device)
            temporal_seq   = inputs["temporal_seq"]
            physics_feats  = inputs["physics_feats"]
            geo_feats      = inputs["geo_feats"]
            storm_feats    = inputs["storm_feats"]
            bottomside_tec = inputs["bottomside_tec"]
            geometry_feats = inputs["geometry_feats"]   # May be None

            # ── Extract targets via BatchManager ─────────────────────
            targets_topside = BatchManager.get_topside_target(batch, self.device)

            # ── Forward pass ─────────────────────────────────────────
            output = self.model(
                temporal_seq, physics_feats, geo_feats, storm_feats,
                bottomside_tec, geometry_feats
            )

            # ── Loss ─────────────────────────────────────────────────
            loss = criterion(output.topside_tec, targets_topside)
            total_loss += loss.item()

            all_topside_preds.append(output.topside_tec.detach().cpu())
            all_topside_targets.append(targets_topside.detach().cpu())

        avg_loss = total_loss / max(1, len(val_loader))

        # ── Global metrics ────────────────────────────────────────────
        preds   = torch.cat(all_topside_preds,   dim=0)
        targets = torch.cat(all_topside_targets, dim=0)

        metrics = self.metrics_calc.compute(preds, targets, prefix="val_")
        metrics["val_loss"] = avg_loss

        return metrics
