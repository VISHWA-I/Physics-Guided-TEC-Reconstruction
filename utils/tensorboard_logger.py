"""
utils/tensorboard_logger.py
===========================
TensorBoard Logging Wrapper

A thin, gracefully-degrading wrapper around ``torch.utils.tensorboard.SummaryWriter``
that:

* Writes to the project's canonical ``tensorboard_logs/`` directory (routed to
  Google Drive on Colab via :mod:`utils.env_config`).
* Silently no-ops if TensorBoard is not installed — the rest of the training
  loop is never affected.
* Supports all metrics logged by the training pipeline: losses, learning rate,
  gradient norms, GPU memory.

Usage
-----
    from utils.tensorboard_logger import TensorBoardLogger

    tb = TensorBoardLogger(log_dir="tensorboard_logs/exp_001", enabled=True)

    for epoch in range(100):
        tb.log_scalars(epoch, train_loss=0.5, val_loss=0.4, lr=1e-4)
        tb.log_gpu_memory(epoch)

    tb.close()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from utils.logger import get_model_logger

logger = get_model_logger("TensorBoardLogger")


class TensorBoardLogger:
    """
    Gracefully-degrading TensorBoard SummaryWriter wrapper.

    All ``log_*`` methods are safe to call even when TensorBoard is not
    installed — they simply no-op instead of raising.

    Parameters
    ----------
    log_dir : str | Path
        Directory where TensorBoard event files will be written.
        On Colab this should point inside Google Drive.
    enabled : bool
        Master switch.  Set to *False* to disable all logging (e.g., in
        unit tests or ``development`` execution mode).
    flush_secs : int
        How often TensorBoard flushes pending events to disk (seconds).
    """

    def __init__(
        self,
        log_dir: str | Path = "tensorboard_logs",
        *,
        enabled: bool = True,
        flush_secs: int = 30,
    ) -> None:
        self.enabled = enabled
        self._writer = None

        if not enabled:
            logger.info("TensorBoard logging disabled.")
            return

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
            self._writer = SummaryWriter(log_dir=str(log_path), flush_secs=flush_secs)
            logger.info("TensorBoard writer → %s", log_path)
        except ImportError:
            logger.warning(
                "TensorBoard not installed — logging disabled. "
                "Install with: pip install tensorboard"
            )
            self.enabled = False
        except Exception as exc:
            logger.warning("TensorBoard writer creation failed: %s", exc)
            self.enabled = False

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_scalars(self, epoch: int, **kwargs: float) -> None:
        """
        Log any number of scalar values for a given epoch.

        Parameters
        ----------
        epoch : int
            Global step / epoch number.
        **kwargs : float
            Metric names and their values.  Example:
            ``tb.log_scalars(5, train_loss=0.4, val_loss=0.35, lr=1e-4)``
        """
        if not self.enabled or self._writer is None:
            return
        try:
            for name, value in kwargs.items():
                if value is not None:
                    self._writer.add_scalar(name, float(value), global_step=epoch)
        except Exception as exc:
            logger.debug("TensorBoard log_scalars failed: %s", exc)

    def log_loss_dict(self, epoch: int, loss_dict: Dict[str, float], prefix: str = "") -> None:
        """
        Log an entire dictionary of losses (e.g. from LossManager).

        Parameters
        ----------
        epoch : int
            Global step.
        loss_dict : dict
            ``{metric_name: value}`` mapping.
        prefix : str
            Optional prefix prepended to all keys (e.g. ``"train/"``).
        """
        if not self.enabled or self._writer is None:
            return
        try:
            for k, v in loss_dict.items():
                if v is not None:
                    tag = f"{prefix}{k}" if prefix else k
                    self._writer.add_scalar(tag, float(v), global_step=epoch)
        except Exception as exc:
            logger.debug("TensorBoard log_loss_dict failed: %s", exc)

    def log_learning_rate(self, epoch: int, lr: float) -> None:
        """Log current learning rate."""
        self.log_scalars(epoch, learning_rate=lr)

    def log_gpu_memory(self, epoch: int) -> None:
        """Log current GPU memory allocated (MB), if CUDA is available."""
        if not self.enabled or self._writer is None:
            return
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
                self._writer.add_scalar("GPU/memory_allocated_MB", alloc, global_step=epoch)
                self._writer.add_scalar("GPU/memory_reserved_MB", reserved, global_step=epoch)
        except Exception as exc:
            logger.debug("TensorBoard log_gpu_memory failed: %s", exc)

    def log_gradient_norm(self, epoch: int, grad_norm: float) -> None:
        """Log the global gradient L2 norm."""
        self.log_scalars(epoch, gradient_norm=grad_norm)

    def log_histogram(self, epoch: int, tag: str, values) -> None:
        """
        Log a histogram of tensor values (e.g., weight distributions).

        Parameters
        ----------
        epoch : int
            Global step.
        tag : str
            Label for the histogram.
        values : tensor or ndarray
            Values to histogram.
        """
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.add_histogram(tag, values, global_step=epoch)
        except Exception as exc:
            logger.debug("TensorBoard log_histogram failed: %s", exc)

    def flush(self) -> None:
        """Flush pending events to disk immediately."""
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
                logger.info("TensorBoard writer closed.")
            except Exception as exc:
                logger.debug("TensorBoard close error: %s", exc)
            finally:
                self._writer = None

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def is_active(self) -> bool:
        """True if the writer is enabled and successfully initialised."""
        return self.enabled and self._writer is not None


def get_tensorboard_logger(
    log_dir: str | Path | None = None,
    *,
    enabled: bool = True,
) -> TensorBoardLogger:
    """
    Convenience factory that resolves ``log_dir`` from :mod:`utils.env_config`
    when not explicitly provided.

    Parameters
    ----------
    log_dir : str | Path | None
        Explicit log directory.  If *None*, uses ``paths.tensorboard`` from
        :func:`utils.env_config.get_paths`.
    enabled : bool
        Master switch.

    Returns
    -------
    TensorBoardLogger
    """
    if log_dir is None:
        try:
            from utils.env_config import get_paths
            log_dir = get_paths().tensorboard
        except Exception:
            log_dir = Path("tensorboard_logs")

    return TensorBoardLogger(log_dir=log_dir, enabled=enabled)
