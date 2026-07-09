"""
training/trainer.py
===================
Master Orchestrator for Phase 3 Training.

Performance Optimization Framework Additions
--------------------------------------------
* ``PerformanceProfiler`` integration — per-section timing probes (BENCHMARK mode)
* ``tqdm`` progress bar — Epoch / Batch / Loss / ETA / Samples-per-second
* ``torch.compile`` support — optional PyTorch 2.x graph optimization
* Execution-mode guards — expensive debug/validation disabled in production
* One-time batch validation — BatchManager.validate_batch() only on epoch 1
* Per-epoch speed report — logged and saved via profiler
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from training.training_config import TrainingConfig
from training.physics_loss import PhysicsLoss
from training.loss_manager import LossManager
from training.curriculum_learning import CurriculumLearning
from training.optimizer_factory import OptimizerFactory
from training.scheduler_factory import SchedulerFactory
from training.callbacks import CallbackList
from training.early_stopping import EarlyStopping
from training.gradient_monitor import GradientMonitor
from training.checkpoint_manager import CheckpointManager
from training.experiment_manager import ExperimentManager
from training.validator import Validator
from training.batch_manager import BatchManager

from utils.logger import get_model_logger
from utils.performance_profiler import PerformanceProfiler, ExecutionMode

logger = get_model_logger("Trainer")


class Trainer:
    """
    Master Orchestrator for Phase 3 Training.
    Handles Mixed Precision, Curriculum Learning, Multi-Task Adaptive Loss, and Checkpointing.
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # ── Execution Mode ─────────────────────────────────────────────
        try:
            self._exec_mode = ExecutionMode(config.execution_mode)
        except ValueError:
            logger.warning(
                "Unknown execution_mode '%s'. Defaulting to PRODUCTION.", config.execution_mode
            )
            self._exec_mode = ExecutionMode.PRODUCTION

        logger.info("Execution mode: %s", self._exec_mode.value.upper())

        # ── Model + optional DataParallel ──────────────────────────────
        if config.multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        # ── torch.compile (PyTorch 2.x) ───────────────────────────────
        if config.use_torch_compile:
            if hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("torch.compile enabled (mode=reduce-overhead).")
                except Exception as exc:
                    logger.warning("torch.compile failed — falling back to eager. Reason: %s", exc)
            else:
                logger.info("torch.compile not available on this PyTorch version — skipping.")
            
        # ── Optimization ───────────────────────────────────────────────
        self.optimizer = OptimizerFactory.create(
            config.optimizer, self.model.parameters(), config.learning_rate, config.weight_decay
        )
        self.scheduler = SchedulerFactory.create(
            config.scheduler, self.optimizer, config.epochs, config.min_lr
        )
        
        # ── Loss & Curriculum ─────────────────────────────────────────
        self.physics_loss_fn = PhysicsLoss()
        
        # In this multi-task setting, we have: Topside, NetTEC, GNSS(x7), Density. 
        # The LossManager handles Adaptive Multi-Task weighting and shape validations.
        self.num_tasks = 4 
        self.loss_manager = LossManager(strategy=config.adaptive_loss_strategy, num_tasks=self.num_tasks).to(self.device)
        self.curriculum = CurriculumLearning(enable=config.enable_curriculum, advance_patience=config.curriculum_advance_patience)
        
        # ── Mixed Precision ────────────────────────────────────────────
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler(device='cuda', enabled=config.mixed_precision)
        else:
            self.scaler = torch.amp.GradScaler(device='cpu', enabled=False)
        
        # ── Validation & Managers ──────────────────────────────────────
        self.validator = Validator(self.model, self.device, execution_mode=config.execution_mode)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, save_top_k=config.save_top_k)
        
        # Use upgraded ExperimentManager (Auto-increments Experiment_001, etc.)
        self.experiment = ExperimentManager("experiments")
        self.experiment.save_config(config)
        
        # ── Callbacks ─────────────────────────────────────────────────
        self.callbacks = CallbackList([
            EarlyStopping(patience=config.early_stopping_patience),
            GradientMonitor(self.model, log_freq=50),
            self.checkpoint_manager
        ])

        # ── Performance Profiler ───────────────────────────────────────
        self.profiler = PerformanceProfiler(
            mode=self._exec_mode,
            output_dir="results/performance",
            enabled=True,  # Profiler init decides actual enabling based on mode
        )

        # ── One-time batch validation flag ────────────────────────────
        # Batch interface validation runs only once (first epoch) then is skipped.
        self._batch_interface_verified: bool = False
        
        logger.info("Trainer successfully initialized.")

    # ──────────────────────────────────────────────────────────────────
    # fit()
    # ──────────────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader):
        """Main training loop."""
        self.callbacks.on_train_begin()

        for epoch in range(1, self.config.epochs + 1):
            self.callbacks.on_epoch_begin(epoch)
            self.profiler.start_epoch()

            # ── Train one epoch ───────────────────────────────────────
            with self.profiler.section("training_epoch"):
                train_metrics = self._train_epoch(train_loader, epoch)

            # ── Validate ──────────────────────────────────────────────
            with self.profiler.section("validation"):
                val_metrics = self.validator.validate(val_loader)

            # ── Update Curriculum & Schedulers ────────────────────────
            self.curriculum.on_epoch_end(val_metrics["val_loss"])
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            else:
                self.scheduler.step()

            # ── Log ───────────────────────────────────────────────────
            with self.profiler.section("logging"):
                logs = {**train_metrics, **val_metrics}
                self.experiment.log_metrics(epoch, logs)
                self.callbacks.on_epoch_end(epoch, logs)

            # ── Checkpoint ────────────────────────────────────────────
            with self.profiler.section("checkpoint"):
                is_best = (
                    val_metrics["val_loss"] < self.checkpoint_manager.best_losses[0][0]
                    if self.checkpoint_manager.best_losses
                    else True
                )
                self.checkpoint_manager.save(
                    epoch, self.model, self.optimizer, val_metrics["val_loss"], is_best=is_best
                )

            # ── Epoch speed report ────────────────────────────────────
            n_samples = len(train_loader.dataset)
            n_batches = len(train_loader)
            epoch_rec = self.profiler.record_epoch(epoch, n_samples, n_batches)
            logger.info(self.profiler.get_speed_summary(epoch_rec))

            # ── Early Stopping Check ──────────────────────────────────
            for cb in self.callbacks.callbacks:
                if isinstance(cb, EarlyStopping) and cb.stop_training:
                    logger.info("Training halted early.")
                    self.callbacks.on_train_end()
                    self._finalize(epoch)
                    return

        self.callbacks.on_train_end()
        self._finalize(self.config.epochs)

    def _finalize(self, last_epoch: int) -> None:
        """Save performance reports after training ends."""
        try:
            self.profiler.generate_report()
            logger.info("Performance report saved to results/performance/")
        except Exception as exc:
            logger.warning("Could not generate performance report: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # _train_epoch()
    # ──────────────────────────────────────────────────────────────────

    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss_val = 0.0
        mse_criterion  = nn.MSELoss()

        n_samples     = len(train_loader.dataset)
        epoch_start   = time.perf_counter()

        # ── One-time batch interface validation ───────────────────────
        # Run only on the very first epoch, then never again.
        if not self._batch_interface_verified and self._exec_mode == ExecutionMode.DEVELOPMENT:
            _sample_batch = next(iter(train_loader))
            logger.info("Validating batch interface before training begins...")
            BatchManager.validate_batch(_sample_batch, verbose=True)
            BatchManager.generate_debug_report(
                _sample_batch,
                output_path="results/debug/batch_interface_report.json",
            )
            del _sample_batch
            self._batch_interface_verified = True
        elif not self._batch_interface_verified:
            # Production / benchmark: do minimal one-shot check, suppress verbose output
            _sample_batch = next(iter(train_loader))
            try:
                BatchManager.validate_batch(_sample_batch, verbose=False)
            except Exception as exc:
                logger.error("Batch interface validation failed: %s", exc)
                raise
            del _sample_batch
            self._batch_interface_verified = True

        # ── Build tqdm iterator ───────────────────────────────────────
        use_pbar = self.config.enable_progress_bar and _TQDM_AVAILABLE
        iterator = (
            _tqdm(
                train_loader,
                desc=f"Epoch {epoch:>3d}/{self.config.epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            )
            if use_pbar
            else train_loader
        )

        batches_seen = 0
        samples_seen = 0

        for batch_idx, batch in enumerate(iterator):
            # Curriculum Filtering
            if not self.curriculum.filter_batch(batch):
                continue

            self.callbacks.on_batch_begin(batch_idx)

            # ── Extract inputs and targets via BatchManager ───────────
            with self.profiler.section("dataloader"):
                inputs = BatchManager.get_inputs(batch, self.device)
                temporal_seq   = inputs["temporal_seq"]
                physics_feats  = inputs["physics_feats"]
                geo_feats      = inputs["geo_feats"]
                storm_feats    = inputs["storm_feats"]
                bottomside_tec = inputs["bottomside_tec"]
                geometry_feats = inputs["geometry_feats"]   # May be None
                targets = BatchManager.get_all_targets(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward pass with AMP ─────────────────────────────────
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with self.profiler.section("forward_pass"):
                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self.config.mixed_precision and self.device.type == "cuda",
                ):
                    output = self.model(
                        temporal_seq, physics_feats, geo_feats, storm_feats,
                        bottomside_tec, geometry_feats
                    )

            # ── Loss ─────────────────────────────────────────────────
            with self.profiler.section("loss"):
                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self.config.mixed_precision and self.device.type == "cuda",
                ):
                    total_loss, loss_dict = self.loss_manager(output, targets)

                    # Physics Constraints
                    phys_penalty, phys_logs = self.physics_loss_fn(
                        output.topside_tec, output.net_tec, output.electron_density,
                        output.physics_intermediates, temporal_seq
                    )
                    loss = total_loss + (self.config.physics_penalty_weight * phys_penalty)

            # ── Backward ─────────────────────────────────────────────
            with self.profiler.section("backward_pass"):
                self.scaler.scale(loss).backward()

            # ── Optimizer ────────────────────────────────────────────
            with self.profiler.section("optimizer"):
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss_val += loss.item()
            batches_seen   += 1
            batch_sz        = temporal_seq.size(0)
            samples_seen   += batch_sz

            # ── Update tqdm postfix ───────────────────────────────────
            if use_pbar:
                elapsed  = time.perf_counter() - epoch_start
                sps      = samples_seen / max(elapsed, 1e-9)
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sps=f"{sps:.0f}",
                    refresh=False,
                )

            batch_logs = {"loss": loss.item()}
            batch_logs.update(loss_dict)
            batch_logs.update(phys_logs)
            self.callbacks.on_batch_end(batch_idx, batch_logs)

        avg_loss = total_loss_val / max(batches_seen, 1)
        return {"train_loss": avg_loss}
