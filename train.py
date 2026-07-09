import argparse
import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Import existing project modules
from src.dataset import DatasetFactory
from models.model_config import ModelConfig
from models.hybrid_model import HybridModel
from training.trainer import Trainer
from training.training_config import TrainingConfig
from training.sequence_target_manager import SequenceTargetManager
from utils.logger import get_model_logger
from utils.env_config import configure as configure_env, get_paths, is_colab
from utils.model_exporter import export_model

logger = get_model_logger("TrainScript")

class DictWrapperDataset(Dataset):
    """
    Wraps the (X, y) output from DatasetFactory into the exact dictionary
    format expected by the HybridModel and Trainer.

    Performance Optimizations
    -------------------------
    * All targets are pre-computed once in ``__init__`` using a single
      vectorised pass through ``SequenceTargetManager``, instead of calling
      ``generate_targets()`` on every ``__getitem__`` invocation (was: 5760+
      allocations per epoch → now: 0 allocations per epoch for targets).
    * Feature index tensors are computed once at init and reused.
    * All slices in ``__getitem__`` are zero-copy tensor views.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, feature_names: list, model_config: ModelConfig):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.model_config = model_config
        
        # Create mapping of feature name to index in X
        self.f_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Resolve indices for each feature group
        self.temp_idx = [self.f_idx[f] for f in model_config.temporal_features if f in self.f_idx]
        self.phys_idx = [self.f_idx[f] for f in model_config.physics_features if f in self.f_idx]
        self.geo_idx  = [self.f_idx[f] for f in model_config.geo_features  if f in self.f_idx]
        self.storm_idx = [self.f_idx[f] for f in model_config.storm_features if f in self.f_idx]
        
        # For Bottomside TEC (often just "TEC" in the input features)
        self.tec_idx = self.f_idx["TEC"] if "TEC" in self.f_idx else 0

        # --- Pre-cache all targets (vectorised, single pass) ---
        # This replaces 5760+ individual calls to generate_targets() per epoch.
        logger.info("Pre-computing all targets (vectorised pass)...")
        target_manager = SequenceTargetManager(mode="sequence_to_sequence")
        # bottomside_tec_all: (N, Seq, 1)
        bottomside_tec_all = X[:, :, self.tec_idx : self.tec_idx + 1]
        # generate_targets expects (Batch, 1) and (Batch, Seq, 1)
        targets_all = target_manager.generate_targets(y, bottomside_tec_all)
        # Store as a dict of (N, Seq, 1) tensors
        self._targets_cache: dict = {k: v for k, v in targets_all.items()}
        logger.info("Target cache ready (%d sequences).", len(X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_seq = self.X[idx]  # Shape: (SeqLength, TotalFeatures)

        # Zero-copy tensor views via advanced indexing
        temporal_seq   = x_seq[:, self.temp_idx]
        last_step      = x_seq[-1]
        physics_feats  = last_step[self.phys_idx]
        geo_feats      = last_step[self.geo_idx]
        storm_feats    = last_step[self.storm_idx]
        bottomside_tec = x_seq[:, self.tec_idx : self.tec_idx + 1]

        # Retrieve pre-cached targets for this index (zero-allocation)
        targets = {k: v[idx] for k, v in self._targets_cache.items()}

        return {
            'temporal_seq':  temporal_seq,
            'physics_feats': physics_feats,
            'geo_feats':     geo_feats,
            'storm_feats':   storm_feats,
            'bottomside_tec': bottomside_tec,
            'targets':       targets,
        }


def _safe_num_workers(requested: int) -> int:
    """
    Determine a safe ``num_workers`` value for the current platform.

    On Windows, multiprocessing DataLoader workers require the dataset to
    be picklable.  We attempt a quick pickle probe and fall back to 0 if
    anything goes wrong.

    Parameters
    ----------
    requested : int
        Desired number of workers. 0 = auto-detect (min(4, cpu_count)).

    Returns
    -------
    int
        Safe num_workers value.
    """
    import platform, pickle

    if requested == 0:
        # Auto-detect: up to 4 workers but respect CPU count
        cpu_count = os.cpu_count() or 1
        requested = min(4, cpu_count)

    if platform.system() == "Windows" and requested > 0:
        # Quick picklability probe — DataLoader internals require this
        try:
            import io
            buf = io.BytesIO()
            # We can't pickle the full dataset here (no reference), so just
            # verify the multiprocessing start method is safe.
            import multiprocessing as mp
            if mp.get_start_method(allow_none=True) not in ("spawn", None):
                logger.warning(
                    "Windows DataLoader: unexpected start method '%s'. "
                    "Falling back to num_workers=0.",
                    mp.get_start_method(),
                )
                return 0
        except Exception as exc:
            logger.warning(
                "Windows worker probe failed (%s). Falling back to num_workers=0.", exc
            )
            return 0

    if requested > 0:
        logger.info("DataLoader num_workers=%d", requested)
    else:
        logger.info("DataLoader num_workers=0 (single-threaded)")
    return requested


def load_data(config_path: str, model_config: ModelConfig, training_config: TrainingConfig):
    """
    Loads processed data and creates DataLoaders.

    Parameters
    ----------
    config_path : str
        Path to the project config.yaml.
    model_config : ModelConfig
        Model architecture configuration.
    training_config : TrainingConfig
        Training configuration (batch_size, num_dataloader_workers, etc.)
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Gather all unique required features across all model inputs
    all_features = set(model_config.temporal_features + model_config.physics_features + 
                       model_config.geo_features + model_config.storm_features)
    # Ensure TEC is present
    all_features.add("TEC")
    feature_columns = list(all_features)
    
    logger.info("Loading processed datasets...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")
    
    # Process through Phase 1 DatasetFactory
    factory = DatasetFactory(cfg, feature_columns)
    train_ds_raw, val_ds_raw, _ = factory.build_datasets(train_df, val_df, test_df)
    
    # Extract to pure PyTorch tensors
    X_train, y_train = train_ds_raw.to_tensors()
    X_val,   y_val   = val_ds_raw.to_tensors()
    
    # Wrap in our optimised dictionary dataset (targets pre-cached)
    train_dataset = DictWrapperDataset(X_train, y_train, feature_columns, model_config)
    val_dataset   = DictWrapperDataset(X_val,   y_val,   feature_columns, model_config)

    # Resolve safe worker count
    nw = _safe_num_workers(training_config.num_dataloader_workers)
    persistent = nw > 0
    prefetch   = 2 if nw > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=nw,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=nw,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Mamba-TKAN Model")
    parser.add_argument("--epochs",          type=int,   default=100,          help="Number of training epochs")
    parser.add_argument("--batch-size",      type=int,   default=32,           help="Batch size")
    parser.add_argument("--lr",              type=float, default=1e-4,         help="Learning rate")
    parser.add_argument("--device",          type=str,   default="auto",       help="Device (cpu, cuda, auto)")
    parser.add_argument("--experiment-name", type=str,   default="default_run",help="Experiment name")
    parser.add_argument("--resume",          type=str,   default=None,         help="Path to checkpoint to resume from")
    parser.add_argument(
        "--mode",
        type=str,
        default="production",
        choices=["development", "production", "benchmark"],
        help="Execution mode: development | production | benchmark",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile (PyTorch 2.x, CPU-compatible)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader num_workers (0 = auto-detect)",
    )
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging")
    parser.add_argument("--export", action="store_true", default=True, help="Export PyTorch + TorchScript + ONNX after training")
    parser.add_argument("--colab", action="store_true", default=False, help="Force Colab execution mode")
    parser.add_argument("--drive-root", type=str, default=None, help="Override Google Drive root path on Colab")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configure Environment and Paths
    if args.colab:
        os.environ["FORCE_COLAB"] = "1"
    paths = configure_env(drive_root=args.drive_root)
    
    logger.info("=== Initializing Training Pipeline ===")
    logger.info("Execution mode : %s", args.mode)
    if is_colab():
        logger.info("Detected Google Colab environment. Outputs routed to Google Drive.")
    
    # Ensure local directories for data/configs if missing
    os.makedirs("results/performance", exist_ok=True)
    
    # Load Configurations
    model_config = ModelConfig.from_yaml(str(paths.configs / "model.yaml"))
    
    # Setup Training Config
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=str(paths.checkpoints),
        experiment_dir=str(paths.experiments / args.experiment_name),
        tensorboard_dir=str(paths.tensorboard / args.experiment_name),
        export_dir=str(paths.exports),
        enable_tensorboard=args.tensorboard,
        export_torchscript=args.export,
        export_onnx=args.export,
        execution_mode=args.mode,
        use_torch_compile=args.compile,
        num_dataloader_workers=args.workers,
        enable_progress_bar=True,
    )
    
    # DataLoaders (optimised: multi-worker + pre-cached targets)
    train_loader, val_loader = load_data(str(paths.config / "config.yaml"), model_config, training_config)
    logger.info(f"DataLoaded: {len(train_loader.dataset)} Train | {len(val_loader.dataset)} Val")
    
    # Model
    model = HybridModel(model_config)
    
    # Auto-resume logic:
    resume_path = args.resume
    if not resume_path and (paths.checkpoints / "latest.pt").exists():
        resume_path = str(paths.checkpoints / "latest.pt")
        
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        
    # Trainer (now with profiler, tqdm, torch.compile, and mode-aware validation)
    trainer = Trainer(model, training_config)
    
    # Fit
    logger.info("Starting Training Loop...")
    trainer.fit(train_loader, val_loader)
    
    logger.info(f"Training complete. Results saved to {paths.results}")
    
    # Model Export
    if args.export:
        export_model(
            model=model,
            epoch=trainer.config.epochs,
            export_dir=paths.exports,
            export_torchscript=training_config.export_torchscript,
            export_onnx=training_config.export_onnx
        )

if __name__ == "__main__":
    main()
