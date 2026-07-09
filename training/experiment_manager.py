import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from utils.logger import get_model_logger
from utils.tensorboard_logger import get_tensorboard_logger

logger = get_model_logger("ExperimentManager")

class ExperimentManager:
    """
    Tracks and saves experiment configurations, hyperparameters, and metric curves.
    Ensures reproducibility of research runs.
    Automatically increments experiment folders (Experiment_001, Experiment_002, etc.).
    """
    
    def __init__(self, base_experiment_dir: str = "experiments"):
        self.base_dir = Path(base_experiment_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-increment logic
        existing_exps = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("Experiment_")]
        exp_nums = []
        for d in existing_exps:
            try:
                num = int(d.name.split("_")[1])
                exp_nums.append(num)
            except ValueError:
                pass
                
        next_num = max(exp_nums) + 1 if exp_nums else 1
        self.exp_name = f"Experiment_{next_num:03d}"
        self.exp_dir = self.base_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_log = []
        logger.info(f"Initialized Tracking for {self.exp_name}")
        
        # Initialize TensorBoard
        # If running on Colab, this will automatically resolve to Drive via env_config if setup correctly,
        # but we can also just rely on training config passing it later or use the default.
        # Actually, we will instantiate it without explicit dir so it uses env_config defaults,
        # or we could wait until save_config to get the explicit tensorboard_dir.
        # For simplicity, we just use the exp_dir / "tensorboard"
        self.tb = get_tensorboard_logger(log_dir=self.exp_dir / "tensorboard")
        
    def save_config(self, config: Any):
        """Saves the dataclass config as a JSON file."""
        config_path = self.exp_dir / "config.json"
        
        if hasattr(config, "__dict__"):
            conf_dict = config.__dict__
        else:
            conf_dict = dict(config)
            
        with open(config_path, "w") as f:
            json.dump(conf_dict, f, indent=4)
            
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Logs metrics for an epoch, flushes to CSV, and generates plots."""
        metrics["epoch"] = epoch
        self.metrics_log.append(metrics)
        
        df = pd.DataFrame(self.metrics_log)
        
        # Save CSVs
        df.to_csv(self.exp_dir / "metrics.csv", index=False)
        
        # Log to TensorBoard
        self.tb.log_scalars(epoch, **metrics)
        self.tb.flush()
        
        # Generate generic loss curves
        self._plot_metrics(df)
        
    def _plot_metrics(self, df: pd.DataFrame):
        """Automatically generates standard scientific plots."""
        # 1. Total Loss Curve
        if "train_loss" in df.columns and "val_loss" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
            plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.title("Learning Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.exp_dir / "learning_curve.png")
            plt.close()
            
        # 2. Component Losses
        loss_components = [c for c in df.columns if c.startswith("loss_") and c not in ["loss_topside", "loss_net", "loss_density", "loss_delay"]] 
        # Actually, let's plot the standard ones explicitly
        components = ["loss_topside", "loss_net", "loss_density", "loss_delay"]
        valid_comps = [c for c in components if c in df.columns]
        if valid_comps:
            plt.figure(figsize=(10, 6))
            for c in valid_comps:
                plt.plot(df["epoch"], df[c], label=c)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Multi-Task Loss Components")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.exp_dir / "loss_components.png")
            plt.close()
