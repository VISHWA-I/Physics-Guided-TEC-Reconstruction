from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils.logger import get_model_logger

logger = get_model_logger("CheckpointManager")

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints, including optimizers and metrics.
    """
    
    def __init__(self, directory: str | Path):
        """
        Initializes the CheckpointManager.

        Args:
            directory (str | Path): The directory to save/load checkpoints.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, 
                        model: nn.Module, 
                        optimizer: Optimizer, 
                        epoch: int, 
                        metrics: Dict[str, float], 
                        is_best: bool = False,
                        filename: str = "latest_checkpoint.pt") -> None:
        """
        Saves the training state to a file.

        Args:
            model (nn.Module): The model being trained.
            optimizer (Optimizer): The optimizer.
            epoch (int): The current epoch number.
            metrics (Dict[str, float]): The validation/training metrics.
            is_best (bool): If True, also saves a copy as 'best_checkpoint.pt'.
            filename (str): The filename for the standard checkpoint.
        """
        checkpoint_path = self.directory / filename
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path} (Epoch {epoch})")
        
        if is_best:
            best_path = self.directory / "best_checkpoint.pt"
            torch.save(state, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")

    def load_checkpoint(self, 
                        model: nn.Module, 
                        optimizer: Optional[Optimizer] = None, 
                        filename: str = "latest_checkpoint.pt",
                        device: Optional[torch.device] = None) -> Tuple[int, Dict[str, float]]:
        """
        Loads the training state from a file.

        Args:
            model (nn.Module): The model to load weights into.
            optimizer (Optional[Optimizer]): The optimizer to load state into (if resuming).
            filename (str): The filename of the checkpoint to load.
            device (Optional[torch.device]): Target device for loading.

        Returns:
            Tuple[int, Dict[str, float]]: The epoch number and metrics dictionary from the checkpoint.
            
        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = self.directory / filename
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file {checkpoint_path} not found.")
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state dict from {checkpoint_path}")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded optimizer state dict from {checkpoint_path}")
            
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Resumed from epoch {epoch} with metrics {metrics}")
        return epoch, metrics

