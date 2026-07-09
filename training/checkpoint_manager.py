import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any

from training.callbacks import Callback
from utils.logger import get_model_logger

logger = get_model_logger("CheckpointManager")

class CheckpointManager(Callback):
    """
    Handles saving and restoring model states, optimizer states, and epoch counts.
    """
    
    def __init__(self, save_dir: str, save_top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.best_losses = [] # list of tuples (loss, filepath)
        
    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
             val_loss: float, is_best: bool = False):
        """Saves a checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save latest
        latest_path = self.save_dir / "latest.pt"
        torch.save(state, latest_path)
        
        if is_best:
            best_path = self.save_dir / "best.pt"
            shutil.copyfile(latest_path, best_path)
            logger.info(f"Saved new best model at epoch {epoch} with loss {val_loss:.4f}")
            
        # Top K management
        filename = self.save_dir / f"epoch_{epoch}_loss_{val_loss:.4f}.pt"
        torch.save(state, filename)
        self.best_losses.append((val_loss, filename))
        self.best_losses.sort(key=lambda x: x[0])
        
        if len(self.best_losses) > self.save_top_k:
            # Remove worst
            worst_loss, worst_file = self.best_losses.pop(-1)
            if worst_file.exists():
                os.remove(worst_file)
                
    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
        """Loads a checkpoint."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path} (Epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
