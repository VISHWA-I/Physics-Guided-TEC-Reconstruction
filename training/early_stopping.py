from training.callbacks import Callback
from utils.logger import get_model_logger

logger = get_model_logger("EarlyStopping")

class EarlyStopping(Callback):
    """
    Halts training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stop_training = False
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"Early Stopping triggered at epoch {epoch}. Best Val Loss: {self.best_loss:.4f}")
                self.stop_training = True
