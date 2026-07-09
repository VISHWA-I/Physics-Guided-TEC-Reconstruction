import torch
from training.callbacks import Callback
from utils.logger import get_model_logger

logger = get_model_logger("GradientMonitor")

class GradientMonitor(Callback):
    """
    Logs exploding or vanishing gradients during the backward pass.
    """
    
    def __init__(self, model: torch.nn.Module, log_freq: int = 100):
        self.model = model
        self.log_freq = log_freq
        self.global_step = 0
        
    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        if self.global_step % self.log_freq != 0:
            return
            
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Log to dictionary so trainer can capture it
        if logs is not None:
            logs["grad_norm"] = total_norm
            
        if total_norm > 100.0:
            logger.warning(f"High Gradient Norm detected: {total_norm:.2f} at step {self.global_step}")
        elif total_norm < 1e-4:
            logger.warning(f"Vanishing Gradient detected: {total_norm:.6f} at step {self.global_step}")
