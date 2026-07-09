import torch
import torch.nn as nn
from typing import Dict, Tuple

from utils.logger import get_model_logger
from training.sequence_target_manager import ShapeMismatchError

logger = get_model_logger("LossManager")

class LossManager(nn.Module):
    """
    Centralized Loss Orchestrator.
    Handles strict shape validation, multi-task aggregation (GradNorm/DWA),
    and physics constraint losses.
    """
    
    def __init__(self, strategy: str = "gradnorm", num_tasks: int = 4):
        super().__init__()
        self.strategy = strategy
        self.num_tasks = num_tasks
        self.criterion = nn.MSELoss()
        
        # Adaptive weights for tasks
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
        logger.info(f"LossManager initialized with strategy: {self.strategy}")

    def _validate_shape(self, pred: torch.Tensor, target: torch.Tensor, name: str):
        """Strictly validates that prediction and target perfectly match."""
        if pred.shape != target.shape:
            raise ShapeMismatchError(
                f"[{name}] Shape Mismatch! Prediction: {tuple(pred.shape)}, Target: {tuple(target.shape)}. "
                "PyTorch broadcasting is strictly forbidden."
            )

    def forward(self, outputs, targets: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes aggregated loss and component dictionary.
        
        Args:
            outputs (ModelOutput): Output object/dict from HybridModel.
            targets (dict): Targets from SequenceTargetManager.
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Total loss and dictionary of individual task losses.
        """
        # Extract predictions
        pred_topside = outputs.topside_tec
        pred_net = outputs.net_tec
        pred_density = outputs.electron_density
        pred_delay = outputs.gnss_delays["vertical_delay"] # Proxy for delay cluster
        
        # Extract targets
        targ_topside = targets["topside"]
        targ_net = targets["net"]
        targ_density = targets["density"]
        targ_delay = targets["delay"]
        
        # 1. Strict Shape Validation
        self._validate_shape(pred_topside, targ_topside, "Topside TEC")
        self._validate_shape(pred_net, targ_net, "Net TEC")
        self._validate_shape(pred_density, targ_density, "Electron Density")
        self._validate_shape(pred_delay, targ_delay, "GNSS Delay")
        
        # 2. Compute Individual Losses
        loss_topside = self.criterion(pred_topside, targ_topside)
        loss_net = self.criterion(pred_net, targ_net)
        loss_density = self.criterion(pred_density, targ_density)
        loss_delay = self.criterion(pred_delay, targ_delay)
        
        losses = [loss_topside, loss_net, loss_density, loss_delay]
        
        # 3. Multi-Task Aggregation
        total_loss = 0.0
        
        if self.strategy == "gradnorm":
            # Simplified GradNorm logic (applies weighted sum using task parameters)
            weights = torch.softmax(self.task_weights, dim=0) * self.num_tasks
            for i, l in enumerate(losses):
                total_loss += weights[i] * l
        elif self.strategy == "homoscedastic":
            for i, l in enumerate(losses):
                total_loss += 0.5 * torch.exp(-self.task_weights[i]) * l + 0.5 * self.task_weights[i]
        else: # static
            total_loss = sum(losses)
            
        # Logging Dict
        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_topside": loss_topside.item(),
            "loss_net": loss_net.item(),
            "loss_density": loss_density.item(),
            "loss_delay": loss_delay.item()
        }
        
        return total_loss, loss_dict
