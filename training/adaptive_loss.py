import torch
import torch.nn as nn
from typing import Dict, List, Optional

class AdaptiveLoss(nn.Module):
    """
    Dynamically balances multiple loss functions (Multi-Task Learning) without 
    manual lambda tuning.
    Supported strategies: UncertaintyWeighting, GradNorm, DWA.
    """

    def __init__(self, num_tasks: int, strategy: str = "UncertaintyWeighting"):
        super().__init__()
        self.num_tasks = num_tasks
        self.strategy = strategy
        
        if self.strategy == "UncertaintyWeighting":
            # Learnable log-variances for each task (Kendall et al. 2018)
            # log(sigma^2) is used for numerical stability
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        elif self.strategy == "DWA":
            # Dynamic Weight Averaging
            # Requires tracking losses over previous epochs
            self.prev_losses: Optional[torch.Tensor] = None
            self.curr_losses: Optional[torch.Tensor] = None
            self.temperature = 2.0
            self.task_weights = nn.Parameter(torch.ones(num_tasks), requires_grad=False)
        elif self.strategy == "GradNorm":
            # GradNorm requires hooking into a shared layer. This will be implemented in the Trainer.
            self.task_weights = nn.Parameter(torch.ones(num_tasks))
        else:
            raise ValueError(f"Unknown AdaptiveLoss strategy: {strategy}")

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Combines a list of task losses into a single scalar loss.

        Args:
            losses (List[torch.Tensor]): List of individual task losses.

        Returns:
            torch.Tensor: The adaptively weighted total loss.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")
            
        stacked_losses = torch.stack(losses)
        
        if self.strategy == "UncertaintyWeighting":
            # Loss = sum( L_i / (2 * exp(s_i)) + s_i / 2 )
            # We use s_i = log(sigma_i^2)
            precision = torch.exp(-self.log_vars)
            total_loss = torch.sum(precision * stacked_losses + self.log_vars)
            return total_loss
            
        elif self.strategy == "DWA":
            # Uses pre-computed weights (updated at epoch end)
            weighted = self.task_weights.to(stacked_losses.device) * stacked_losses
            return torch.sum(weighted)
            
        elif self.strategy == "GradNorm":
            # Weights are updated via a separate GradNorm backward pass in the Trainer.
            # Here we just apply the current weights.
            weighted = self.task_weights.to(stacked_losses.device) * stacked_losses
            return torch.sum(weighted)
            
        return torch.sum(stacked_losses)

    def update_dwa_weights(self, avg_epoch_losses: torch.Tensor):
        """Called at the end of an epoch to update DWA weights."""
        if self.strategy != "DWA":
            return
            
        if self.prev_losses is None:
            self.prev_losses = avg_epoch_losses
            return
            
        if self.curr_losses is None:
            self.curr_losses = avg_epoch_losses
            
        # w_i(t) = L_i(t-1) / L_i(t-2)
        rates = self.curr_losses / (self.prev_losses + 1e-8)
        
        # Softmax with temperature
        exp_rates = torch.exp(rates / self.temperature)
        self.task_weights.data = (self.num_tasks * exp_rates / torch.sum(exp_rates)).detach()
        
        # Shift
        self.prev_losses = self.curr_losses
        self.curr_losses = avg_epoch_losses
