import torch
import torch.nn as nn
from typing import Dict, Tuple

class PhysicsLoss(nn.Module):
    """
    Implements physical constraints as soft penalties for the neural network.
    Instead of hard clamping, we apply penalty gradients when the model predicts
    physically impossible states.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, 
                topside_pred: torch.Tensor, 
                net_tec_pred: torch.Tensor,
                electron_density_pred: torch.Tensor,
                physics_intermediates: Dict[str, torch.Tensor],
                temporal_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the aggregate physics penalty.
        
        Args:
            topside_pred: (Batch, Seq, 1)
            net_tec_pred: (Batch, Seq, 1)
            electron_density_pred: (Batch, Seq, 1)
            physics_intermediates: Dict containing bottomside_tec_input, base_net_tec, etc.
            temporal_seq: Sequence used for smoothness constraint.
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 
                - The total physics penalty scalar.
                - A dictionary logging the individual penalty components.
        """
        loss = 0.0
        logs = {}
        
        # 1. Positivity Constraints (TEC and Density cannot be negative)
        # ReLU(-x) is 0 if x > 0, and linearly penalizes if x < 0.
        topside_neg_penalty = torch.nn.functional.relu(-topside_pred).mean()
        density_neg_penalty = torch.nn.functional.relu(-electron_density_pred).mean()
        
        loss += topside_neg_penalty + density_neg_penalty
        logs["penalty_topside_neg"] = topside_neg_penalty.item()
        logs["penalty_density_neg"] = density_neg_penalty.item()
        
        # 2. Net TEC Consistency
        # The NetTECHead predicts a final value that should closely hover around the analytical sum.
        # Too large of a deviation implies a physical break.
        if "base_net_tec" in physics_intermediates:
            base_net = physics_intermediates["base_net_tec"]
            # We penalize heavy deviations from the Base Net TEC. 
            # A small deviation is expected (plasmaspheric tail), but not massive variance.
            consistency_loss = self.mse(net_tec_pred, base_net) * 0.1 # Soft constraint
            loss += consistency_loss
            logs["penalty_consistency"] = consistency_loss.item()
            
        # 3. Temporal Smoothness
        # Penalize massive high-frequency jumps in consecutive time steps (first derivative constraint)
        # d_pred = |pred[t] - pred[t-1]|
        topside_diff = torch.abs(topside_pred[:, 1:, :] - topside_pred[:, :-1, :]).mean()
        # Mild constraint to prevent jitter
        smoothness_penalty = topside_diff * 0.05
        loss += smoothness_penalty
        logs["penalty_smoothness"] = smoothness_penalty.item()
        
        logs["total_physics_penalty"] = loss.item() if isinstance(loss, torch.Tensor) else loss
        return loss, logs
