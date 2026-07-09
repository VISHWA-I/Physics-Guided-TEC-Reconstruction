import torch
import torch.nn as nn
from typing import Tuple

from utils.logger import get_model_logger

logger = get_model_logger("PredictionUncertaintyHead")

class PredictionUncertaintyHead(nn.Module):
    """
    Computes Aleatoric uncertainty (data noise) by directly estimating the variance (sigma^2).
    In a full production setting, Monte Carlo (Epistemic) dropout can be achieved by running 
    the entire `HybridModel` forward pass multiple times, but this head handles the 
    analytic Aleatoric confidence intervals directly.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.variance_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus() # Variance must be positive
        )
        logger.info("PredictionUncertaintyHead initialized.")

    def forward(self, hidden_rep: torch.Tensor, point_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimates uncertainty bounds.

        Args:
            hidden_rep (torch.Tensor): Deep latent state. (Batch, Seq, Hidden)
            point_prediction (torch.Tensor): The central prediction (e.g., Topside TEC). (Batch, Seq, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Confidence Score (inverse of variance, normalized roughly to 0-1)
                - Lower Bound
                - Upper Bound
        """
        # Estimate variance (sigma^2)
        variance = self.variance_estimator(hidden_rep)
        
        # Standard deviation (sigma)
        sigma = torch.sqrt(variance + 1e-6)
        
        # 95% Confidence Interval (~1.96 sigma)
        lower_bound = point_prediction - 1.96 * sigma
        upper_bound = point_prediction + 1.96 * sigma
        
        # Confidence score (heuristic mapping of variance to a 0-1 scale)
        confidence_score = torch.exp(-variance)
        
        return confidence_score, lower_bound, upper_bound
