import torch
import torch.nn as nn

from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("StormEncoder")

class StormContextEncoder(nn.Module):
    """
    Encodes space weather storm indices (Kp, Dst, Ap, F10.7, SSN).
    These features provide a macroeconomic view of the current space environment.
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.1, norm_type: str = "rmsnorm"):
        """
        Initializes the StormContextEncoder.

        Args:
            in_features (int): Number of storm features (default: 5).
            hidden_dim (int): Target hidden dimension.
            dropout (float): Dropout probability.
            norm_type (str): Type of normalization to use.
        """
        super().__init__()
        
        if in_features <= 0:
            raise ValueError(f"StormContextEncoder requires >0 input features, got {in_features}")
            
        # Single layer MLP for context embedding
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            get_normalization_layer(norm_type, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        logger.info(f"StormContextEncoder initialized mapping {in_features} -> {hidden_dim}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Storm Context Encoder.

        Args:
            x (torch.Tensor): Raw storm features. Expected shape: (Batch, Features)

        Returns:
            torch.Tensor: Storm Embedding. Shape: (Batch, HiddenDimension)
        """
        if x.dim() != 2:
            raise ValueError(f"StormContextEncoder expects 2D input (Batch, Features), got {x.dim()}D")
            
        return self.projection(x)
