import torch
import torch.nn as nn

from models.normalization import get_normalization_layer
from models.residual_block import ResidualBlock
from utils.logger import get_model_logger

logger = get_model_logger("PhysicsEncoder")

class PhysicsEncoder(nn.Module):
    """
    Encodes the physical parameters (foF2, hmF2, etc.) derived from the GIRO digisonde.
    Transforms raw features into a dense vector compatible with the hidden dimension.
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.1, norm_type: str = "rmsnorm"):
        """
        Initializes the PhysicsEncoder.

        Args:
            in_features (int): Number of physical features (default: 9).
            hidden_dim (int): Target hidden dimension for the fusion network.
            dropout (float): Dropout probability.
            norm_type (str): Type of normalization to use.
        """
        super().__init__()
        
        if in_features <= 0:
            raise ValueError(f"PhysicsEncoder requires >0 input features, got {in_features}")
            
        # We project the physical features in a two-step MLP
        intermediate_dim = max(in_features * 4, hidden_dim // 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, intermediate_dim),
            get_normalization_layer(norm_type, intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            get_normalization_layer(norm_type, hidden_dim)
        )
        
        logger.info(f"PhysicsEncoder initialized mapping {in_features} -> {hidden_dim}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Physics Encoder.

        Args:
            x (torch.Tensor): Raw physical features. Expected shape: (Batch, Features)

        Returns:
            torch.Tensor: Physics Embedding. Shape: (Batch, HiddenDimension)
        """
        if x.dim() != 2:
            raise ValueError(f"PhysicsEncoder expects 2D input (Batch, Features), got {x.dim()}D")
            
        return self.mlp(x)
