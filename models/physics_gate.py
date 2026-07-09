import torch
import torch.nn as nn
from utils.logger import get_model_logger

logger = get_model_logger("PhysicsGate")

class PhysicsGate(nn.Module):
    """
    Dynamically routes information flow between Temporal and Physical branches
    based on the current geomagnetic storm context.
    
    During quiet times, Temporal (historical) data is highly reliable.
    During storm times (high Kp, low Dst), Temporal data becomes chaotic,
    so the model should rely more heavily on instantaneous physical GIRO measurements.
    """

    def __init__(self, hidden_dim: int):
        """
        Initializes the Physics Gate.

        Args:
            hidden_dim (int): The dimensionality of the Storm Context Embedding.
        """
        super().__init__()
        
        # A simple lightweight network to compute the gating scalar (0 to 1)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Bounds weights between 0 and 1
        )

    def forward(self, storm_embedding: torch.Tensor) -> torch.Tensor:
        """
        Computes the adaptive physics weight.

        Args:
            storm_embedding (torch.Tensor): The encoded storm context. Shape: (Batch, HiddenDimension)

        Returns:
            torch.Tensor: The scalar gate value per batch item. Shape: (Batch, 1, 1)
                          Formatted for broadcasting over sequence lengths.
        """
        if storm_embedding.dim() != 2:
            raise ValueError(f"PhysicsGate expects 2D input (Batch, HiddenDimension), got {storm_embedding.dim()}D")
            
        gate_val = self.gate_network(storm_embedding)
        
        # Expand dimensions so it can easily multiply with (Batch, Sequence, HiddenDimension) tensors
        return gate_val.unsqueeze(-1)
