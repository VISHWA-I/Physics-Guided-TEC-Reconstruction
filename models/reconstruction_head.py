import torch
import torch.nn as nn
from typing import Tuple

from utils.logger import get_model_logger

logger = get_model_logger("ReconstructionHead")

class ReconstructionHead(nn.Module):
    """
    Transforms the deeply processed hidden representation into the final physical Topside TEC value.
    Implements a 3-stage coarse-to-fine reconstruction.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initializes the Reconstruction Head.

        Args:
            hidden_dim (int): Dimensionality of the incoming latent representation.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        # Stage 1: Coarse Estimate
        self.stage1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Stage 2: Fine Correction (predicts a delta)
        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Stage 3: Residual Refinement
        self.stage3 = nn.Sequential(
            nn.Linear(hidden_dim + 1, 1)
        )
        
        logger.info("ReconstructionHead initialized (3 stages).")

    def forward(self, hidden_rep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TEC Reconstruction.

        Args:
            hidden_rep (torch.Tensor): Final latent representation. Shape (Batch, Seq, HiddenDim)

        Returns:
            torch.Tensor: Final Topside TEC prediction. Shape (Batch, Seq, 1)
        """
        # Stage 1
        coarse_tec = self.stage1(hidden_rep) # (Batch, Seq, 1)
        
        # Stage 2
        # Concatenate hidden representation with coarse estimate
        h_stage2 = torch.cat([hidden_rep, coarse_tec], dim=-1)
        fine_delta = self.stage2(h_stage2) # (Batch, Seq, 1)
        
        fine_tec = coarse_tec + fine_delta
        
        # Stage 3
        h_stage3 = torch.cat([hidden_rep, fine_tec], dim=-1)
        residual = self.stage3(h_stage3)
        
        final_tec = fine_tec + residual
        
        # Note: In Phase 2.5, we will apply the Physics Loss (e.g. ReLU to enforce TEC >= 0).
        # We do not strictly clamp it here to allow gradients to flow freely before the loss.
        # The hooks are implicitly the outputs of this function.
        
        return final_tec
