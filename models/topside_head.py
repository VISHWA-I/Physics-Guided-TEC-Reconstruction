import torch
import torch.nn as nn
from models.reconstruction_head import ReconstructionHead
from utils.logger import get_model_logger

logger = get_model_logger("TopsideTECHead")

class TopsideTECHead(nn.Module):
    """
    The primary prediction head for Topside TEC.
    Wraps the multi-stage coarse-to-fine Reconstruction Head defined in Phase 2.4.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.reconstructor = ReconstructionHead(hidden_dim, dropout)
        logger.info("TopsideTECHead initialized.")

    def forward(self, hidden_rep: torch.Tensor) -> torch.Tensor:
        """
        Predicts Topside TEC from the TKAN hidden representation.

        Args:
            hidden_rep (torch.Tensor): Final latent representation. Shape (Batch, Seq, HiddenDim)

        Returns:
            torch.Tensor: Predicted Topside TEC. Shape (Batch, Seq, 1)
        """
        return self.reconstructor(hidden_rep)
