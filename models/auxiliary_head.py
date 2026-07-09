import torch
import torch.nn as nn
from typing import Dict

from utils.logger import get_model_logger

logger = get_model_logger("AuxiliaryReconstructionHead")

class AuxiliaryReconstructionHead(nn.Module):
    """
    Self-Supervised Head active only during training.
    Reconstructs core physics variables to force the latent representation to retain physical meaning.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4) # foF2, hmF2, B0, scaleF2
        )
        logger.info("AuxiliaryReconstructionHead initialized.")

    def forward(self, hidden_rep: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_rep (torch.Tensor): Deep latent state. Shape (Batch, Seq, Hidden)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of reconstructed variables.
        """
        out = self.network(hidden_rep) # (Batch, Seq, 4)
        
        return {
            "foF2": out[..., 0:1],
            "hmF2": out[..., 1:2],
            "B0": out[..., 2:3],
            "scaleF2": out[..., 3:4]
        }
