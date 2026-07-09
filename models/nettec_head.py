import torch
import torch.nn as nn

from utils.logger import get_model_logger

logger = get_model_logger("NetTECHead")

class NetTECHead(nn.Module):
    """
    Predicts the final Net TEC.
    Takes the physically summed Base Net TEC and applies an optional residual correction 
    to account for unmodeled high-altitude plasmaspheric contributions using the hidden state.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Corrects the base assumption using the latent representation
        self.plasmaspheric_correction = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        logger.info("NetTECHead initialized.")

    def forward(self, base_net_tec: torch.Tensor, hidden_rep: torch.Tensor) -> torch.Tensor:
        """
        Computes final Net TEC.

        Args:
            base_net_tec (torch.Tensor): Sum of bottomside and topside. Shape (Batch, Seq, 1)
            hidden_rep (torch.Tensor): Latent representation. Shape (Batch, Seq, Hidden)

        Returns:
            torch.Tensor: Final Net TEC. Shape (Batch, Seq, 1)
        """
        # We concatenate the base physical estimate with the deep latent state
        h = torch.cat([hidden_rep, base_net_tec], dim=-1)
        
        # The network predicts a small delta correction (e.g., plasmaspheric tails)
        correction = self.plasmaspheric_correction(h)
        
        return base_net_tec + correction
