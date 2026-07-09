import torch
import torch.nn as nn

from utils.logger import get_model_logger

logger = get_model_logger("ElectronDensityHead")

class ElectronDensityHead(nn.Module):
    """
    Estimates the peak electron density (NmF2).
    Follows the physical dependency hierarchy: it relies on the newly predicted Net TEC
    and the deep latent representation.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Input size: Hidden_dim (256) + Net TEC (1)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        logger.info("ElectronDensityHead initialized.")

    def forward(self, final_net_tec: torch.Tensor, hidden_rep: torch.Tensor) -> torch.Tensor:
        """
        Estimates NmF2.

        Args:
            final_net_tec (torch.Tensor): Output from NetTECHead. Shape (Batch, Seq, 1)
            hidden_rep (torch.Tensor): Deep latent state. Shape (Batch, Seq, Hidden)

        Returns:
            torch.Tensor: Electron Density. Shape (Batch, Seq, 1)
        """
        h = torch.cat([hidden_rep, final_net_tec], dim=-1)
        return self.network(h)
