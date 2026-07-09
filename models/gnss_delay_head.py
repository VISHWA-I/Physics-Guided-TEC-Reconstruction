import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.logger import get_model_logger

logger = get_model_logger("GNSSDelayHead")

class GNSSDelayHead(nn.Module):
    """
    Predicts multi-constellation GNSS delays.
    Relies on Net TEC, Electron Density, and explicit Geometry Features (Elevation, Azimuth).
    """

    def __init__(self, hidden_dim: int, geometry_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Base representation shared among all delay heads
        # Inputs: Hidden (256) + Net TEC (1) + Electron Density (1) + Geometry (2)
        in_features = hidden_dim + 1 + 1 + geometry_dim
        
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU()
        )
        
        shared_out = hidden_dim // 2
        
        # Specific Heads
        self.vertical_head = nn.Linear(shared_out, 1)
        self.slant_head = nn.Linear(shared_out, 1)
        self.time_head = nn.Linear(shared_out, 1)
        
        self.gps_head = nn.Linear(shared_out, 1)
        self.navic_head = nn.Linear(shared_out, 1)
        self.galileo_head = nn.Linear(shared_out, 1)
        self.beidou_head = nn.Linear(shared_out, 1)
        
        logger.info("GNSSDelayHead initialized.")

    def forward(self, 
                final_net_tec: torch.Tensor, 
                electron_density: torch.Tensor, 
                hidden_rep: torch.Tensor,
                geometry_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predicts delays.

        Args:
            final_net_tec (torch.Tensor): (Batch, Seq, 1)
            electron_density (torch.Tensor): (Batch, Seq, 1)
            hidden_rep (torch.Tensor): (Batch, Seq, Hidden)
            geometry_feats (Optional[torch.Tensor]): (Batch, Seq, GeometryDim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of delay predictions.
        """
        batch, seq, _ = hidden_rep.shape
        
        # Handle optional geometry features gracefully (if not provided, default to zeros)
        if geometry_feats is None:
            # We assume geometry_dim is 2 based on config defaults
            geometry_feats = torch.zeros(batch, seq, 2, device=hidden_rep.device)
            
        # Concatenate physical dependencies
        h = torch.cat([hidden_rep, final_net_tec, electron_density, geometry_feats], dim=-1)
        
        trunk_out = self.shared_trunk(h)
        
        return {
            "vertical_delay": self.vertical_head(trunk_out),
            "slant_delay": self.slant_head(trunk_out),
            "time_delay": self.time_head(trunk_out),
            "gps_delay": self.gps_head(trunk_out),
            "navic_delay": self.navic_head(trunk_out),
            "galileo_delay": self.galileo_head(trunk_out),
            "beidou_delay": self.beidou_head(trunk_out),
        }
