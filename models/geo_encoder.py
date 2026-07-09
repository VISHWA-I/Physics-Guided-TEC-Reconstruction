import torch
import torch.nn as nn

from models.fusion_utils import generate_sin_cos_encoding
from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("GeoEncoder")

class GeoEncoder(nn.Module):
    """
    Encodes Geophysical features. Handles continuous cyclic variables
    (Local Time, Day of Year, Longitude) via Sine/Cosine projections.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, norm_type: str = "rmsnorm"):
        """
        Initializes the GeoEncoder.

        Args:
            hidden_dim (int): Target hidden dimension.
            dropout (float): Dropout probability.
            norm_type (str): Type of normalization to use.
        """
        super().__init__()
        
        # We have 4 features: Latitude, Longitude, DayOfYear, LocalTime
        # Latitude is non-cyclic (-90 to 90). We'll treat it directly.
        # Longitude (360), DOY (365.25), LT (24) are cyclic.
        # So we have 1 direct feature + 3 cyclic * 2 (sin/cos) = 7 effective dimensions.
        self.effective_dim = 1 + (3 * 2)
        
        self.projection = nn.Sequential(
            nn.Linear(self.effective_dim, hidden_dim),
            get_normalization_layer(norm_type, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        logger.info(f"GeoEncoder initialized targeting dim {hidden_dim}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Geophysical features. Shape (Batch, 4).
                              Assuming order: [Latitude, Longitude, DayOfYear, LocalTime]

        Returns:
            torch.Tensor: Geo Embedding. Shape (Batch, HiddenDimension)
        """
        if x.dim() != 2 or x.size(1) != 4:
            raise ValueError(f"GeoEncoder expects shape (Batch, 4), got {x.shape}")
            
        lat = x[:, 0:1]
        lon = x[:, 1]
        doy = x[:, 2]
        lt = x[:, 3]
        
        # Cyclic encodings
        lon_enc = generate_sin_cos_encoding(lon, period=360.0)
        doy_enc = generate_sin_cos_encoding(doy, period=365.25)
        lt_enc = generate_sin_cos_encoding(lt, period=24.0)
        
        # Concat all features
        # Flatten the sin/cos encodings from (Batch, 2) to match lat (Batch, 1)
        encoded_features = torch.cat([lat, lon_enc, doy_enc, lt_enc], dim=-1)
        
        # Project to hidden dimension
        return self.projection(encoded_features)
