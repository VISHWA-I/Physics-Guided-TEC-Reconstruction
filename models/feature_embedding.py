import torch
import torch.nn as nn

from models.normalization import get_normalization_layer

class FeatureEmbedding(nn.Module):
    """
    Projects raw features to a dense high-dimensional embedding space.
    Input Shape: (Batch, Sequence, Features)
    Output Shape: (Batch, Sequence, EmbeddingDimension)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, norm_type: str = "layernorm"):
        """
        Initializes the FeatureEmbedding module.

        Args:
            in_features (int): Number of raw input features.
            out_features (int): Dimension of the output embedding.
            dropout (float): Dropout probability.
            norm_type (str): Type of normalization to apply ('layernorm' or 'rmsnorm').
        """
        super().__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be > 0.")
            
        self.projection = nn.Linear(in_features, out_features)
        self.norm = get_normalization_layer(norm_type, out_features)
        
        # SiLU is frequently used with Mamba architectures, but we'll use GELU as a standard alternative
        # or follow the user's implicit setup. We use SiLU here by default.
        self.activation = nn.SiLU() 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature projection.

        Args:
            x (torch.Tensor): Raw input (Batch, Sequence, Features)

        Returns:
            torch.Tensor: Dense embedding (Batch, Sequence, out_features)
        """
        x = self.projection(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
