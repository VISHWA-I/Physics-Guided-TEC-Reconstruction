import torch
import torch.nn as nn

from utils.logger import get_model_logger

logger = get_model_logger("Embeddings")

class BranchEmbedding(nn.Module):
    """
    Generic embedding layer for a single input branch.
    Normalizes the input, projects it to a higher dimension,
    applies LayerNorm, and applies dropout.
    
    Expected Input Shape: (Batch, Window, Features)
    Output Shape: (Batch, Window, Embedding_Dimension)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        """
        Initializes the BranchEmbedding module.

        Args:
            in_features (int): Number of features in the input sequence.
            out_features (int): The dimension to project the features to.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Input and output features must be strictly positive.")
            
        # 1. Input Normalization (LayerNorm over the feature dimension)
        self.input_norm = nn.LayerNorm(in_features)
        
        # 2. Linear Projection
        self.projection = nn.Linear(in_features, out_features)
        
        # 3. Output LayerNorm and Dropout
        self.output_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Window, Features)

        Returns:
            torch.Tensor: Embedded tensor of shape (Batch, Window, out_features)
        """
        # Validate dimensionality
        if x.dim() != 3:
            logger.error(f"Expected 3D input (Batch, Window, Features), got {x.dim()}D.")
            raise ValueError(f"Input must be 3D, got {x.dim()}D tensor.")
            
        # Normalize raw input
        x = self.input_norm(x)
        
        # Project to embedding space
        x = self.projection(x)
        
        # Apply normalization and dropout in the embedding space
        x = self.output_norm(x)
        x = self.dropout(x)
        
        return x

