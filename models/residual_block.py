import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A generic reusable residual connection wrapper.
    Performs: Output = Normalization(Input + Dropout(Layer(Input)))
    Alternatively, pre-norm style: Output = Input + Dropout(Layer(Normalization(Input)))
    
    We implement the pre-norm style as it's standard for modern deep networks like Mamba.
    """
    
    def __init__(self, layer: nn.Module, d_model: int, norm: nn.Module, dropout_prob: float = 0.0):
        """
        Initializes the ResidualBlock.

        Args:
            layer (nn.Module): The core computation block (e.g., Mamba block, Linear).
            d_model (int): Feature dimension size.
            norm (nn.Module): The normalization layer instance.
            dropout_prob (float): Dropout probability for the residual branch.
        """
        super().__init__()
        self.layer = layer
        self.norm = norm
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass applying the pre-norm residual connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with residual added.
        """
        # Pre-Norm residual architecture
        residual = x
        x = self.norm(x)
        x = self.layer(x, *args, **kwargs)
        x = self.dropout(x)
        
        return residual + x
