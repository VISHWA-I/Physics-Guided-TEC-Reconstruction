import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Often preferred in modern architectures (like LLaMA and Mamba) over LayerNorm
    as it removes the mean-centering step, providing similar performance with lower
    computational cost.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initializes RMSNorm.

        Args:
            d_model (int): The dimensionality of the input/output features.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        # Calculate the root mean square along the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        
        # Apply the learnable scale parameter
        return x_norm * self.weight

def get_normalization_layer(norm_type: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    """
    Factory function to get the requested normalization layer.

    Args:
        norm_type (str): Type of normalization ('layernorm', 'rmsnorm').
        d_model (int): The feature dimension.
        eps (float): Epsilon for numerical stability.

    Returns:
        nn.Module: The instantiated normalization layer.
    """
    norm_type = norm_type.lower().strip()
    if norm_type == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    elif norm_type == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Supported: 'layernorm', 'rmsnorm'.")
