import torch
import math

def generate_sin_cos_encoding(values: torch.Tensor, period: float) -> torch.Tensor:
    """
    Generates sine and cosine positional encodings for continuous cyclic features
    (e.g., Day of Year, Local Time).

    Args:
        values (torch.Tensor): The continuous feature values. Shape (Batch, ...)
        period (float): The maximum value of the cycle (e.g., 365.25 for DOY, 24 for LT).

    Returns:
        torch.Tensor: Encoded tensor where the last dimension is expanded by 2 (sin and cos).
                      Shape (Batch, ..., 2)
    """
    # Normalize to [0, 2 * pi]
    radians = values * (2 * math.pi / period)
    sin_enc = torch.sin(radians)
    cos_enc = torch.cos(radians)
    
    # Stack along the last dimension
    encoded = torch.stack([sin_enc, cos_enc], dim=-1)
    
    # Flatten the last two dimensions if necessary, but returning (..., 2) is cleaner
    return encoded

def create_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal boolean mask for attention mechanisms.
    (If needed for causal temporal masking, though this phase uses cross-attention).

    Args:
        seq_len (int): The sequence length.
        device (torch.device): The device to place the mask on.

    Returns:
        torch.Tensor: A boolean tensor of shape (seq_len, seq_len) where True indicates a masked position.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask
