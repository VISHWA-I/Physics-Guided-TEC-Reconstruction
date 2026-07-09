import torch
import math

def chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Computes Chebyshev polynomials of the first kind up to a specific degree.
    This acts as a differentiable, learnable non-linear basis for our Pure PyTorch KAN.

    Args:
        x (torch.Tensor): Input tensor. Values should ideally be scaled between [-1, 1].
        degree (int): The maximum polynomial degree to compute.

    Returns:
        torch.Tensor: A tensor of shape (..., degree), where the last dimension 
                      contains the polynomial evaluations: [T_0(x), T_1(x), ..., T_{degree-1}(x)]
    """
    # x must be in [-1, 1] for Chebyshev polynomials to be stable
    # The KAN layer should handle this normalization before calling this.
    x = torch.clamp(x, -1.0 + 1e-5, 1.0 - 1e-5)
    
    if degree == 0:
        return torch.empty(list(x.shape) + [0], device=x.device, dtype=x.dtype)
        
    bases = [torch.ones_like(x)]
    if degree > 1:
        bases.append(x)
        
    for i in range(2, degree):
        bases.append(2.0 * x * bases[-1] - bases[-2])
        
    return torch.stack(bases, dim=-1)

def top_k_masking(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Applies Top-k masking to attention scores. 
    Keeps the top k values and masks the rest with -inf.

    Args:
        scores (torch.Tensor): Attention logits of shape (..., num_keys)
        k (int): Number of keys to keep.

    Returns:
        torch.Tensor: Masked attention logits.
    """
    if k >= scores.size(-1):
        return scores
        
    # Find the k-th largest value along the last dimension
    topk_vals, _ = torch.topk(scores, k, dim=-1)
    # The k-th largest value is the threshold
    threshold = topk_vals[..., -1:]
    
    # Mask out anything below the threshold
    mask = scores < threshold
    masked_scores = scores.masked_fill(mask, float('-inf'))
    
    return masked_scores
