import torch
import torch.nn as nn
from typing import List

from models.decoder_utils import chebyshev_basis
from models.residual_block import ResidualBlock
from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("MultiScaleTKAN")

class KANBlock(nn.Module):
    """
    Pure PyTorch implementation of a Kolmogorov-Arnold Network (KAN) block.
    Uses learnable Chebyshev polynomial bases instead of standard Linear MLPs
    for non-linear edge transformations.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 3, dropout: float = 0.1):
        """
        Initializes the KAN Block.

        Args:
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            degree (int): The polynomial degree for the basis functions.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # In a KAN, instead of a simple weight matrix (out_features, in_features),
        # we have a weight matrix for *each* polynomial degree.
        # Shape: (out_features, in_features, degree)
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features, degree))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialization
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        
        self.dropout = nn.Dropout(dropout)
        # Activation is inherently handled by the polynomial basis, but we can add SiLU
        # for extra stabilization on the final sum.
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the KAN layer.

        Args:
            x (torch.Tensor): Input tensor. Shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor. Shape (..., out_features)
        """
        # x is assumed to be normalized (handled by LayerNorm before this block)
        # Generate basis: (..., in_features, degree)
        basis = chebyshev_basis(x, self.degree)
        
        # We need to compute: out_j = sum_i sum_d (weight_{j,i,d} * basis_{...,i,d})
        # This can be elegantly done with torch.einsum.
        # '...id, jid -> ...j'
        out = torch.einsum('...id, jid -> ...j', basis, self.weights) + self.bias
        
        out = self.activation(out)
        return self.dropout(out)


class TKANBranch(nn.Module):
    """
    A single branch of the Temporal KAN (TKAN).
    Consists of stacked KANBlocks with residual connections.
    """

    def __init__(self, hidden_dim: int, depth: int, degree: int, dropout: float, norm_type: str):
        """
        Initializes a TKAN Branch.
        """
        super().__init__()
        
        layers = []
        for _ in range(depth):
            kan_layer = KANBlock(in_features=hidden_dim, out_features=hidden_dim, degree=degree, dropout=dropout)
            block = ResidualBlock(
                layer=kan_layer,
                d_model=hidden_dim,
                norm=get_normalization_layer(norm_type, hidden_dim),
                dropout_prob=0.0 # Handled inside KANBlock
            )
            layers.append(block)
            
        self.pipeline = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x)


class MultiScaleTKAN(nn.Module):
    """
    Multi-Scale Temporal KAN.
    Processes the memory-enhanced latent sequence through parallel TKAN branches
    operating at different polynomial complexities (representing different scales of dynamics).
    """

    def __init__(self, hidden_dim: int, num_branches: int, depth: int, dropout: float, norm_type: str):
        super().__init__()
        
        self.branches = nn.ModuleList()
        # Create branches with increasing polynomial degrees (e.g., 2, 3, 4)
        for b in range(num_branches):
            degree = 2 + b
            branch = TKANBranch(
                hidden_dim=hidden_dim,
                depth=depth,
                degree=degree,
                dropout=dropout,
                norm_type=norm_type
            )
            self.branches.append(branch)
            
        # Fusion projection to combine branches
        self.fusion_proj = nn.Linear(hidden_dim * num_branches, hidden_dim)
        self.norm = get_normalization_layer(norm_type, hidden_dim)
        
        logger.info(f"MultiScaleTKAN initialized with {num_branches} branches (depth={depth}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Multi-Scale TKAN.

        Args:
            x (torch.Tensor): Shape (Batch, Seq, Hidden)

        Returns:
            torch.Tensor: Shape (Batch, Seq, Hidden)
        """
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
            
        # Concatenate along the hidden dimension
        fused = torch.cat(branch_outputs, dim=-1)
        
        # Project back to original hidden dim
        out = self.fusion_proj(fused)
        out = self.norm(out)
        
        # Final residual connection against the original input
        return x + out
