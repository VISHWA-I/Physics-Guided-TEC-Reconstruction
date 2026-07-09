import torch
import torch.nn as nn
import torch.nn.functional as F

from models.residual_block import ResidualBlock
from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("MambaBlock")

class MambaReferenceLayer(nn.Module):
    """
    A pure PyTorch reference implementation of the Mamba (Selective State Space) Layer.
    This provides equivalent mathematical modeling to the CUDA Mamba kernel but runs
    natively on CPU, CUDA, and Apple MPS without requiring custom C++ extensions.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        """
        Initializes the selective state space layer.

        Args:
            d_model (int): Hidden dimension size.
            d_state (int): State expansion factor (N).
            d_conv (int): Convolution kernel size.
            expand (int): Expansion factor for the inner dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.d_inner = int(self.expand * self.d_model)
        
        # Input projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution for short-term temporal features
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # State space parameters (Delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Base state matrix A (learnable, diagonal)
        # Initialize A to log(range) to enforce stable decaying states
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Base parameter D (skip connection equivalent)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the selective state space model.

        Args:
            x (torch.Tensor): Shape (Batch, Sequence, d_model)

        Returns:
            torch.Tensor: Shape (Batch, Sequence, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # 1. Project input to expanded dimension and split into x and res (for gating)
        xz = self.in_proj(x)
        x_inner, res = xz.chunk(2, dim=-1)
        
        # 2. 1D Convolution (apply over sequence length)
        # Conv1d expects (Batch, Channels, Length)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Truncate padding
        x_conv = x_conv.transpose(1, 2)
        
        # Activation (Silu)
        x_conv = F.silu(x_conv)
        
        # 3. State Space parameter generation (Data dependent)
        # x_proj outputs Delta, B, C
        x_dbl = self.x_proj(x_conv) # (Batch, Sequence, d_inner + 2 * d_state)
        delta, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Delta passes through softplus
        delta = F.softplus(self.dt_proj(delta)) # (Batch, Sequence, d_inner)
        
        # Recover A from A_log
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # 4. Selective Scan (Iterative for reference implementation)
        # Discretize continuous parameters
        # For sequence modeling without custom kernel, we run a sequential loop.
        # This is mathematically identical but slower than hardware-aware scan.
        y = torch.empty_like(x_conv)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        # B and C are (Batch, Sequence, d_state). We need them as (Batch, Sequence, 1, d_state) for broadcasting
        B = B.unsqueeze(2) 
        C = C.unsqueeze(2)
        
        for t in range(seq_len):
            delta_t = delta[:, t].unsqueeze(-1) # (Batch, d_inner, 1)
            
            # Discretize A and B
            deltaA = torch.exp(delta_t * A) # (Batch, d_inner, d_state)
            deltaB = delta_t * B[:, t]      # (Batch, d_inner, d_state)
            
            # Update state
            x_t = x_conv[:, t].unsqueeze(-1) # (Batch, d_inner, 1)
            h = deltaA * h + deltaB * x_t    # (Batch, d_inner, d_state)
            
            # Compute output
            y_t = (h * C[:, t]).sum(dim=-1)  # (Batch, d_inner)
            y[:, t] = y_t
            
        # Add D skip connection
        y = y + x_conv * self.D
        
        # 5. Gating and output projection
        y = y * F.silu(res)
        out = self.out_proj(y)
        
        return out


class MambaBlock(nn.Module):
    """
    A complete Mamba-2 style block containing the Selective State Space Layer,
    wrapped in residual connections with normalization. Includes a FeedForward
    network for feature transformation.
    """
    
    def __init__(self, 
                 d_model: int, 
                 d_state: int = 16, 
                 d_conv: int = 4, 
                 expand: int = 2, 
                 dropout: float = 0.1, 
                 norm_type: str = "rmsnorm",
                 use_checkpointing: bool = False):
        """
        Initializes the complete MambaBlock.

        Args:
            d_model (int): Hidden dimension size.
            d_state (int): State expansion factor.
            d_conv (int): Convolution kernel size.
            expand (int): Expansion factor for inner features.
            dropout (float): Dropout probability.
            norm_type (str): Normalization type.
            use_checkpointing (bool): Whether to use gradient checkpointing.
        """
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # The core SSM module
        ssm_layer = MambaReferenceLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Wrap SSM in pre-norm residual block
        self.ssm_block = ResidualBlock(
            layer=ssm_layer,
            d_model=d_model,
            norm=get_normalization_layer(norm_type, d_model),
            dropout_prob=dropout
        )
        
        # Feed forward network
        ffn_layer = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model)
        )
        
        # Wrap FFN in pre-norm residual block
        self.ffn_block = ResidualBlock(
            layer=ffn_layer,
            d_model=d_model,
            norm=get_normalization_layer(norm_type, d_model),
            dropout_prob=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MambaBlock.

        Args:
            x (torch.Tensor): Input tensor (Batch, Sequence, d_model)

        Returns:
            torch.Tensor: Output tensor (Batch, Sequence, d_model)
        """
        if self.use_checkpointing and self.training:
            # Gradient checkpointing saves memory by trading compute for memory
            x = torch.utils.checkpoint.checkpoint(self.ssm_block, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.ffn_block, x, use_reentrant=False)
        else:
            x = self.ssm_block(x)
            x = self.ffn_block(x)
            
        return x

