import torch
import torch.nn as nn
from typing import Tuple, Optional

from utils.logger import get_model_logger

logger = get_model_logger("PhysicsCrossAttention")

class PhysicsCrossAttention(nn.Module):
    """
    Physics-Aware Multi-Head Cross Attention.
    Allows the Temporal Sequence (Queries) to attend to the 
    Static Physical and Geophysical Context tokens (Keys, Values).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initializes Cross Attention.

        Args:
            hidden_dim (int): The feature dimension of the inputs.
            num_heads (int): Number of attention heads.
            dropout (float): Attention dropout probability.
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"PhysicsCrossAttention initialized with {num_heads} heads.")

    def forward(self, 
                query: torch.Tensor, 
                key_value: torch.Tensor, 
                storm_modulation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Cross Attention.

        Args:
            query (torch.Tensor): Temporal sequence. Shape: (Batch, SeqLen_Q, Hidden)
            key_value (torch.Tensor): Context tokens. Shape: (Batch, SeqLen_KV, Hidden)
            storm_modulation (Optional[torch.Tensor]): Optional modifier for attention scores.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Attention output: (Batch, SeqLen_Q, Hidden)
                - Attention weights: (Batch, Num_Heads, SeqLen_Q, SeqLen_KV)
        """
        batch_size, seq_q, _ = query.shape
        _, seq_kv, _ = key_value.shape
        
        # 1. Linear projections
        Q = self.q_proj(query)       # (B, Sq, H)
        K = self.k_proj(key_value)   # (B, Skv, H)
        V = self.v_proj(key_value)   # (B, Skv, H)
        
        # 2. Reshape for multi-head attention: (B, Seq, Heads, HeadDim) -> (B, Heads, Seq, HeadDim)
        Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Compute Attention Scores: Q * K^T / sqrt(d_k)
        # Q: (B, Heads, Sq, Hd)
        # K.transpose: (B, Heads, Hd, Skv)
        # scores: (B, Heads, Sq, Skv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Optional: Storm Modulation
        # The storm context can dynamically amplify or dampen attention to specific context tokens
        if storm_modulation is not None:
            # Assume storm_modulation is broadly broadcasting, or learned per-head
            # For simplicity, we just add it to the logits.
            scores = scores + storm_modulation
            
        # 4. Attention Weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)
        
        # 5. Apply weights to Values
        # (B, Heads, Sq, Skv) @ (B, Heads, Skv, Hd) -> (B, Heads, Sq, Hd)
        out = torch.matmul(attn_weights_drop, V)
        
        # 6. Recombine heads: (B, Sq, Heads, Hd) -> (B, Sq, Hidden)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_q, self.hidden_dim)
        
        # 7. Final output projection
        out = self.out_proj(out)
        
        return out, attn_weights
