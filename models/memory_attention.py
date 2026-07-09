import torch
import torch.nn as nn
from typing import Tuple

from models.decoder_utils import top_k_masking
from utils.logger import get_model_logger

logger = get_model_logger("MemoryAttention")

class MemoryAttention(nn.Module):
    """
    Multi-Head Attention mechanism designed specifically to retrieve information
    from the static Physics Memory Bank using the dynamic Latent Sequence as the query.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, top_k: int = 8, dropout: float = 0.1):
        """
        Initializes Memory Attention.

        Args:
            hidden_dim (int): Feature dimension of queries/keys/values.
            num_heads (int): Number of attention heads.
            top_k (int): Number of memory slots to retrieve (hard sparsity).
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k = top_k
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # We project the memory bank (Keys and Values)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, memory_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for memory retrieval.

        Args:
            query (torch.Tensor): The dynamic sequence. Shape: (Batch, Sequence, HiddenDim)
            memory_pool (torch.Tensor): The static memory bank. Shape: (Total_Slots, HiddenDim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Retrieved Context: (Batch, Sequence, HiddenDim)
                - Attention Scores: (Batch, Heads, Sequence, Total_Slots)
        """
        batch_size, seq_len, _ = query.shape
        num_slots = memory_pool.size(0)
        
        # 1. Linear Projections
        Q = self.q_proj(query) # (B, Seq, H)
        
        # Memory pool is (Slots, H), we project it directly.
        K = self.k_proj(memory_pool) # (Slots, H)
        V = self.v_proj(memory_pool) # (Slots, H)
        
        # 2. Reshape for Multi-Head Attention
        # Q: (B, Seq, Heads, HeadDim) -> (B, Heads, Seq, HeadDim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K/V: (Slots, Heads, HeadDim) -> (Heads, HeadDim, Slots)
        # Note: K/V do not have a batch dimension because memory is shared across the batch
        K = K.view(num_slots, self.num_heads, self.head_dim).permute(1, 2, 0)
        V = V.view(num_slots, self.num_heads, self.head_dim).transpose(0, 1) # (Heads, Slots, HeadDim)
        
        # 3. Compute Attention Scores (Batch-wise broadcasting against shared memory)
        # Q: (B, Heads, Seq, HeadDim)
        # K: (Heads, HeadDim, Slots)
        # Result: (B, Heads, Seq, Slots)
        scores = torch.matmul(Q, K) / (self.head_dim ** 0.5)
        
        # 4. Top-K Masking (Sparsity constraint)
        scores = top_k_masking(scores, self.top_k)
        
        # 5. Attention Weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)
        
        # 6. Apply Weights to Values
        # attn: (B, Heads, Seq, Slots)
        # V: (Heads, Slots, HeadDim) -> broadcast to (B, Heads, Slots, HeadDim)
        # Result: (B, Heads, Seq, HeadDim)
        V_expanded = V.unsqueeze(0).expand(batch_size, -1, -1, -1)
        out = torch.matmul(attn_weights_drop, V_expanded)
        
        # 7. Recombine Heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 8. Output Projection
        out = self.out_proj(out)
        
        return out, attn_weights
