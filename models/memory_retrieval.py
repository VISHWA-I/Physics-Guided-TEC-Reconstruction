import torch
import torch.nn as nn
from typing import Tuple

from models.memory_attention import MemoryAttention
from models.residual_block import ResidualBlock
from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("MemoryRetrieval")

class MemoryRetrieval(nn.Module):
    """
    High-level module that handles the retrieval and integration of information
    from the MemoryManager into the active latent sequence.
    """

    def __init__(self, hidden_dim: int, num_heads: int, top_k: int, dropout: float, norm_type: str):
        """
        Initializes Memory Retrieval.

        Args:
            hidden_dim (int): Feature dimension.
            num_heads (int): Number of attention heads for memory queries.
            top_k (int): Sparsity constraint for retrieval.
            dropout (float): Dropout probability.
            norm_type (str): Normalization type.
        """
        super().__init__()
        
        self.memory_attn = MemoryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            top_k=top_k,
            dropout=dropout
        )
        
        self.norm = get_normalization_layer(norm_type, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"MemoryRetrieval initialized (Heads: {num_heads}, Top-K: {top_k}).")

    def forward(self, latent_seq: torch.Tensor, memory_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves context from the memory bank and adds it to the latent sequence.

        Args:
            latent_seq (torch.Tensor): Current dynamic state. Shape (Batch, Seq, Hidden)
            memory_pool (torch.Tensor): Static global memory. Shape (Slots, Hidden)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Memory-enhanced sequence: (Batch, Seq, Hidden)
                - Attention Scores: (Batch, Heads, Seq, Slots)
        """
        # Pre-Norm
        normed_seq = self.norm(latent_seq)
        
        # Query memory
        memory_context, attn_scores = self.memory_attn(query=normed_seq, memory_pool=memory_pool)
        
        # Residual add
        enhanced_seq = latent_seq + self.dropout(memory_context)
        
        return enhanced_seq, attn_scores
