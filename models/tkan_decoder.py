import torch
import torch.nn as nn
from typing import Tuple, Dict

from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry
from models.memory_manager import MemoryManager
from models.memory_retrieval import MemoryRetrieval
from models.multiscale_tkan import MultiScaleTKAN
from models.reconstruction_head import ReconstructionHead
from utils.logger import get_model_logger

logger = get_model_logger("TKANDecoder")

@ModelRegistry.register("TKANDecoder")
class TKANDecoder(BaseModel):
    """
    Physics-Guided Memory-Augmented TKAN Decoder.
    Takes the fused latent representation (from Phase 2.3), queries the physics memory bank,
    processes it through parallel KAN branches, and reconstructs the Topside TEC.
    """

    def __init__(self, config: ModelConfig, debug_mode: bool = False):
        """
        Initializes the TKAN Decoder.

        Args:
            config (ModelConfig): Global configuration.
            debug_mode (bool): Enables shape printing during forward pass.
        """
        super().__init__(config)
        self.debug_mode = debug_mode
        hidden_dim = config.hidden_dimension
        
        # 1. Physics Memory Bank & Manager
        self.memory_manager = MemoryManager(
            num_slots=config.memory_slots,
            embedding_dim=hidden_dim
        )
        
        # 2. Memory Retrieval (Attention)
        self.memory_retrieval = MemoryRetrieval(
            hidden_dim=hidden_dim,
            num_heads=config.memory_heads,
            top_k=config.top_k_retrieval,
            dropout=config.dropout,
            norm_type=config.normalization
        )
        
        # 3. Multi-Scale TKAN
        self.multiscale_tkan = MultiScaleTKAN(
            hidden_dim=hidden_dim,
            num_branches=config.tkan_branches,
            depth=config.kan_depth,
            dropout=config.dropout,
            norm_type=config.normalization
        )
        
        # 4. Reconstruction Head
        self.reconstruction_head = ReconstructionHead(
            hidden_dim=hidden_dim,
            dropout=config.dropout
        )
        
        logger.info("TKANDecoder fully initialized.")

    def forward(self, fused_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the Decoder.

        Args:
            fused_latent (torch.Tensor): Unified representation from Phase 2.3. Shape: (Batch, Seq, Hidden)

        Returns:
            Tuple containing:
                - Final Topside TEC prediction: (Batch, Seq, 1)
                - Final Hidden Representation (for Phase 2.5): (Batch, Seq, Hidden)
                - Dictionary of metrics (e.g., Memory Attention Scores).
        """
        if self.debug_mode:
            print(f"[DEBUG] Latent Input Shape: {fused_latent.shape}")
            
        # 1. Get entire memory pool (Slots, Hidden)
        memory_pool = self.memory_manager.get_memory_pool()
        if self.debug_mode:
            print(f"[DEBUG] Full Memory Pool Shape: {memory_pool.shape}")
            
        # 2. Memory Retrieval
        memory_enhanced_seq, memory_attn_scores = self.memory_retrieval(fused_latent, memory_pool)
        if self.debug_mode:
            print(f"[DEBUG] Memory-Enhanced Shape: {memory_enhanced_seq.shape}")
            print(f"[DEBUG] Memory Attn Weights Shape: {memory_attn_scores.shape}")
            
        # 3. Multi-Scale TKAN processing
        tkan_out = self.multiscale_tkan(memory_enhanced_seq)
        if self.debug_mode:
            print(f"[DEBUG] TKAN Output Shape: {tkan_out.shape}")
            
        # 4. Coarse-to-Fine Reconstruction
        tec_prediction = self.reconstruction_head(tkan_out)
        if self.debug_mode:
            print(f"[DEBUG] Reconstructed TEC Shape: {tec_prediction.shape}")
            
        metrics = {
            "memory_attention_scores": memory_attn_scores
        }
        
        # We return the `tkan_out` (final hidden representation) so that Phase 2.5 
        # can enforce physical constraints (like Net TEC equality) easily.
        return tec_prediction, tkan_out, metrics
