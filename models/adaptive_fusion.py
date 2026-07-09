import torch
import torch.nn as nn
from typing import Tuple, Dict

from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry
from models.physics_encoder import PhysicsEncoder
from models.geo_encoder import GeoEncoder
from models.storm_encoder import StormContextEncoder
from models.physics_gate import PhysicsGate
from models.cross_attention import PhysicsCrossAttention
from models.residual_block import ResidualBlock
from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("AdaptiveFusionLayer")

@ModelRegistry.register("AdaptiveFusionLayer")
class AdaptiveFusionLayer(BaseModel):
    """
    Fuses the Temporal representation with Physical, Geophysical, and Storm contexts.
    Uses Cross Attention for fine-grained feature interaction, and a Physics Gate
    for macro-level adaptive weighting.
    """

    def __init__(self, config: ModelConfig, debug_mode: bool = False):
        """
        Initializes the Fusion Layer.

        Args:
            config (ModelConfig): Global configuration.
            debug_mode (bool): Enables shape printing during forward pass.
        """
        super().__init__(config)
        self.debug_mode = debug_mode
        self.hidden_dim = config.hidden_dimension
        
        # 1. Encoders for the continuous context vectors
        self.physics_enc = PhysicsEncoder(
            in_features=len(config.physics_features),
            hidden_dim=self.hidden_dim,
            dropout=config.dropout,
            norm_type=config.normalization
        )
        
        self.geo_enc = GeoEncoder(
            hidden_dim=self.hidden_dim,
            dropout=config.dropout,
            norm_type=config.normalization
        )
        
        self.storm_enc = StormContextEncoder(
            in_features=len(config.storm_features),
            hidden_dim=self.hidden_dim,
            dropout=config.dropout,
            norm_type=config.normalization
        )
        
        # 2. Physics Gate
        self.physics_gate = PhysicsGate(hidden_dim=self.hidden_dim)
        
        # 3. Cross Attention
        cross_attn = PhysicsCrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout
        )
        
        # We wrap Cross Attention in a ResidualBlock. Note that ResidualBlock usually expects a layer
        # with signature `layer(x)`. CrossAttention has `layer(query, key_value)`.
        # To reuse ResidualBlock, we use a lambda/wrapper or just implement the residual natively here.
        # It's cleaner to implement the residual natively here because of the dual inputs.
        self.cross_attn = cross_attn
        self.attn_norm = get_normalization_layer(config.normalization, self.hidden_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        
        # 4. Feed Forward Network (Post-Fusion)
        ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        self.ffn_block = ResidualBlock(
            layer=ffn,
            d_model=self.hidden_dim,
            norm=get_normalization_layer(config.normalization, self.hidden_dim),
            dropout_prob=config.dropout
        )
        
        logger.info("AdaptiveFusionLayer fully initialized.")

    def forward(self, 
                temporal_seq: torch.Tensor, 
                physics_feats: torch.Tensor, 
                geo_feats: torch.Tensor, 
                storm_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuses all branches together.

        Args:
            temporal_seq: (Batch, Sequence, HiddenDim)
            physics_feats: (Batch, NumPhysics)
            geo_feats: (Batch, NumGeo)
            storm_feats: (Batch, NumStorm)

        Returns:
            Tuple containing:
                - Unified Representation: (Batch, Sequence, HiddenDim)
                - Dictionary of visualization metrics (Attention Maps, Gate Weights)
        """
        if self.debug_mode:
            print(f"[DEBUG] Temporal In: {temporal_seq.shape}")
            print(f"[DEBUG] Physics In: {physics_feats.shape}")
            print(f"[DEBUG] Geo In: {geo_feats.shape}")
            print(f"[DEBUG] Storm In: {storm_feats.shape}")
            
        batch_size = temporal_seq.size(0)

        # 1. Encode Context Features
        # (Batch, HiddenDim)
        phys_emb = self.physics_enc(physics_feats)
        geo_emb = self.geo_enc(geo_feats)
        storm_emb = self.storm_enc(storm_feats)
        
        # Expand them into a pseudo-sequence of context tokens
        # Context Token Sequence: [Physics Token, Geo Token, Storm Token]
        # Shape: (Batch, 3, HiddenDim)
        context_seq = torch.stack([phys_emb, geo_emb, storm_emb], dim=1)
        
        if self.debug_mode:
            print(f"[DEBUG] Context Seq: {context_seq.shape}")

        # 2. Physics Gate
        # Compute the weight for Physics vs Temporal based on Storm context
        # physics_weight is (Batch, 1, 1)
        physics_weight = self.physics_gate(storm_emb)
        temporal_weight = 1.0 - physics_weight
        
        if self.debug_mode:
            print(f"[DEBUG] Physics Gate Weight: {physics_weight.shape}")

        # 3. Cross Attention (Pre-Norm style residual)
        norm_temporal = self.attn_norm(temporal_seq)
        
        # Temporal attends to [Physics, Geo, Storm] tokens
        attn_out, attn_weights = self.cross_attn(query=norm_temporal, key_value=context_seq)
        attn_out = self.attn_dropout(attn_out)
        
        # The cross-attention output represents the "contextualized physics" features
        # We dynamically blend it with the temporal representation using the Gate.
        fused_seq = (temporal_weight * temporal_seq) + (physics_weight * attn_out)
        
        # 4. Feed Forward Network processing
        fused_seq = self.ffn_block(fused_seq)
        
        if self.debug_mode:
            print(f"[DEBUG] Fused Output: {fused_seq.shape}")
            
        metrics = {
            "attention_maps": attn_weights,
            "physics_gate_weights": physics_weight.squeeze(-1).squeeze(-1) # (Batch,)
        }

        return fused_seq, metrics
