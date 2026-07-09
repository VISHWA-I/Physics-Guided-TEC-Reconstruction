import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry
from models.feature_embedding import FeatureEmbedding
from models.temporal_embedding import PhysicsAwareTemporalEmbedding, FeatureTypeEmbedding
from models.mamba_block import MambaBlock
from models.normalization import get_normalization_layer
from utils.logger import get_model_logger

logger = get_model_logger("TemporalEncoder")

@ModelRegistry.register("TemporalEncoder")
class TemporalEncoder(BaseModel):
    """
    Physics-Aware Temporal Encoder based on Mamba-2.
    Processes historical temporal observations (TEC, Solar, Geomagnetic indices)
    into a dense temporal latent representation.
    """

    def __init__(self, config: ModelConfig, debug_mode: bool = False):
        """
        Initializes the Temporal Encoder pipeline.

        Args:
            config (ModelConfig): The global model configuration.
            debug_mode (bool): If True, prints tensor shapes during the forward pass.
        """
        super().__init__(config)
        self.debug_mode = debug_mode
        
        if len(config.temporal_features) == 0:
            logger.error("No temporal features provided in configuration.")
            raise ValueError("TemporalEncoder requires a non-empty list of temporal_features.")

        # 1. Feature Embedding (Project 6 features to Hidden Dimension)
        self.feature_embed = FeatureEmbedding(
            in_features=len(config.temporal_features),
            out_features=config.embedding_dimension,
            dropout=config.dropout,
            norm_type=config.normalization
        )
        
        # 2. Temporal & Feature-Type Embeddings
        self.temporal_embed = PhysicsAwareTemporalEmbedding(
            config=config,
            initial_seq_len=config.window_size
        )
        
        self.feature_type_embed = FeatureTypeEmbedding(
            d_model=config.embedding_dimension,
            feature_names=config.temporal_features
        )
        
        # If embedding_dimension != hidden_dimension, we need a projection layer before Mamba blocks
        if config.embedding_dimension != config.hidden_dimension:
            self.emb_to_hidden = nn.Linear(config.embedding_dimension, config.hidden_dimension)
        else:
            self.emb_to_hidden = nn.Identity()

        # 3. Stack of Mamba Blocks
        blocks = []
        for i in range(config.number_of_blocks):
            blocks.append(
                MambaBlock(
                    d_model=config.hidden_dimension,
                    d_state=config.state_expansion_factor,
                    d_conv=config.conv_kernel_size,
                    expand=config.expand_factor,
                    dropout=config.dropout,
                    norm_type=config.normalization,
                    use_checkpointing=config.use_gradient_checkpointing
                )
            )
        self.mamba_layers = nn.Sequential(*blocks)
        
        # 4. Final Normalization
        self.final_norm = get_normalization_layer(config.normalization, config.hidden_dimension)
        
        logger.info(f"TemporalEncoder initialized with {config.number_of_blocks} Mamba blocks.")

    def forward(self, x: torch.Tensor, geo_feats: torch.Tensor, storm_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Temporal Encoder.

        Args:
            x (torch.Tensor): Raw temporal features. Shape: (Batch, Sequence, Features)
            geo_feats (torch.Tensor): Geographic features for physics injection.
            storm_feats (torch.Tensor): Storm parameters for physics injection.

        Returns:
            torch.Tensor: Latent temporal representation. Shape: (Batch, Sequence, HiddenDimension)
        """
        if self.debug_mode:
            print(f"[DEBUG] Input Shape: {x.shape}")
            
        # Basic validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (Batch, Sequence, Features), got {x.dim()}D")
            
        b, seq, f = x.shape
        if f != len(self.config.temporal_features):
            raise ValueError(f"Expected {len(self.config.temporal_features)} features, got {f}")
            
        # 1. Project Features
        x = self.feature_embed(x)
        if self.debug_mode:
            print(f"[DEBUG] After FeatureEmbedding: {x.shape}")
            
        # 2. Inject Contextual Embeddings
        x = self.temporal_embed(x, geo_feats, storm_feats)
        x = self.feature_type_embed(x)
        if self.debug_mode:
            print(f"[DEBUG] After Context Embeddings: {x.shape}")
            
        # Project to hidden space if necessary
        x = self.emb_to_hidden(x)
        
        # 3. Process through Mamba Sequence
        x = self.mamba_layers(x)
        if self.debug_mode:
            print(f"[DEBUG] After Mamba Blocks: {x.shape}")
            
        # 4. Final Norm
        out = self.final_norm(x)
        
        return out

