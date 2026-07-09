import torch
import torch.nn as nn
from typing import Optional

from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry
from models.model_output import ModelOutput

# Phase 2.2
from models.temporal_encoder import TemporalEncoder

# Phase 2.3
from models.adaptive_fusion import AdaptiveFusionLayer

# Phase 2.4
from models.tkan_decoder import TKANDecoder

# Phase 2.5
from models.topside_head import TopsideTECHead
from models.physics_consistency import PhysicsConstraintEngine
from models.nettec_head import NetTECHead
from models.electron_density_head import ElectronDensityHead
from models.gnss_delay_head import GNSSDelayHead
from models.uncertainty_head import PredictionUncertaintyHead
from models.auxiliary_head import AuxiliaryReconstructionHead

from utils.logger import get_model_logger

logger = get_model_logger("HybridModel")

@ModelRegistry.register("HybridModel")
class HybridModel(BaseModel):
    """
    Physics-Guided Multi-Branch Memory-Augmented Mamba-TKAN Network.
    This is the master model orchestrating Phase 2.2 -> 2.3 -> 2.4 -> 2.5.
    """

    def __init__(self, config: ModelConfig, debug_mode: bool = False):
        super().__init__(config)
        self.debug_mode = debug_mode
        self.config = config
        
        # --- Phase 2.2: Temporal Processing ---
        self.temporal_encoder = TemporalEncoder(config)
        
        # --- Phase 2.3: Feature Fusion (Physics, Geo, Storm) ---
        self.adaptive_fusion = AdaptiveFusionLayer(config, debug_mode=False)
        
        # --- Phase 2.4: TKAN Memory Decoder ---
        self.tkan_decoder = TKANDecoder(config, debug_mode=False)
        
        # --- Phase 2.5: Hierarchical Prediction Heads ---
        hidden_dim = config.hidden_dimension
        dropout = config.dropout
        
        self.topside_head = TopsideTECHead(hidden_dim, dropout)
        self.nettec_head = NetTECHead(hidden_dim, dropout)
        self.electron_density_head = ElectronDensityHead(hidden_dim, dropout)
        self.gnss_delay_head = GNSSDelayHead(
            hidden_dim=hidden_dim, 
            geometry_dim=config.geometry_features_dim, 
            dropout=dropout
        )
        self.uncertainty_head = PredictionUncertaintyHead(hidden_dim)
        
        # Instantiate the Physics Constraint Engine and wrap the downstream heads
        self.physics_constraint_engine = PhysicsConstraintEngine(
            nettec_head=self.nettec_head,
            electron_density_head=self.electron_density_head,
            gnss_delay_head=self.gnss_delay_head,
            debug_mode=debug_mode
        )
        
        if config.enable_auxiliary:
            self.auxiliary_head = AuxiliaryReconstructionHead(hidden_dim)
        else:
            self.auxiliary_head = None
            
        logger.info("HybridModel (Mamba-TKAN) fully assembled.")

    def forward(self, 
                temporal_seq: torch.Tensor, 
                physics_feats: torch.Tensor, 
                geo_feats: torch.Tensor, 
                storm_feats: torch.Tensor,
                bottomside_tec: torch.Tensor,
                geometry_feats: Optional[torch.Tensor] = None) -> ModelOutput:
        """
        Master Forward Pass.

        Args:
            temporal_seq: (Batch, Seq, Features)
            physics_feats: (Batch, NumPhysics)
            geo_feats: (Batch, NumGeo)
            storm_feats: (Batch, NumStorm)
            bottomside_tec: (Batch, Seq, 1) - Required for physical consistency
            geometry_feats: Optional (Batch, Seq, GeometryDim)

        Returns:
            ModelOutput: Complete dataclass containing all predictions and metrics.
        """
        metrics = {}
        
        # 1. Phase 2.2: Temporal Encoding (Mamba)
        temporal_rep = self.temporal_encoder(temporal_seq, geo_feats, storm_feats)
        
        # 2. Phase 2.3: Physics Fusion
        fused_latent, fusion_metrics = self.adaptive_fusion(
            temporal_rep, physics_feats, geo_feats, storm_feats
        )
        metrics.update(fusion_metrics)
        
        # 3. Phase 2.4: Memory-Augmented TKAN
        # We don't use the topside TEC directly from TKAN decoder here; we use it to get the deep latent state.
        # But TKANDecoder outputs tec_prediction, hidden_out, tkan_metrics
        tkan_tec_coarse, hidden_rep, tkan_metrics = self.tkan_decoder(fused_latent)
        metrics.update(tkan_metrics)
        
        if self.debug_mode:
            print(f"[DEBUG] Shared Latent Shape: {hidden_rep.shape}")
            
        # 4. Phase 2.5: Hierarchical Prediction
        # 4.1 Topside TEC
        topside_tec = self.topside_head(hidden_rep)
        if self.debug_mode:
            print(f"[DEBUG] Topside TEC Shape (Initial): {topside_tec.shape}")
            
        # 4.2 Physics Constraint Engine (Net TEC -> Density -> Delay)
        net_tec, electron_density, gnss_delays, physics_intermediates = self.physics_constraint_engine(
            bottomside_tec=bottomside_tec,
            predicted_topside_tec=topside_tec,
            hidden_rep=hidden_rep,
            geometry_feats=geometry_feats
        )
        
        # 4.3 Uncertainty (Focused on the primary Topside TEC task)
        conf_score, lower, upper = self.uncertainty_head(hidden_rep, topside_tec)
        if self.debug_mode:
            print(f"[DEBUG] Confidence Shape: {conf_score.shape}")
            
        # 4.7 Auxiliary Self-Supervised Tasks (Training Only)
        aux_preds = {}
        if self.auxiliary_head is not None and self.training:
            aux_preds = self.auxiliary_head(hidden_rep)
            
        # Construct Output DataClass
        return ModelOutput(
            topside_tec=topside_tec,
            net_tec=net_tec,
            electron_density=electron_density,
            gnss_delays=gnss_delays,
            confidence_score=conf_score,
            lower_bound=lower,
            upper_bound=upper,
            auxiliary_predictions=aux_preds,
            physics_intermediates=physics_intermediates,
            metrics=metrics
        )
