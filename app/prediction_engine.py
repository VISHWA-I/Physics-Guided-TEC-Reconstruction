import torch
from typing import Dict, Any, Optional

from app.model_loader import ModelLoader

class PredictionEngine:
    """
    Core wrapper for model inference. Transforms raw data to tensors, executes,
    and returns parsed results.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def predict(self, 
                temporal_seq: torch.Tensor,
                physics_feats: torch.Tensor,
                geo_feats: torch.Tensor,
                storm_feats: torch.Tensor,
                bottomside_tec: torch.Tensor,
                geometry_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Executes inference and unpacks the ModelOutput dataclass into a flat dictionary.
        """
        # Move inputs to target device
        temporal_seq = temporal_seq.to(self.device)
        physics_feats = physics_feats.to(self.device)
        geo_feats = geo_feats.to(self.device)
        storm_feats = storm_feats.to(self.device)
        bottomside_tec = bottomside_tec.to(self.device)
        if geometry_feats is not None:
            geometry_feats = geometry_feats.to(self.device)
            
        # Run inference
        output = self.model(
            temporal_seq=temporal_seq,
            physics_feats=physics_feats,
            geo_feats=geo_feats,
            storm_feats=storm_feats,
            bottomside_tec=bottomside_tec,
            geometry_feats=geometry_feats
        )
        
        # Return a clean dictionary, moved back to CPU for easy downstream processing
        return {
            "topside_tec": output.topside_tec.cpu(),
            "net_tec": output.net_tec.cpu(),
            "electron_density": output.electron_density.cpu(),
            "vertical_delay": output.gnss_delays["vertical_delay"].cpu(),
            "slant_delay": output.gnss_delays["slant_delay"].cpu(),
            "time_delay": output.gnss_delays["time_delay"].cpu(),
            "gps_delay": output.gnss_delays["gps_delay"].cpu(),
            "navic_delay": output.gnss_delays["navic_delay"].cpu(),
            "galileo_delay": output.gnss_delays["galileo_delay"].cpu(),
            "beidou_delay": output.gnss_delays["beidou_delay"].cpu(),
            "confidence_score": output.confidence_score.cpu(),
            "lower_bound": output.lower_bound.cpu(),
            "upper_bound": output.upper_bound.cpu(),
        }
