import torch
from typing import Dict, Any, Optional

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None
    print("Warning: Captum not installed. Explainability tools will be limited. Please run `pip install captum`.")

class ExplainabilityModule:
    """
    Extracts deep insights from the trained model using PyTorch Captum and hook-based attention maps.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        
        # To compute feature attribution on the input sequence, we need a wrapper 
        # since Captum IntegratedGradients expects a single continuous input tensor to a forward function.
        self.ig = IntegratedGradients(self._forward_wrapper) if IntegratedGradients else None
        
    def _forward_wrapper(self, temporal_seq: torch.Tensor) -> torch.Tensor:
        """
        A wrapper for Captum that isolates the temporal_seq attribution while mocking other inputs.
        In a full implementation, you would write wrappers for each input tensor type.
        """
        # Create zero tensors for other inputs matching batch size of temporal_seq
        batch_size = temporal_seq.size(0)
        device = temporal_seq.device
        
        # Dimensions based on config defaults. In a real system these are passed in during instantiation.
        phys_dim = 9
        geo_dim = 6
        storm_dim = 5
        geom_dim = 2
        seq_len = temporal_seq.size(1)
        
        physics_feats = torch.zeros(batch_size, phys_dim, device=device)
        geo_feats = torch.zeros(batch_size, geo_dim, device=device)
        storm_feats = torch.zeros(batch_size, storm_dim, device=device)
        bottomside_tec = torch.zeros(batch_size, seq_len, 1, device=device)
        geometry_feats = torch.zeros(batch_size, seq_len, geom_dim, device=device)
        
        output = self.model(
            temporal_seq=temporal_seq,
            physics_feats=physics_feats,
            geo_feats=geo_feats,
            storm_feats=storm_feats,
            bottomside_tec=bottomside_tec,
            geometry_feats=geometry_feats
        )
        
        # Return the target we want to explain (e.g. sum of Topside TEC)
        return output.topside_tec.sum(dim=1)
        
    def compute_integrated_gradients(self, temporal_seq: torch.Tensor, target_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Computes feature attributions using Integrated Gradients.
        """
        if self.ig is None:
            print("Captum not available.")
            return None
            
        # target defines which output index to attribute to (e.g. sequence index 0)
        attributions, delta = self.ig.attribute(
            temporal_seq, 
            target=target_idx, 
            return_convergence_delta=True
        )
        return attributions

    def extract_attention_maps(self, *inputs) -> Dict[str, torch.Tensor]:
        """
        Extracts internal attention matrices (Memory Retrieval, Physics Cross Attention).
        Assumes the model layers save their attention scores internally during the forward pass.
        """
        # Execute forward pass
        with torch.no_grad():
            self.model(*inputs)
            
        maps = {}
        # This requires the model to have saved its internal states during forward()
        # For Phase 4 we mock the extraction process for the architecture.
        if hasattr(self.model, "tkan_decoder") and hasattr(self.model.tkan_decoder, "last_attention_weights"):
            maps["TKAN_Attention"] = self.model.tkan_decoder.last_attention_weights
            
        if hasattr(self.model, "adaptive_fusion") and hasattr(self.model.adaptive_fusion, "last_gate_weights"):
            maps["Physics_Gate"] = self.model.adaptive_fusion.last_gate_weights
            
        return maps
