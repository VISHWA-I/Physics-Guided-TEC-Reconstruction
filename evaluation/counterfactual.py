import torch
import torch.nn as nn
from typing import Dict, Tuple

class CounterfactualSimulator:
    """
    Simulates "What-If" scenarios to test model robustness and physical understanding.
    Perturbs specific physical drivers (Kp index, F10.7, etc.) and measures Delta TEC.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
    @torch.no_grad()
    def simulate_storm(self, inputs: Dict[str, torch.Tensor], delta_kp: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates the onset of a geomagnetic storm by increasing the Kp index.
        Assumes Kp is the 0th index in the storm_feats tensor.
        """
        # Original Prediction
        orig_out = self.model(**inputs)
        orig_topside = orig_out.topside_tec
        
        # Perturbed Prediction
        perturbed_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # Increment Kp
        perturbed_inputs['storm_feats'][:, 0] += delta_kp
        
        pert_out = self.model(**perturbed_inputs)
        pert_topside = pert_out.topside_tec
        
        delta_tec = pert_topside - orig_topside
        return orig_topside, delta_tec
        
    @torch.no_grad()
    def simulate_solar_flare(self, inputs: Dict[str, torch.Tensor], delta_f107: float = 50.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates a solar flare by increasing the F10.7 index.
        Assumes F10.7 is the 1st index in the storm_feats tensor.
        """
        orig_out = self.model(**inputs)
        orig_topside = orig_out.topside_tec
        
        perturbed_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # Increment F10.7
        perturbed_inputs['storm_feats'][:, 1] += delta_f107
        
        pert_out = self.model(**perturbed_inputs)
        pert_topside = pert_out.topside_tec
        
        delta_tec = pert_topside - orig_topside
        return orig_topside, delta_tec
