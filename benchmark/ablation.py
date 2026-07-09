import torch
import torch.nn as nn
from typing import Dict, Any

class AblationStudy:
    """
    Evaluates the contribution of individual architectural components.
    Performs Feature/Activation Ablation by dynamically intercepting the forward pass
    and zeroing out targeted tensors or bypassing specific modules, without requiring retraining.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
    @torch.no_grad()
    def run_ablation(self, inputs: Dict[str, torch.Tensor], component: str) -> Dict[str, torch.Tensor]:
        """
        Executes a forward pass with the specified component ablated.
        """
        # Create a deep copy of inputs to avoid modifying the original references
        ablated_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 1. Ablate Storm Encoder
        if component == "Without_Storm_Encoder":
            ablated_inputs["storm_feats"] = torch.zeros_like(ablated_inputs["storm_feats"])
            
        # 2. Ablate Physics Memory Bank
        elif component == "Without_Physics_Memory":
            ablated_inputs["physics_feats"] = torch.zeros_like(ablated_inputs["physics_feats"])
            
        # 3. Ablate Spatial/Geophysical Encoder
        elif component == "Without_Geo_Encoder":
            ablated_inputs["geo_feats"] = torch.zeros_like(ablated_inputs["geo_feats"])
            
        # Note: True structural ablation (e.g., removing Cross-Attention entirely) 
        # requires modifying the model definition and loading specifically trained weights for that structure.
        # Since the prompt forbids retraining, we simulate architectural ablation by starving the modules of information.
        
        # Execute model
        output = self.model(**ablated_inputs)
        
        return {
            "topside_tec": output.topside_tec,
            "net_tec": output.net_tec,
            "electron_density": output.electron_density
        }
