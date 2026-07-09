import torch
from typing import Dict, Any

from app.prediction_engine import PredictionEngine

class SimulationEngine:
    """
    Runs real-time 'What-If' scenarios by perturbing physical inputs and observing the output.
    Does not retrain the model.
    """
    
    def __init__(self, pred_engine: PredictionEngine):
        self.pred_engine = pred_engine
        
    def simulate(self, 
                 base_inputs: Dict[str, torch.Tensor], 
                 overrides: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Modifies base inputs based on overrides and returns new predictions.
        Assumes indices based on Phase 1 feature layout:
        storm_feats: [Kp, Dst, F10.7, AE, Ap]
        """
        perturbed = {k: v.clone() for k, v in base_inputs.items() if v is not None}
        
        # Override Kp
        if "Kp" in overrides:
            perturbed["storm_feats"][:, 0] = overrides["Kp"]
            
        # Override Dst
        if "Dst" in overrides:
            perturbed["storm_feats"][:, 1] = overrides["Dst"]
            
        # Override F10.7
        if "F10.7" in overrides:
            perturbed["storm_feats"][:, 2] = overrides["F10.7"]
            
        # Run inference on perturbed data
        new_preds = self.pred_engine.predict(**perturbed)
        return new_preds
