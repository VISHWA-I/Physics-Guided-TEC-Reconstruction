from dataclasses import dataclass, field
import torch
from typing import Dict, Optional

@dataclass
class ModelOutput:
    """
    Standardized data structure returned by the HybridModel.
    Contains all hierarchical predictions, physical intermediate states, and introspection metrics.
    """
    
    # 1. Primary Predictions
    topside_tec: torch.Tensor
    net_tec: torch.Tensor
    electron_density: torch.Tensor
    
    # 2. GNSS Delay Predictions (Vertical, Slant, Time, Constellations)
    gnss_delays: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 3. Uncertainty Estimation (Aleatoric)
    confidence_score: Optional[torch.Tensor] = None
    lower_bound: Optional[torch.Tensor] = None
    upper_bound: Optional[torch.Tensor] = None
    
    # 4. Auxiliary Self-Supervised Targets (Only active during training)
    auxiliary_predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 5. Physics Intermediate Variables (For Loss Functions)
    physics_intermediates: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 6. Introspection Metrics (Attention Maps, Gate Weights, Latent States)
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)
