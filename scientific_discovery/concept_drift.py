import numpy as np
from typing import Dict, Any

class ConceptDriftMonitor:
    """
    Monitors statistical distribution shifts indicating long-term changes 
    (e.g., Solar Cycle transitions).
    """
    
    @staticmethod
    def detect_drift(current_window: np.ndarray, historical_mean: float, historical_std: float, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Calculates simple z-score drift across a window.
        """
        current_mean = np.mean(current_window)
        z_score = np.abs((current_mean - historical_mean) / (historical_std + 1e-8))
        
        is_drifting = bool(z_score > threshold)
        
        return {
            "is_drifting": is_drifting,
            "drift_z_score": float(z_score),
            "current_mean": float(current_mean)
        }
