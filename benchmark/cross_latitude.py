import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class CrossLatitudeBenchmark:
    """
    Evaluates spatial generalization across distinct geomagnetic latitudes.
    """
    
    @staticmethod
    def evaluate(latitudes: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculates performance banded by absolute latitude.
        """
        abs_lats = np.abs(latitudes)
        metrics = ScientificMetrics()
        results = {}
        
        # Equatorial (0-20), Mid (20-60), High (60-90)
        bands = {
            "Equatorial": (0, 20),
            "Mid_Latitude": (20, 60),
            "High_Latitude": (60, 90)
        }
        
        for name, (low, high) in bands.items():
            mask = (abs_lats >= low) & (abs_lats < high)
            if np.sum(mask) > 0:
                results[name] = metrics.compute(y_true[mask], y_pred[mask])
                
        return results
