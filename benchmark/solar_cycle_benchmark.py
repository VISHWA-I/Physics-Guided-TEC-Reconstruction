import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class SolarCycleBenchmark:
    """
    Evaluates deep temporal generalization across completely different phases of the 11-year solar cycle.
    """
    
    @staticmethod
    def evaluate(f107: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Bands performance by F10.7 solar flux (Proxy for solar minimum vs maximum).
        """
        metrics = ScientificMetrics()
        results = {}
        
        solar_min_mask = f107 < 100.0
        solar_max_mask = f107 >= 150.0
        
        if np.sum(solar_min_mask) > 0:
            results["Solar_Minimum"] = metrics.compute(y_true[solar_min_mask], y_pred[solar_min_mask])
            
        if np.sum(solar_max_mask) > 0:
            results["Solar_Maximum"] = metrics.compute(y_true[solar_max_mask], y_pred[solar_max_mask])
            
        return results
