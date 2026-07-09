import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class SeasonalBenchmark:
    """
    Evaluates temporal generalization across distinct seasonal profiles.
    """
    
    @staticmethod
    def evaluate(doy: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Bands performance by season (assuming Northern Hemisphere for simplicity in this pure benchmark module).
        """
        metrics = ScientificMetrics()
        results = {}
        
        winter_mask = (doy < 80) | (doy >= 355)
        summer_mask = (doy >= 172) & (doy < 266)
        equinox_mask = ~winter_mask & ~summer_mask
        
        if np.sum(winter_mask) > 0: results["Winter"] = metrics.compute(y_true[winter_mask], y_pred[winter_mask])
        if np.sum(summer_mask) > 0: results["Summer"] = metrics.compute(y_true[summer_mask], y_pred[summer_mask])
        if np.sum(equinox_mask) > 0: results["Equinox"] = metrics.compute(y_true[equinox_mask], y_pred[equinox_mask])
        
        return results
