import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class StormBenchmark:
    """
    Evaluates temporal generalization specifically focused on extreme space weather events.
    """
    
    @staticmethod
    def evaluate(kp_index: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculates performance during extreme vs quiet times.
        """
        metrics = ScientificMetrics()
        results = {}
        
        quiet_mask = kp_index <= 3.0
        storm_mask = kp_index > 5.0
        
        if np.sum(quiet_mask) > 0:
            results["Quiet_Time"] = metrics.compute(y_true[quiet_mask], y_pred[quiet_mask])
            
        if np.sum(storm_mask) > 0:
            results["Storm_Time"] = metrics.compute(y_true[storm_mask], y_pred[storm_mask])
            
        return results
