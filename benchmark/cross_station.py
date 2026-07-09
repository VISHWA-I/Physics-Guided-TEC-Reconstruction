import numpy as np
from typing import Dict, List
from evaluation.metrics import ScientificMetrics

class CrossStationBenchmark:
    """
    Evaluates Leave-One-Station-Out (LOSO) spatial generalization.
    """
    
    @staticmethod
    def evaluate(station_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates metric variance across uniquely held-out stations.
        """
        unique_stations = np.unique(station_ids)
        metrics = ScientificMetrics()
        rmses = []
        
        for station in unique_stations:
            mask = station_ids == station
            if np.sum(mask) > 0:
                res = metrics.compute(y_true[mask], y_pred[mask])
                rmses.append(res.get("RMSE", 0.0))
                
        if not rmses:
            return {}
            
        return {
            "LOSO_Mean_RMSE": float(np.mean(rmses)),
            "LOSO_Std_RMSE": float(np.std(rmses)),
            "LOSO_Max_RMSE": float(np.max(rmses)),
            "LOSO_Min_RMSE": float(np.min(rmses))
        }
