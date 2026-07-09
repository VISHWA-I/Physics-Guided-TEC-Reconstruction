import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Any

class OutlierAnalyzer:
    """
    Identifies localized anomalies using Local Outlier Factor (LOF).
    """
    
    def __init__(self, n_neighbors: int = 20):
        # LOF is fast for moderate batch sizes
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, n_jobs=-1)
        
    def analyze(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Flags point-anomalies in the prediction stream.
        """
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
            
        # Returns 1 for inliers, -1 for outliers
        preds = self.lof.fit_predict(predictions)
        lof_scores = -self.lof.negative_outlier_factor_ # Convert to positive scores
        
        outlier_mask = preds == -1
        
        return {
            "is_outlier": outlier_mask,
            "outlier_scores": lof_scores,
            "num_outliers": int(np.sum(outlier_mask))
        }
