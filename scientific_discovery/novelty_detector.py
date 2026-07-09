import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any

class NoveltyDetector:
    """
    Identifies completely unprecedented ionospheric events using Isolation Forests.
    Extremely fast for < 2s operational requirement.
    """
    
    def __init__(self, contamination: float = 0.01):
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        
    def detect(self, features: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Flags samples that differ fundamentally from the majority distribution.
        """
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
            
        combined = np.hstack([features, predictions])
        
        # Returns 1 for inliers, -1 for outliers
        preds = self.iso_forest.fit_predict(combined)
        anomaly_scores = self.iso_forest.decision_function(combined)
        
        novel_mask = preds == -1
        
        return {
            "is_novel": novel_mask,
            "novelty_scores": anomaly_scores,
            "num_novel_events": int(np.sum(novel_mask))
        }
