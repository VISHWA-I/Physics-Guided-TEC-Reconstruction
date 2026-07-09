import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from typing import Dict, Any

class PatternMiner:
    """
    Discovers repeated TEC patterns and macroscopic behaviors using rapid clustering.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        # DBSCAN for density-based arbitrary shapes, KMeans for rapid centroids
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init="auto")
        
    def mine_patterns(self, predictions: np.ndarray, features: np.ndarray) -> Dict[str, Any]:
        """
        Groups predictions into behavioral clusters.
        Expects flattened arrays or 2D feature matrices.
        """
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
            
        # Fast K-Means to identify primary behavioral states (e.g., Low, Normal, High, Extreme)
        kmeans_labels = self.kmeans.fit_predict(predictions)
        
        # Identify dense micro-patterns using DBSCAN
        dbscan_labels = self.dbscan.fit_predict(predictions)
        num_anomalous_patterns = np.sum(dbscan_labels == -1)
        
        return {
            "primary_behavior_states": kmeans_labels,
            "dense_pattern_groups": dbscan_labels,
            "unique_patterns_found": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            "unclustered_anomalies": int(num_anomalous_patterns)
        }
