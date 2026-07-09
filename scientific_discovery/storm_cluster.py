import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Any

class StormClusterer:
    """
    Specialized clustering for storm recovery profiles based on Kp/Dst dynamics.
    """
    
    def __init__(self):
        self.kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init="auto")
        
    def analyze_storm_responses(self, storm_feats: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Maps the physical storm drivers against the model's TEC response to classify storm archetypes.
        """
        # Combine drivers and response for clustering
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
            
        combined_matrix = np.hstack([storm_feats, predictions])
        
        labels = self.kmeans.fit_predict(combined_matrix)
        
        # Heuristic archetype assignment based on centroids
        archetypes = ["Positive Phase Dominant", "Negative Phase Dominant", "Delayed Recovery"]
        # Simplified mapping for demonstration
        mapped_labels = [archetypes[l % 3] for l in labels]
        
        return {
            "storm_cluster_ids": labels,
            "storm_archetypes": mapped_labels
        }
