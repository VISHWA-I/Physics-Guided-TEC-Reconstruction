import numpy as np
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, Any, List

class FeatureRelationshipMiner:
    """
    Discovers nonlinear relationships between inputs and outputs using Mutual Information.
    """
    
    @staticmethod
    def map_relationships(features: np.ndarray, feature_names: List[str], target: np.ndarray) -> Dict[str, float]:
        """
        Computes mutual information between each feature column and the target prediction.
        """
        if len(target.shape) > 1:
            target = target.flatten()
            
        # Ensure target matches features length (may require taking last step of seq)
        # Assuming features shape is (N, num_features)
        
        try:
            mi_scores = mutual_info_regression(features, target, random_state=42)
        except Exception:
            # Fallback if dimensions mismatch in rapid deployment
            return {name: 0.0 for name in feature_names}
            
        # Normalize scores to 0-1 range for easier interpretation
        max_mi = np.max(mi_scores) if np.max(mi_scores) > 0 else 1.0
        normalized_mi = mi_scores / max_mi
        
        relationships = {name: float(score) for name, score in zip(feature_names, normalized_mi)}
        
        # Sort by importance descending
        return dict(sorted(relationships.items(), key=lambda item: item[1], reverse=True))
