import numpy as np
from typing import Dict

class ConfidenceEngine:
    """
    Translates raw prediction uncertainty bounds into operational confidence metrics.
    """
    
    @staticmethod
    def evaluate(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Parses confidence scores and bounds from the prediction dictionary.
        Outputs a structured dictionary mapping confidence to timestamps (if arrays).
        """
        conf_score = np.squeeze(np.asarray(predictions.get("confidence_score", 1.0)))
        lower = np.squeeze(np.asarray(predictions.get("lower_bound", 0.0)))
        upper = np.squeeze(np.asarray(predictions.get("upper_bound", 0.0)))
        
        # Uncertainty width
        width = np.abs(upper - lower)
        
        # A flag indicating if the model is highly uncertain (e.g., confidence < 0.3)
        # Note: In a real system, these thresholds are calibrated statistically.
        high_uncertainty_flag = conf_score < 0.3
        
        return {
            "Confidence_Score": conf_score,
            "Lower_Bound_95CI": lower,
            "Upper_Bound_95CI": upper,
            "Uncertainty_Width": width,
            "Requires_Manual_Review": high_uncertainty_flag
        }
