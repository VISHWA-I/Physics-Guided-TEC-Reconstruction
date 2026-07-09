import numpy as np
from typing import Dict, Any

class QualityScore:
    """
    Calculates a real-time Operational Readiness Score (0-100) based on
    prediction confidence, data completeness, and anomaly flags.
    """
    
    @staticmethod
    def calculate(confidence_dict: Dict[str, Any], anomaly_dict: Dict[str, Any], storm_dict: Dict[str, str]) -> float:
        """
        Returns a single scalar representing how much operational trust to place in the current prediction window.
        """
        score = 100.0
        
        # 1. Deduct based on Confidence
        mean_conf = np.mean(confidence_dict.get("Confidence_Score", 1.0))
        # If confidence is 0.5, deduct 25 points
        score -= (1.0 - mean_conf) * 50.0
        
        # 2. Deduct based on Anomalies
        if anomaly_dict.get("negative_topside_detected"): score -= 40.0
        if anomaly_dict.get("extreme_tec_detected"): score -= 20.0
        if anomaly_dict.get("massive_jump_detected"): score -= 15.0
        
        # 3. Adjust for extreme storms (models are inherently less reliable during chaos)
        if storm_dict.get("Warning_Level") == "CRITICAL":
            score -= 10.0
            
        return float(max(0.0, min(100.0, score)))
