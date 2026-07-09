import numpy as np
from typing import Dict, List, Tuple
from evaluation.metrics import ScientificMetrics

class StormAnalysis:
    """
    Evaluates model performance stratified by Geomagnetic Storm intensity.
    """
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, kp_index: np.ndarray, dst_index: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Bins data by Storm Conditions and evaluates metrics independently.
        """
        # Ensure flat arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        kp = np.asarray(kp_index).flatten()
        dst = np.asarray(dst_index).flatten()
        
        results = {}
        metrics_calc = ScientificMetrics()
        
        # 1. Quiet Conditions (Kp <= 2, Dst > -30)
        quiet_mask = (kp <= 2.0) | (dst > -30)
        if np.any(quiet_mask):
            results["Quiet"] = metrics_calc.compute(y_true[quiet_mask], y_pred[quiet_mask])
            
        # 2. Moderate Storms (Kp 3-5, Dst -30 to -80)
        moderate_mask = ((kp >= 3.0) & (kp < 6.0)) | ((dst <= -30) & (dst > -80))
        if np.any(moderate_mask):
            results["Moderate_Storm"] = metrics_calc.compute(y_true[moderate_mask], y_pred[moderate_mask])
            
        # 3. Strong/Extreme Storms (Kp >= 6, Dst <= -80)
        strong_mask = (kp >= 6.0) | (dst <= -80)
        if np.any(strong_mask):
            results["Strong_Storm"] = metrics_calc.compute(y_true[strong_mask], y_pred[strong_mask])
            
        # Calculate Storm Degradation Factor (Strong RMSE / Quiet RMSE)
        if "Quiet" in results and "Strong_Storm" in results:
            quiet_rmse = results["Quiet"].get("RMSE", 1.0)
            strong_rmse = results["Strong_Storm"].get("RMSE", 1.0)
            results["Storm_Degradation_Factor"] = strong_rmse / (quiet_rmse + 1e-8)
            
        return results
