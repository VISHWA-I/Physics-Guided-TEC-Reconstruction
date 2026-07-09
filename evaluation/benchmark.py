import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class Benchmark:
    """
    Compares the Hybrid Mamba-TKAN model against standard baselines.
    """
    
    def __init__(self):
        self.metrics_calc = ScientificMetrics()
        
    def run_comparison(self, y_true: np.ndarray, our_pred: np.ndarray, baseline_preds: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Computes metrics for our model and compares against provided baseline arrays.
        
        Args:
            y_true: Ground truth Topside TEC.
            our_pred: Our model's predictions.
            baseline_preds: Dictionary mapping baseline names (e.g., 'IRI-Plas', 'LSTM') to their prediction arrays.
            
        Returns:
            Nested dictionary of metrics per model.
        """
        results = {}
        
        # Our Model
        results["Hybrid_Mamba_TKAN"] = self.metrics_calc.compute(y_true, our_pred)
        
        # Baselines
        for model_name, pred in baseline_preds.items():
            # Check length matches to avoid broadcasting errors
            if len(pred) != len(y_true):
                print(f"[Benchmark Warning] Length mismatch for {model_name}. Skipping.")
                continue
                
            results[model_name] = self.metrics_calc.compute(y_true, pred)
            
            # Add improvement metric
            if "RMSE" in results[model_name] and "RMSE" in results["Hybrid_Mamba_TKAN"]:
                our_rmse = results["Hybrid_Mamba_TKAN"]["RMSE"]
                base_rmse = results[model_name]["RMSE"]
                improvement = ((base_rmse - our_rmse) / (base_rmse + 1e-8)) * 100
                results[model_name]["Improvement_over_baseline_pct"] = float(improvement)
                
        return results
