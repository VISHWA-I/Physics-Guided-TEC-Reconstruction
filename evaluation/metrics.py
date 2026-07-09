import numpy as np
from scipy import stats
from typing import Dict, Union

class ScientificMetrics:
    """
    Computes comprehensive statistical metrics for scientific evaluation.
    Handles NumPy arrays directly.
    """
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Computes RMSE, MAE, MAPE, R², Correlation, Median Error, and 95th Percentile Error.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove NaNs if any exist in observation gaps
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            return {}
            
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # Standard Metrics
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs_errors)
        mape = np.mean(abs_errors / (np.abs(y_true) + 1e-8)) * 100.0
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Correlations
        pearson_corr, _ = stats.pearsonr(y_true, y_pred)
        spearman_corr, _ = stats.spearmanr(y_true, y_pred)
        
        # Error Distributions
        median_error = np.median(abs_errors)
        p95_error = np.percentile(abs_errors, 95)
        
        return {
            f"{prefix}RMSE": float(rmse),
            f"{prefix}MAE": float(mae),
            f"{prefix}MAPE": float(mape),
            f"{prefix}R2": float(r2),
            f"{prefix}Pearson_r": float(pearson_corr),
            f"{prefix}Spearman_rho": float(spearman_corr),
            f"{prefix}Median_Error": float(median_error),
            f"{prefix}95th_Percentile_Error": float(p95_error)
        }
