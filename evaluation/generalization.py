import numpy as np
from typing import Dict, List

class GeneralizationAnalysis:
    """
    Evaluates spatial and temporal robustness.
    """
    
    @staticmethod
    def evaluate(y_true_train: np.ndarray, y_pred_train: np.ndarray, 
                 y_true_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, float]:
        """
        Computes Generalization Gap.
        A massive gap indicates severe overfitting.
        """
        train_rmse = np.sqrt(np.mean((y_pred_train - y_true_train) ** 2))
        test_rmse = np.sqrt(np.mean((y_pred_test - y_true_test) ** 2))
        
        # Gap percentage: (Test - Train) / Train
        gap_pct = ((test_rmse - train_rmse) / (train_rmse + 1e-8)) * 100.0
        
        return {
            "Train_RMSE": float(train_rmse),
            "Test_RMSE": float(test_rmse),
            "Generalization_Gap_Pct": float(gap_pct)
        }
        
    @staticmethod
    def evaluate_leave_one_station_out(station_results: Dict[str, float]) -> Dict[str, float]:
        """
        Analyzes the variance of RMSE across stations in LOSO cross-validation.
        """
        if not station_results:
            return {}
            
        rmses = list(station_results.values())
        return {
            "Mean_Station_RMSE": float(np.mean(rmses)),
            "Std_Station_RMSE": float(np.std(rmses)),
            "Max_Station_RMSE": float(np.max(rmses)),
            "Min_Station_RMSE": float(np.min(rmses))
        }
