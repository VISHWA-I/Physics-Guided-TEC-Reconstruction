import torch
from typing import Dict

class MetricsCalculator:
    """
    Computes regression metrics for evaluation.
    """
    
    @staticmethod
    def compute(predictions: torch.Tensor, targets: torch.Tensor, prefix: str = "") -> Dict[str, float]:
        """
        Computes RMSE, MAE, MAPE, R2.
        """
        # Ensure flat arrays
        pred = predictions.view(-1)
        true = targets.view(-1)
        
        mse = torch.mean((pred - true) ** 2)
        rmse = torch.sqrt(mse).item()
        mae = torch.mean(torch.abs(pred - true)).item()
        
        # MAPE with epsilon to prevent division by zero
        mape = torch.mean(torch.abs((true - pred) / (true + 1e-8))).item() * 100.0
        
        # R2 Score
        ss_res = torch.sum((true - pred) ** 2)
        ss_tot = torch.sum((true - torch.mean(true)) ** 2)
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        return {
            f"{prefix}rmse": rmse,
            f"{prefix}mae": mae,
            f"{prefix}mape": mape,
            f"{prefix}r2": r2
        }
