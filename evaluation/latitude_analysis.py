import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class LatitudeAnalysis:
    """
    Evaluates model performance stratified by Geomagnetic Latitude bands.
    """
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, latitude: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Bins data by Latitude Region and evaluates metrics independently.
        Assumes latitude is provided in degrees (geomagnetic).
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        lat = np.abs(np.asarray(latitude).flatten()) # Use absolute for North/South symmetry
        
        results = {}
        metrics_calc = ScientificMetrics()
        
        # Low Latitude (Equatorial Anomaly Region): 0 - 30 deg
        low_mask = (lat >= 0) & (lat < 30)
        if np.any(low_mask):
            results["Low_Latitude"] = metrics_calc.compute(y_true[low_mask], y_pred[low_mask])
            
        # Mid Latitude (Trough Region): 30 - 60 deg
        mid_mask = (lat >= 30) & (lat < 60)
        if np.any(mid_mask):
            results["Mid_Latitude"] = metrics_calc.compute(y_true[mid_mask], y_pred[mid_mask])
            
        # High Latitude (Auroral / Polar Region): 60 - 90 deg
        high_mask = (lat >= 60) & (lat <= 90)
        if np.any(high_mask):
            results["High_Latitude"] = metrics_calc.compute(y_true[high_mask], y_pred[high_mask])
            
        return results
