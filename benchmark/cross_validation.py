import numpy as np
from typing import Dict

class GeneralizationValidation:
    """
    Core engine for Leave-One-Out validation routines.
    """
    
    @staticmethod
    def compute_gap(train_metric: float, test_metric: float) -> float:
        """
        Calculates the generalization gap percentage.
        """
        return ((test_metric - train_metric) / (train_metric + 1e-8)) * 100.0
