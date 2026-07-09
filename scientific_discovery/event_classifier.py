import numpy as np
from typing import Dict, Any, List

class EventClassifier:
    """
    Classifies space weather events into distinct operational categories.
    """
    
    @staticmethod
    def classify(kp_values: np.ndarray, novelty_mask: np.ndarray) -> List[str]:
        """
        Assigns categories: Quiet, Moderate, Strong, Extreme, or Novel.
        """
        classifications = []
        
        for kp, is_novel in zip(kp_values, novelty_mask):
            if is_novel:
                classifications.append("Novel Event")
            elif kp >= 7.0:
                classifications.append("Extreme Storm")
            elif kp >= 5.0:
                classifications.append("Strong Storm")
            elif kp >= 4.0:
                classifications.append("Moderate Storm")
            else:
                classifications.append("Quiet")
                
        return classifications
