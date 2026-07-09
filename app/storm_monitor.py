import numpy as np
from typing import Dict, Union

class StormMonitor:
    """
    Evaluates real-time solar wind drivers to trigger Storm Warnings.
    """
    
    @staticmethod
    def monitor(kp_index: Union[float, np.ndarray], dst_index: Union[float, np.ndarray]) -> Dict[str, str]:
        """
        Given the current (or windowed) Kp and Dst, categorizes the operational state.
        Returns the most severe state found.
        """
        kp = np.max(np.asarray(kp_index))
        dst = np.min(np.asarray(dst_index))
        
        if kp >= 8.0 or dst <= -150:
            category = "Extreme Storm"
            warning_level = "CRITICAL"
        elif kp >= 6.0 or dst <= -80:
            category = "Strong Storm"
            warning_level = "HIGH"
        elif kp >= 3.0 or dst <= -30:
            category = "Moderate Storm"
            warning_level = "WARNING"
        else:
            category = "Quiet"
            warning_level = "NORMAL"
            
        return {
            "Max_Kp": float(kp),
            "Min_Dst": float(dst),
            "Storm_Category": category,
            "Warning_Level": warning_level
        }
