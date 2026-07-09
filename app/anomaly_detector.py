import numpy as np
from typing import Dict, Any

class AnomalyDetector:
    """
    Monitors operational inference output to flag unphysical or historically anomalous predictions.
    """
    
    def __init__(self, tec_hard_cap: float = 200.0, max_jump_tecu: float = 20.0):
        self.tec_hard_cap = tec_hard_cap
        self.max_jump_tecu = max_jump_tecu
        
    def detect(self, topside_tec: np.ndarray, net_tec: np.ndarray, electron_density: np.ndarray) -> Dict[str, Any]:
        """
        Scans arrays for anomalies.
        """
        topside = np.squeeze(np.asarray(topside_tec))
        net = np.squeeze(np.asarray(net_tec))
        density = np.squeeze(np.asarray(electron_density))
        
        anomalies = {
            "negative_topside_detected": bool(np.any(topside < 0)),
            "negative_density_detected": bool(np.any(density < 0)),
            "extreme_tec_detected": bool(np.any(net > self.tec_hard_cap)),
        }
        
        # Check for massive discontinuous jumps (requires > 1 element)
        if topside.size > 1:
            diffs = np.abs(np.diff(topside))
            anomalies["massive_jump_detected"] = bool(np.any(diffs > self.max_jump_tecu))
        else:
            anomalies["massive_jump_detected"] = False
            
        anomalies["is_anomalous"] = any(anomalies.values())
        
        return anomalies
