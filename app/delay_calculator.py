import numpy as np
from typing import Dict

class DelayCalculator:
    """
    Parses and formats frequency-dependent multi-constellation GNSS delays.
    Converts standard tensor predictions into physical engineering units (meters / nanoseconds).
    """
    
    @staticmethod
    def process_delays(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extracts delays and applies any final physical unit conversions if necessary.
        (Assumes the network outputs delays in raw standard units, typically meters for slant/vertical, 
        and nanoseconds for time_delay).
        """
        
        delays = {
            "Vertical_Delay_m": np.squeeze(np.asarray(predictions.get("vertical_delay", 0.0))),
            "Slant_Delay_m": np.squeeze(np.asarray(predictions.get("slant_delay", 0.0))),
            "Time_Delay_ns": np.squeeze(np.asarray(predictions.get("time_delay", 0.0))),
            
            # Constellation specific variations (Frequency-dependent dispersion adjustments)
            "GPS_L1_Delay_m": np.squeeze(np.asarray(predictions.get("gps_delay", 0.0))),
            "NavIC_L5_Delay_m": np.squeeze(np.asarray(predictions.get("navic_delay", 0.0))),
            "Galileo_E1_Delay_m": np.squeeze(np.asarray(predictions.get("galileo_delay", 0.0))),
            "BeiDou_B1_Delay_m": np.squeeze(np.asarray(predictions.get("beidou_delay", 0.0)))
        }
        
        return delays
