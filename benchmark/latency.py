import torch
import torch.nn as nn
import time
from typing import Dict

class LatencyProfiler:
    """
    Measures real-world inference speed to verify operational suitability
    at 4, 5, 10, and 15-minute cadences.
    """
    
    @staticmethod
    @torch.no_grad()
    def profile_inference(model: nn.Module, dummy_inputs: Dict[str, torch.Tensor], num_runs: int = 100) -> Dict[str, float]:
        """
        Runs the model `num_runs` times and calculates average latency.
        Warms up the hardware first.
        """
        device = dummy_inputs["temporal_seq"].device
        
        # Warmup
        for _ in range(10):
            model(**dummy_inputs)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(num_runs):
            model(**dummy_inputs)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / num_runs
        
        # 4 min cadence = 240,000 ms. If avg_time < 240,000, it's suitable.
        return {
            "Average_Inference_ms": avg_time_ms,
            "Suitable_for_4min_cadence": bool(avg_time_ms < 240000),
            "Suitable_for_5min_cadence": bool(avg_time_ms < 300000)
        }
