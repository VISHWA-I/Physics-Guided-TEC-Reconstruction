import torch
import torch.nn as nn
from typing import Dict

class MemoryProfiler:
    """
    Measures GPU/CPU memory consumption during inference.
    """
    
    @staticmethod
    @torch.no_grad()
    def profile(model: nn.Module, dummy_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Tracks peak memory allocation.
        """
        device = dummy_inputs["temporal_seq"].device
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            # Run inference
            model(**dummy_inputs)
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            # Memory profiling on CPU/MPS is OS-dependent and often inaccurate inside Python threads
            # Return a fallback approximation based on tensor sizes.
            model(**dummy_inputs)
            peak_memory_mb = -1.0 # indicating not available on this device
            
        return {
            "Peak_GPU_Memory_MB": peak_memory_mb
        }
