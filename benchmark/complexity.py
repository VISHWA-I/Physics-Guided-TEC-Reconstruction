import torch
import torch.nn as nn
from typing import Dict

class ComplexityProfiler:
    """
    Estimates theoretical computational complexity of the PyTorch model without relying
    on heavy external libraries (like fvcore/thop) that break in offline environments.
    """
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Returns total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "Total_Parameters": total_params,
            "Trainable_Parameters": trainable_params
        }
        
    @staticmethod
    def estimate_flops(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
        """
        A very coarse heuristic estimation of MACs/FLOPs based on parameter count and input sequence length.
        (For exact FLOPs in publications, fvcore is recommended in a connected environment).
        """
        total_params = sum(p.numel() for p in model.parameters())
        # Assuming sequence length is dim 1 of temporal_seq
        seq_len = inputs["temporal_seq"].size(1)
        batch_size = inputs["temporal_seq"].size(0)
        
        # Heuristic: Each parameter is used approximately once per sequence step in RNNs/Mamba
        # Multiply by 2 for Multiply-Accumulate (MAC)
        estimated_flops = total_params * seq_len * batch_size * 2
        return float(estimated_flops)
        
    @staticmethod
    def profile(model: nn.Module, dummy_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Returns full complexity profile.
        """
        params = ComplexityProfiler.count_parameters(model)
        flops = ComplexityProfiler.estimate_flops(model, dummy_inputs)
        
        return {
            "Total_Parameters_Millions": params["Total_Parameters"] / 1e6,
            "Estimated_GFLOPs": flops / 1e9
        }
