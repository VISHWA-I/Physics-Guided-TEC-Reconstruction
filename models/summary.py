from typing import Dict, Tuple, Any

import torch
import torch.nn as nn

def compute_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Computes summary metrics for a PyTorch model.
    
    Args:
        model (nn.Module): The model to analyze.
        
    Returns:
        Dict[str, Any]: Dictionary containing total_parameters, trainable_parameters,
                        and memory estimates in megabytes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory footprint for parameters (assuming float32 = 4 bytes)
    param_memory_mb = (total_params * 4) / (1024 ** 2)
    
    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameters_memory_mb": round(param_memory_mb, 2),
    }
    
    return summary

def print_layer_summary(model: nn.Module) -> None:
    """
    Prints a layer-wise summary of the model parameters.
    
    Args:
        model (nn.Module): The PyTorch model.
    """
    print("-" * 80)
    print(f"{'Layer Name':<45} | {'Type':<15} | {'Params':<10} | {'Trainable'}")
    print("-" * 80)
    
    for name, module in model.named_modules():
        # Only print leaf modules with parameters
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                mod_type = module.__class__.__name__
                print(f"{name:<45} | {mod_type:<15} | {params:<10} | {trainable > 0}")
                
    print("-" * 80)
    summary = compute_model_summary(model)
    print(f"Total Parameters:      {summary['total_parameters']:,}")
    print(f"Trainable Parameters:  {summary['trainable_parameters']:,}")
    print(f"Estimated Model Size:  {summary['parameters_memory_mb']} MB")
    print("-" * 80)
