import torch
import torch.nn as nn

from utils.logger import get_model_logger

logger = get_model_logger("ModelUtils")

def check_gradient_norms(model: nn.Module) -> float:
    """
    Computes the overall L2 norm of the gradients of all trainable parameters.
    Useful for detecting exploding/vanishing gradients.

    Args:
        model (nn.Module): The model.

    Returns:
        float: The L2 norm of the gradients. Returns 0.0 if no gradients are found.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def find_unused_parameters(model: nn.Module) -> list[str]:
    """
    Identifies parameters that require gradients but currently have None for their grad.
    Useful during DistributedDataParallel (DDP) debugging or complex branch architectures.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        list[str]: A list of parameter names that do not have gradients.
    """
    unused = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            unused.append(name)
            
    if unused:
        logger.warning(f"Found {len(unused)} unused parameters (no gradients computed).")
    
    return unused

def is_mixed_precision_ready(model: nn.Module) -> bool:
    """
    Validates that a model does not contain problematic layer configurations
    that commonly crash under Float16/Bfloat16 mixed precision.
    
    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if it passes basic heuristic checks.
    """
    # Just a simple heuristic implementation for the skeleton
    ready = True
    for name, module in model.named_modules():
        # Example heuristic: extremely large linear layers can underflow in FP16
        if isinstance(module, nn.Linear):
            if module.in_features > 100000 or module.out_features > 100000:
                logger.warning(f"Layer {name} is unusually large, might underflow in mixed precision.")
                ready = False
    return ready

