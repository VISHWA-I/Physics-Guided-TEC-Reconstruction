from typing import Tuple

import torch

from utils.logger import get_model_logger

logger = get_model_logger("TensorUtils")

def check_finite(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Checks if a tensor contains any NaN or infinite values.

    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): Name of the tensor for logging purposes.

    Returns:
        bool: True if all values are finite, False otherwise.
    """
    if not torch.isfinite(tensor).all():
        logger.error(f"Tensor '{name}' contains NaN or infinite values.")
        return False
    return True

def validate_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor") -> bool:
    """
    Validates that a tensor matches an expected shape.
    Use -1 in expected_shape to indicate that any size is acceptable for that dimension (e.g., batch size).

    Args:
        tensor (torch.Tensor): The tensor to validate.
        expected_shape (Tuple[int, ...]): The expected shape.
        name (str): Name of the tensor for logging purposes.

    Returns:
        bool: True if the shape matches, False otherwise.
    """
    if len(tensor.shape) != len(expected_shape):
        logger.error(f"Tensor '{name}' has {len(tensor.shape)} dimensions, expected {len(expected_shape)}.")
        return False
        
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1 and actual != expected:
            logger.error(f"Tensor '{name}' dim {i} has size {actual}, expected {expected}.")
            return False
            
    return True

def get_memory_format(tensor: torch.Tensor) -> str:
    """
    Gets the memory format of the tensor (Contiguous, Channels Last, etc.).

    Args:
        tensor (torch.Tensor): The tensor.

    Returns:
        str: Memory format description.
    """
    if tensor.is_contiguous():
        return "Contiguous"
    elif tensor.is_contiguous(memory_format=torch.channels_last):
        return "Channels Last"
    return "Non-contiguous"

