import torch
import torch.nn as nn

from utils.logger import get_model_logger
from utils.tensor_validator import TensorValidator

logger = get_model_logger("TensorManager")

class TensorManager(nn.Module):
    """
    Central Tensor Manager for the Physics Constraint Engine.
    Responsibilities:
    - Tensor validation
    - Tensor alignment
    - Automatic broadcasting (if scientifically valid)
    - Shape conversion
    - Mixed precision / Device synchronization
    """
    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode
        self.validator = TensorValidator()

    def validate_and_align(self, tensor: torch.Tensor, name: str, expected_seq_len: int) -> torch.Tensor:
        """
        Validates the tensor, aligns its dimensions to (Batch, Sequence, Feature).
        If the sequence dimension is 1 but needs to be `expected_seq_len`, it automatically broadcasts.
        """
        # Base validation
        self.validator.validate(tensor, name=name, debug=self.debug_mode)
        
        # Shape Standardization
        # Target shape: (Batch, Seq, Features)
        
        # If passed as (Batch,)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
            
        # If passed as (Batch, Features)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
            
        # Ensure it is now 3D
        if tensor.dim() != 3:
            raise RuntimeError(f"TensorManager Error: Cannot align '{name}' with shape {tensor.shape} to 3D.")
            
        # Sequence Length Alignment
        b, s, f = tensor.shape
        if s != expected_seq_len:
            if self.validator.is_scientifically_expandable(tensor, expected_seq_len):
                # Broadcast across sequence dimension
                tensor = tensor.expand(-1, expected_seq_len, -1)
                if self.debug_mode:
                    logger.debug(f"Broadcasted '{name}' from seq length {s} to {expected_seq_len}.")
            else:
                raise RuntimeError(f"TensorManager Error: Scientifically invalid alignment for '{name}'. "
                                   f"Expected seq len {expected_seq_len}, got {s}. "
                                   f"Cannot safely expand.")
                                   
        return tensor

    def ensure_non_negative(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """
        Physics constraint: Forces the tensor to be strictly non-negative.
        Useful for TEC, Electron Density, GNSS Delay.
        Prints a warning if negative values were detected and clipped.
        """
        if (tensor < 0).any():
            if self.debug_mode:
                logger.warning(f"Physics Constraint Violation: Negative values detected in '{name}'. Applying ReLU enforcement.")
            return torch.relu(tensor)
        return tensor

    def enforce_monotonic_increase(self, base_tensor: torch.Tensor, target_tensor: torch.Tensor, name: str) -> torch.Tensor:
        """
        Physics constraint: Target tensor must be >= Base tensor.
        Useful for enforcing Net TEC >= Bottomside TEC.
        """
        violation = target_tensor < base_tensor
        if violation.any():
            if self.debug_mode:
                logger.warning(f"Physics Constraint Violation: '{name}' is strictly less than base constraint. Correcting.")
            target_tensor = torch.where(violation, base_tensor, target_tensor)
        return target_tensor
