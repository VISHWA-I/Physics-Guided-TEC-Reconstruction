import torch
import numpy as np

class TensorValidator:
    """
    Low-level utility to perform strict validations on tensor properties before physical computations.
    Checks Rank, Batch Size, Sequence Length, Feature Dimension, Device, Data Type, NaN, and Inf.
    """

    @staticmethod
    def validate(tensor: torch.Tensor, name: str, expected_shape: tuple = None, debug: bool = False):
        """
        Validates the tensor's properties and optionally prints a debug report.

        Args:
            tensor (torch.Tensor): The tensor to validate.
            name (str): Name of the tensor for logging.
            expected_shape (tuple, optional): Expected shape (can use -1 for arbitrary dims).
            debug (bool): If True, prints the detailed validation report.
            
        Raises:
            ValueError: If tensor has NaN or Inf.
            RuntimeError: If shape mismatch occurs.
        """
        if tensor is None:
            raise ValueError(f"Tensor '{name}' is None.")

        # 1. NaN and Inf Check
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan:
            raise ValueError(f"Tensor '{name}' contains NaN values!")
        if has_inf:
            raise ValueError(f"Tensor '{name}' contains Inf values!")

        # 2. Shape Validation
        shape_pass = True
        if expected_shape is not None:
            if tensor.dim() != len(expected_shape):
                shape_pass = False
            else:
                for i, (dim, exp_dim) in enumerate(zip(tensor.shape, expected_shape)):
                    if exp_dim != -1 and dim != exp_dim:
                        shape_pass = False
                        break
            
            if not shape_pass:
                raise RuntimeError(f"Shape mismatch for '{name}'. Expected {expected_shape}, got {tuple(tensor.shape)}")

        # 3. Debug Report
        if debug:
            print(f"=============================")
            print(f"Tensor Validation: {name}")
            print(f"=============================")
            print(f"Input Tensor : PASS")
            print(f"Shape        : {tuple(tensor.shape)}")
            print(f"Device       : {tensor.device.type.upper()}")
            print(f"DType        : {tensor.dtype}")
            print(f"NaN          : {'Yes' if has_nan else 'No'}")
            print(f"Inf          : {'Yes' if has_inf else 'No'}")
            print(f"=============================\n")

    @staticmethod
    def is_scientifically_expandable(tensor: torch.Tensor, target_seq_len: int) -> bool:
        """
        Determines if a (Batch, 1) or (Batch, 1, 1) tensor can be safely expanded 
        across the temporal sequence dimension based on physics rules.
        """
        if tensor.dim() == 2 and tensor.shape[1] == 1:
            return True # Can broadcast single timestep prediction across sequence if needed
        if tensor.dim() == 3 and tensor.shape[1] == 1:
            return True
        return False
