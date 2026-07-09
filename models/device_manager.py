import torch
import torch.nn as nn

from utils.logger import get_model_logger

logger = get_model_logger("DeviceManager")

class DeviceManager:
    """
    Manages detection and assignment of computation devices (CPU, CUDA, MPS).
    """
    
    @staticmethod
    def get_device(requested_device: str = "auto") -> torch.device:
        """
        Determines the optimal device for computation.

        Args:
            requested_device (str): "auto", "cpu", "cuda", or "mps".

        Returns:
            torch.device: The selected PyTorch device.
        """
        req = requested_device.lower()
        
        if req == "cpu":
            device = torch.device("cpu")
        elif req == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif req == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif req == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            logger.warning(f"Requested device '{requested_device}' not fully supported or available. Falling back to CPU.")
            device = torch.device("cpu")
            
        logger.info(f"Device selected: {device}")
        return device

    @staticmethod
    def move_to_device(obj: torch.Tensor | nn.Module | dict | list, device: torch.device) -> torch.Tensor | nn.Module | dict | list:
        """
        Recursively moves a tensor, module, or collection of tensors to the specified device.

        Args:
            obj: The object to move (Tensor, Module, list, dict).
            device (torch.device): The target device.

        Returns:
            The object moved to the device.
        """
        if isinstance(obj, torch.Tensor) or isinstance(obj, nn.Module):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: DeviceManager.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DeviceManager.move_to_device(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(DeviceManager.move_to_device(v, device) for v in obj)
        else:
            return obj

