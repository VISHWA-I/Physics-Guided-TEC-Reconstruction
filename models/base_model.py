from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from models.model_config import ModelConfig
from models.device_manager import DeviceManager
from models.weight_initialization import initialize_weights
from models.summary import compute_model_summary, print_layer_summary
from utils.logger import get_model_logger

logger = get_model_logger("BaseModel")

class BaseModel(nn.Module):
    """
    Base class for all models in the framework.
    Provides common functionality for saving, loading, weight initialization,
    device management, and parameter introspection.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the BaseModel.

        Args:
            config (ModelConfig): The complete model configuration.
        """
        super().__init__()
        self.config = config
        self._device = torch.device("cpu")

    def save_model(self, path: str | Path) -> None:
        """
        Saves the model state dictionary to the specified path.

        Args:
            path (str | Path): Path to save the model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str | Path, map_location: Any = None) -> None:
        """
        Loads the model state dictionary from the specified path.

        Args:
            path (str | Path): Path to load the model from.
            map_location: Optional map_location for torch.load.
        """
        load_path = Path(path)
        if not load_path.exists():
            logger.error(f"Cannot load model, path does not exist: {load_path}")
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        if map_location is None:
            map_location = self._device

        state_dict = torch.load(load_path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"Model loaded from {load_path}")

    def initialize_weights(self) -> None:
        """
        Applies the weight initialization specified in the configuration.
        """
        method = self.config.weight_initialization_method
        initialize_weights(self, method)
        logger.info(f"Model weights initialized using {method}")

    def move_to_device(self) -> None:
        """
        Moves the model to the device specified in the configuration.
        """
        target_device = DeviceManager.get_device(self.config.device)
        self.to(target_device)
        self._device = target_device
        logger.info(f"Model moved to device: {target_device}")

    @property
    def current_device(self) -> torch.device:
        """Returns the current device of the model."""
        return self._device

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        summary = compute_model_summary(self)
        return summary["trainable_parameters"]

    def model_summary(self) -> None:
        """
        Prints a detailed layer-wise summary of the model.
        """
        print_layer_summary(self)

    def print_trainable_parameters(self) -> None:
        """
        Prints the names and parameter counts of all trainable layers.
        """
        print("Trainable Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,}")

    def freeze_layers(self, keywords: list[str] = None) -> None:
        """
        Freezes layers containing any of the specified keywords in their name.
        If keywords is None, freezes all layers.

        Args:
            keywords (list[str], optional): List of layer name substrings to freeze.
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if keywords is None or any(kw in name for kw in keywords):
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Froze {frozen_count} parameters.")

    def unfreeze_layers(self, keywords: list[str] = None) -> None:
        """
        Unfreezes layers containing any of the specified keywords in their name.
        If keywords is None, unfreezes all layers.

        Args:
            keywords (list[str], optional): List of layer name substrings to unfreeze.
        """
        unfrozen_count = 0
        for name, param in self.named_parameters():
            if keywords is None or any(kw in name for kw in keywords):
                param.requires_grad = True
                unfrozen_count += 1
        logger.info(f"Unfroze {unfrozen_count} parameters.")

    def export_configuration(self) -> dict:
        """
        Exports the model configuration to a dictionary format.

        Returns:
            dict: The configuration dictionary.
        """
        from dataclasses import asdict
        return asdict(self.config)

    def forward(self, *args, **kwargs):
        """
        Forward pass defining the model's computation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward pass.")

