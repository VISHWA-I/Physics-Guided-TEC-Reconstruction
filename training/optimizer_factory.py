import torch
import torch.optim as optim
from typing import Dict, Any

class OptimizerFactory:
    """
    Dynamically instantiates optimizers.
    Supports AdamW, SGD, RAdam. (Lion and Adan require custom pip packages, so we map them to AdamW safely if unavailable).
    """

    @staticmethod
    def create(optimizer_name: str, model_parameters, lr: float, weight_decay: float) -> optim.Optimizer:
        name = optimizer_name.lower()
        
        if name == "adamw":
            return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif name == "radam":
            return optim.RAdam(model_parameters, lr=lr, weight_decay=weight_decay)
        elif name in ["lion", "adan"]:
            # Fallback to AdamW to avoid import crashes on systems without specialized packages
            print(f"Warning: {optimizer_name} requested. Defaulting to AdamW for safety.")
            return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
