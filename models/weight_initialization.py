import torch
import torch.nn as nn

from utils.logger import get_model_logger

logger = get_model_logger("WeightInit")

def initialize_weights(module: nn.Module, method: str = "xavier_uniform") -> None:
    """
    Applies the specified weight initialization method to the module and all its submodules recursively.

    Args:
        module (nn.Module): The PyTorch module to initialize.
        method (str): The initialization method name. Valid options are:
                      'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 
                      'kaiming_normal', 'orthogonal', 'normal'.
    """
    method = method.lower()
    
    for name, m in module.named_modules():
        # Initialize Linear and Convolutional layers
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif method == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown weight initialization method: {method}")
                
            # Initialize bias if present
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        # Initialize Batch/Layer/Group Normalization layers
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        # Initialize Embeddings
        elif isinstance(m, nn.Embedding):
            if method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                nn.init.uniform_(m.weight, -0.1, 0.1)
                
    logger.info(f"Applied '{method}' initialization to all layers.")

