import os
import random
from typing import Optional

import numpy as np
import torch

from utils.logger import get_model_logger

logger = get_model_logger("SeedUtils")

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, configures cuDNN to be deterministic, which might
                              impact performance but guarantees reproducibility.
    """
    logger.info(f"Setting random seed to {seed} (Deterministic: {deterministic})")
    
    # Python built-in random module
    random.seed(seed)
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8.0, setting this to True makes operations deterministic
        torch.use_deterministic_algorithms(False)  # Set to True if strict determinism is required (can crash some ops)
        logger.info("Configured PyTorch backends for deterministic execution (cudnn.benchmark=False).")

