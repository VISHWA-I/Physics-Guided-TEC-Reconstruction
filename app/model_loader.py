import torch
from pathlib import Path
from typing import Optional

from models.model_config import ModelConfig
from models.hybrid_model import HybridModel
from utils.logger import get_model_logger

logger = get_model_logger("ModelLoader")

class ModelLoader:
    """
    Handles instantiating the architecture and loading pre-trained weights safely.
    Automatically detects the optimal hardware backend (CUDA, MPS, CPU).
    """
    
    @staticmethod
    def load(config_path: str, checkpoint_path: Optional[str] = None) -> HybridModel:
        """
        Loads the Hybrid Mamba-TKAN model.
        """
        # Determine Device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Hardware Accelerator Detected: CUDA (NVIDIA GPU)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Hardware Accelerator Detected: MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("No Hardware Accelerator Detected: Defaulting to CPU")
            
        # Load Config
        logger.info(f"Loading configuration from {config_path}")
        config = ModelConfig.from_yaml(config_path)
        
        # Instantiate Architecture
        model = HybridModel(config)
        model.to(device)
        
        # Load Weights
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading weights from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            logger.info("Model weights loaded successfully.")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found. Proceeding with uninitialized weights.")
            
        model.eval()
        return model, device
