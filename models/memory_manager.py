import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.physics_memory_bank import PhysicsMemoryBank
from utils.logger import get_model_logger

logger = get_model_logger("MemoryManager")

class MemoryManager(nn.Module):
    """
    Manages the lifecycle, retrieval, and structural organization of the PhysicsMemoryBank.
    """

    def __init__(self, num_slots: int, embedding_dim: int):
        """
        Initializes the Memory Manager.

        Args:
            num_slots (int): Number of memory slots per domain.
            embedding_dim (int): Dimension of each slot.
        """
        super().__init__()
        self.bank = PhysicsMemoryBank(num_slots, embedding_dim)
        self.total_slots = num_slots * 5 # 5 domains
        self.embedding_dim = embedding_dim
        
    def get_memory_pool(self) -> torch.Tensor:
        """
        Retrieves the complete concatenated memory pool.

        Returns:
            torch.Tensor: The full memory matrix of shape (Total_Slots, Embedding_Dim)
        """
        return self.bank.get_all_memories()

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """
        Extracts the explicit state of the memory banks (useful for visualization or saving).
        """
        return {
            "quiet": self.bank.quiet_memory.detach().cpu(),
            "storm": self.bank.storm_memory.detach().cpu(),
            "latitude": self.bank.latitude_memory.detach().cpu(),
            "seasonal": self.bank.seasonal_memory.detach().cpu(),
            "prototype": self.bank.prototype_memory.detach().cpu()
        }
