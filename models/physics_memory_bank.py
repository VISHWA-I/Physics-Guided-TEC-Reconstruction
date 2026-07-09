import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.logger import get_model_logger

logger = get_model_logger("PhysicsMemoryBank")

class PhysicsMemoryBank(nn.Module):
    """
    Stores distinct categories of learnable physics representations (Quiet, Storm, Latitude, Seasonal, Prototype).
    These act as persistent memory slots that the decoder can query to recall how it should reconstruct TEC 
    under specific historical conditions.

    Optimization
    ------------
    ``get_all_memories()`` previously called ``torch.cat`` on 5 parameter tensors every forward pass.
    A dirty-flag cache is now used: the concatenated pool is only recomputed when at least one
    parameter was updated since the last call.  The flag is set via parameter post-accumulation hooks
    registered at init time, ensuring correctness during both forward and backward passes.
    """

    def __init__(self, num_slots: int, embedding_dim: int):
        """
        Initializes the memory banks.

        Args:
            num_slots (int): Number of memory slots per domain.
            embedding_dim (int): The dimension of each memory slot.
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.embedding_dim = embedding_dim
        
        # 1. Quiet-Time Memory: Typical diurnal variations
        self.quiet_memory = nn.Parameter(torch.randn(num_slots, embedding_dim))
        
        # 2. Storm-Time Memory: High geomagnetic disturbance states
        self.storm_memory = nn.Parameter(torch.randn(num_slots, embedding_dim))
        
        # 3. Latitude Memory: Equatorial vs Mid-latitude vs Polar physics
        self.latitude_memory = nn.Parameter(torch.randn(num_slots, embedding_dim))
        
        # 4. Seasonal Memory: Summer vs Winter anomalies
        self.seasonal_memory = nn.Parameter(torch.randn(num_slots, embedding_dim))
        
        # 5. Prototype Memory: General purpose routing and catch-all representations
        self.prototype_memory = nn.Parameter(torch.randn(num_slots, embedding_dim))
        
        # Initialize memory slots using orthogonal initialization for better feature distinctiveness
        self._init_memory(self.quiet_memory)
        self._init_memory(self.storm_memory)
        self._init_memory(self.latitude_memory)
        self._init_memory(self.seasonal_memory)
        self._init_memory(self.prototype_memory)
        
        # --- Dirty-flag cache ---
        # _cached_pool: the last concatenated memory tensor, None until first call
        # _is_dirty: True whenever any parameter may have been updated
        self._cached_pool: Optional[torch.Tensor] = None
        self._is_dirty: bool = True

        # Register hooks on all five memory parameters so that after each
        # gradient accumulation step (optimizer.step) the cache is invalidated.
        self._register_dirty_hooks()

        logger.info(f"Initialized 5 Memory Banks with {num_slots} slots of dim {embedding_dim}.")

    def _init_memory(self, memory: nn.Parameter) -> None:
        """Applies orthogonal initialization to a memory bank matrix."""
        nn.init.orthogonal_(memory.data)

    def _register_dirty_hooks(self) -> None:
        """Register post-accumulate gradient hooks to mark cache as dirty after optimizer step."""
        _params = [
            self.quiet_memory,
            self.storm_memory,
            self.latitude_memory,
            self.seasonal_memory,
            self.prototype_memory,
        ]
        for param in _params:
            # post_accumulate_grad_hook fires after gradients have been accumulated
            # (i.e., after optimizer.step has been applied to this parameter).
            try:
                param.register_post_accumulate_grad_hook(self._mark_dirty)
            except AttributeError:
                # PyTorch < 2.1 does not have register_post_accumulate_grad_hook.
                # Fall back to a plain grad hook which fires on every backward pass.
                param.register_hook(lambda _grad: self._mark_dirty(None))

    def _mark_dirty(self, _param) -> None:
        """Callback: invalidate the cached memory pool."""
        self._is_dirty = True

    def get_all_memories(self) -> torch.Tensor:
        """
        Returns the concatenated memory pool (5 * num_slots, embedding_dim).

        The result is cached and only recomputed when a parameter was updated
        since the last call, eliminating redundant ``torch.cat`` calls during
        the forward pass when the model is in eval mode or when the same
        batch triggers multiple calls.

        Returns:
            torch.Tensor: Shape (5 * num_slots, embedding_dim)
        """
        if self._is_dirty or self._cached_pool is None:
            self._cached_pool = torch.cat([
                self.quiet_memory,
                self.storm_memory,
                self.latitude_memory,
                self.seasonal_memory,
                self.prototype_memory,
            ], dim=0)
            self._is_dirty = False

        return self._cached_pool
