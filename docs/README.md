# Physics-Guided Multi-Branch Mamba–TKAN Network
## Phase 2.1: Deep Learning Framework Skeleton

This repository phase implements the core PyTorch deep learning framework that will support the Topside Ionosphere–Plasmasphere TEC Reconstruction project. It prepares the architectural skeleton required for advanced neural network models without directly implementing the specific model blocks (Mamba, TKAN, Attention, Fusion). 

The goal of Phase 2.1 is purely structural: providing reusable layers, dynamic instantiation, strict configurations, and standardized evaluation utilities.

### Project Structure
```text
project/
├── configs/
│   └── model.yaml                 # Central configuration for model architectures
├── docs/
│   └── README.md                  # This documentation
├── models/
│   ├── base_model.py              # Abstract BaseModel inherited by all models
│   ├── checkpoint_manager.py      # Saving and loading training state/metrics
│   ├── device_manager.py          # Auto-detection of CPU/CUDA/MPS
│   ├── embeddings.py              # Generic BranchEmbedding with normalization and projection
│   ├── model_config.py            # Dataclass strictly typed representation of model.yaml
│   ├── model_factory.py           # Instantiates models decoupling architecture from training loops
│   ├── model_registry.py          # Decorator-based registry for dynamic class lookup
│   ├── model_utils.py             # Advanced introspection (mixed precision checks, gradient norms)
│   ├── summary.py                 # Layer-wise parameter counting and memory estimates
│   └── weight_initialization.py   # Recursive Kaiming, Xavier, Orthogonal, Normal init
├── utils/
│   ├── logger.py                  # Standardized framework logger to logs/model.log
│   ├── seed.py                    # Deterministic seeds for NumPy, PyTorch, Python
│   └── tensor_utils.py            # Shape and finiteness validators
├── weights/                       # Destination for final trained model weights
├── checkpoints/                   # Destination for intermediate epoch checkpoints
├── logs/                          # Directory for execution logs
├── results/                       # Directory for evaluation outcomes
└── src/                           # Phase 1 Data pipeline (inherited)
```

### Purpose of Key Modules

#### The Registry & Factory Pattern (`model_registry.py`, `model_factory.py`)
To prevent massive `if/else` statements in the training loop and tight coupling between the trainer and the model architectures, this framework employs a Registry pattern. Any future model (like `HybridModel`, `MambaEncoder`, `TKAN`) will simply be decorated with `@ModelRegistry.register("model_name")`. 
The `ModelFactory` then takes a `ModelConfig` and a string name to return a fully instantiated, initialized, and device-ready model.

#### The `BaseModel` (`base_model.py`)
Every PyTorch model developed in Phase 2.2 and beyond MUST inherit from `BaseModel`. 
By doing so, the model instantly gains access to:
- `self.save_model(path)` and `self.load_model(path)`
- `self.initialize_weights()`
- `self.count_parameters()` and `self.model_summary()`
- `self.move_to_device()`
- Layer freezing and unfreezing utilities

#### Configuration (`model_config.py`)
The `ModelConfig` dataclass safely loads and validates all settings from `configs/model.yaml`. This ensures that attributes like `window_size` (24) and `input_features` (19) are strictly enforced at runtime, preventing silent shape mismatches down the line.

### How Future Phases Will Extend `BaseModel`

In Phase 2.2, when you implement the core network, you will follow this exact pattern:

```python
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry

@ModelRegistry.register("HybridTECReconstructor")
class HybridTECReconstructor(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Instantiate your custom blocks here
        # self.temporal_mamba = ...
        # self.tkan_decoder = ...
        
    def forward(self, x_temporal, x_physical, x_geophysical):
        # Implementation of your forward pass
        pass
```

Then, in your training script, instantiation becomes a one-liner:
```python
from models.model_factory import ModelFactory
from models.model_config import ModelConfig

config = ModelConfig.from_yaml("configs/model.yaml")
model = ModelFactory.create("HybridTECReconstructor", config)
```
