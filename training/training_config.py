from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """
    Configuration parameters for the Phase 3 Training Framework.
    """
    # 1. Core Training
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # 2. Optimization
    optimizer: str = "AdamW" # AdamW, Lion, SGD, RAdam, Adan
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    
    # 3. Learning Rate Scheduling
    scheduler: str = "CosineAnnealing" # CosineAnnealing, OneCycleLR, ReduceLROnPlateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 4. Multi-Task & Loss Strategy
    adaptive_loss_strategy: str = "UncertaintyWeighting" # UncertaintyWeighting, GradNorm, DWA
    physics_penalty_weight: float = 1.0
    
    # 5. Hardware & Precision
    device: str = "auto"
    mixed_precision: bool = True
    multi_gpu: bool = True
    
    # 6. Checkpointing & Callbacks
    checkpoint_dir: str = "checkpoints/"
    experiment_dir: str = "experiments/"
    tensorboard_dir: str = ""    # "" = auto from env_config
    export_dir: str = ""         # "" = auto from env_config
    enable_tensorboard: bool = True
    export_torchscript: bool = True
    export_onnx: bool = True
    save_top_k: int = 3
    early_stopping_patience: int = 15
    
    # 7. Curriculum Learning
    enable_curriculum: bool = True
    curriculum_advance_patience: int = 5 # Epochs without improvement before advancing stage

    # 8. Performance Optimization Framework
    # Execution mode: "development" | "production" | "benchmark"
    #   development - all debug/validation/assertions enabled
    #   production  - only forward+loss+backward+optimizer (fastest)
    #   benchmark   - like production + per-section timing probes
    execution_mode: str = "production"

    # torch.compile (PyTorch 2.x) — set True for potential speedup.
    # Automatically disabled if torch.compile is unavailable or fails.
    use_torch_compile: bool = False

    # Show tqdm progress bar during training
    enable_progress_bar: bool = True

    # DataLoader workers. 0 = auto-detect (min(4, cpu_count)).
    # Set explicitly to override auto-detection.
    num_dataloader_workers: int = 0
