import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.logger import get_model_logger

logger = get_model_logger("ModelConfig")

@dataclass
class ModelConfig:
    """
    Configuration for the Multi-Branch Mamba-TKAN Network.
    Stores and validates all structural and training parameters.
    """
    
    # Architecture
    window_size: int = 24
    embedding_dimension: int = 64
    hidden_dimension: int = 128
    dropout: float = 0.1
    activation: str = "silu"
    number_of_blocks: int = 4
    normalization: str = "rmsnorm"
    
    # Mamba specifics
    state_expansion_factor: int = 16
    conv_kernel_size: int = 4
    expand_factor: int = 2
    use_gradient_checkpointing: bool = False
    
    # Temporal Embedding
    te_max_sequence_length: str = "auto"
    te_dynamic_resize: bool = True
    te_position_embedding: str = "learned"
    te_physics_embedding: bool = True
    te_multi_resolution: bool = True
    
    # Inputs
    input_features: int = 19
    temporal_features: list[str] = field(default_factory=list)
    physics_features: list[str] = field(default_factory=list)
    geo_features: list[str] = field(default_factory=list)
    storm_features: list[str] = field(default_factory=list)
    
    # Attention Config
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Decoder & Memory Config
    memory_slots: int = 64
    memory_heads: int = 4
    top_k_retrieval: int = 8
    tkan_branches: int = 3
    kan_width: int = 64
    kan_depth: int = 3
    
    # Prediction Heads Config
    enable_auxiliary: bool = True
    mc_dropout_samples: int = 10
    geometry_features_dim: int = 2
    output_dim: int = 1
    
    # Training
    batch_size: int = 32
    device: str = "auto"
    mixed_precision: bool = True
    random_seed: int = 42
    weight_initialization_method: str = "xavier_uniform"
    
    # Paths
    save_directory: str = "weights/"
    checkpoint_directory: str = "checkpoints/"
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ModelConfig":
        """
        Loads configuration from a YAML file.
        
        Args:
            yaml_path (str | Path): Path to the YAML configuration file.
            
        Returns:
            ModelConfig: Parsed configuration object.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.error(f"Configuration file {yaml_path} not found.")
            raise FileNotFoundError(f"Configuration file {yaml_path} not found.")
            
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        arch = data.get("architecture", {})
        inputs = data.get("inputs", {})
        train = data.get("training", {})
        paths = data.get("paths", {})
        
        config = cls(
            window_size=arch.get("window_size", 24),
            embedding_dimension=arch.get("embedding_dimension", 64),
            hidden_dimension=arch.get("hidden_dimension", 128),
            dropout=arch.get("dropout", 0.1),
            activation=arch.get("activation", "silu"),
            number_of_blocks=arch.get("number_of_blocks", 4),
            normalization=arch.get("normalization", "rmsnorm"),
            
            state_expansion_factor=arch.get("mamba", {}).get("state_expansion_factor", 16),
            conv_kernel_size=arch.get("mamba", {}).get("conv_kernel_size", 4),
            expand_factor=arch.get("mamba", {}).get("expand_factor", 2),
            use_gradient_checkpointing=arch.get("mamba", {}).get("use_gradient_checkpointing", False),
            
            te_max_sequence_length=arch.get("temporal_embedding", {}).get("max_sequence_length", "auto"),
            te_dynamic_resize=arch.get("temporal_embedding", {}).get("dynamic_resize", True),
            te_position_embedding=arch.get("temporal_embedding", {}).get("position_embedding", "learned"),
            te_physics_embedding=arch.get("temporal_embedding", {}).get("physics_embedding", True),
            te_multi_resolution=arch.get("temporal_embedding", {}).get("multi_resolution", True),
            
            input_features=inputs.get("input_features", 19),
            temporal_features=inputs.get("temporal_features", []),
            physics_features=inputs.get("physics_features", []),
            geo_features=inputs.get("geo_features", []),
            storm_features=inputs.get("storm_features", []),
            
            num_attention_heads=arch.get("attention", {}).get("num_heads", 4),
            attention_dropout=arch.get("attention", {}).get("dropout", 0.1),
            
            memory_slots=arch.get("decoder", {}).get("memory_slots", 64),
            memory_heads=arch.get("decoder", {}).get("memory_heads", 4),
            top_k_retrieval=arch.get("decoder", {}).get("top_k_retrieval", 8),
            tkan_branches=arch.get("decoder", {}).get("tkan_branches", 3),
            kan_width=arch.get("decoder", {}).get("kan_width", 64),
            kan_depth=arch.get("decoder", {}).get("kan_depth", 3),
            
            enable_auxiliary=arch.get("heads", {}).get("enable_auxiliary", True),
            mc_dropout_samples=arch.get("heads", {}).get("mc_dropout_samples", 10),
            geometry_features_dim=arch.get("heads", {}).get("geometry_features_dim", 2),
            output_dim=arch.get("heads", {}).get("output_dim", 1),
            
            batch_size=train.get("batch_size", 32),
            device=train.get("device", "auto"),
            mixed_precision=train.get("mixed_precision", True),
            random_seed=train.get("random_seed", 42),
            weight_initialization_method=train.get("weight_initialization_method", "xavier_uniform"),
            
            save_directory=paths.get("save_directory", "weights/"),
            checkpoint_directory=paths.get("checkpoint_directory", "checkpoints/"),
        )
        
        config.validate()
        return config

    def validate(self) -> None:
        """
        Validates the configuration parameters.
        Raises ValueError if any configuration is invalid.
        """
        if self.window_size <= 0:
            raise ValueError(f"window_size must be strictly positive, got {self.window_size}")
        if self.embedding_dimension <= 0:
            raise ValueError(f"embedding_dimension must be strictly positive, got {self.embedding_dimension}")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.input_features <= 0:
            raise ValueError(f"input_features must be strictly positive, got {self.input_features}")
            
        logger.info("ModelConfig validated successfully.")

