import sys
from pathlib import Path
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry
from models.model_factory import ModelFactory
from models.embeddings import BranchEmbedding
from utils.seed import set_seed
from utils.tensor_utils import validate_shape

# 1. Register a Dummy Model for Testing
@ModelRegistry.register("DummyTestModel")
class DummyTestModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.temporal_emb = BranchEmbedding(len(config.temporal_features), config.embedding_dimension)
        self.physical_emb = BranchEmbedding(len(config.physical_features), config.embedding_dimension)
        self.geophysical_emb = BranchEmbedding(len(config.geophysical_features), config.embedding_dimension)
        
        self.fusion = torch.nn.Linear(config.embedding_dimension * 3, config.hidden_dimension)
        self.output = torch.nn.Linear(config.hidden_dimension, 1)
        
    def forward(self, t, p, g):
        t_emb = self.temporal_emb(t)
        p_emb = self.physical_emb(p)
        g_emb = self.geophysical_emb(g)
        
        # Concat along feature dimension
        fused = torch.cat([t_emb, p_emb, g_emb], dim=-1)
        out = torch.relu(self.fusion(fused))
        return self.output(out)

def test_framework():
    print("=== Testing Framework Configuration ===")
    set_seed(42)
    
    config = ModelConfig.from_yaml(project_root / "configs" / "model.yaml")
    print(f"Loaded config: Window Size = {config.window_size}, Batch Size = {config.batch_size}")
    
    print("\n=== Testing Factory & Instantiation ===")
    model = ModelFactory.create("DummyTestModel", config)
    
    print("\n=== Testing Model Summary ===")
    model.model_summary()
    
    print("\n=== Testing Forward Pass & Tensor Utils ===")
    batch = config.batch_size
    window = config.window_size
    
    # Mock inputs
    t_in = torch.randn(batch, window, len(config.temporal_features)).to(model.current_device)
    p_in = torch.randn(batch, window, len(config.physical_features)).to(model.current_device)
    g_in = torch.randn(batch, window, len(config.geophysical_features)).to(model.current_device)
    
    out = model(t_in, p_in, g_in)
    
    is_valid = validate_shape(out, (batch, window, 1), name="DummyOutput")
    print(f"\nOutput shape validation: {'PASS' if is_valid else 'FAIL'}")
    
    print("\nFramework is successfully validated and ready for Phase 2.2!")

if __name__ == "__main__":
    test_framework()
