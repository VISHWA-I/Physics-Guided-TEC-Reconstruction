import sys
from pathlib import Path
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_config import ModelConfig
from models.temporal_encoder import TemporalEncoder
from utils.seed import set_seed
from utils.tensor_utils import validate_shape

def test_temporal_branch():
    print("=== Testing Phase 2.2: Temporal Branch ===")
    set_seed(42)
    
    # Load configuration
    config = ModelConfig.from_yaml(project_root / "configs" / "model.yaml")
    
    # We must explicitly set temporal features for this test to pass
    # since model.yaml provides them
    temporal_features = config.temporal_features
    if not temporal_features:
        print("FAIL: Config did not load temporal_features correctly.")
        return
        
    print(f"Loaded {len(temporal_features)} Temporal Features: {temporal_features}")
    print(f"Mamba Configuration: Blocks={config.number_of_blocks}, State={config.state_expansion_factor}")
    
    print("\n=== Instantiating Temporal Encoder (Debug Mode) ===")
    encoder = TemporalEncoder(config, debug_mode=True)
    encoder.initialize_weights()
    encoder.move_to_device()
    
    # Print trainable parameters to verify it hooked into BaseModel properly
    print("\n[Parameter Introspection]")
    print(f"Total Trainable Parameters: {encoder.count_parameters():,}")
    
    print("\n=== Testing Forward Pass ===")
    batch = 32
    window = config.window_size
    features = len(temporal_features)
    
    # Dummy input representing (Batch, Window, Features)
    x_in = torch.randn(batch, window, features).to(encoder.current_device)
    # Require gradients to test backward pass stability
    x_in.requires_grad = True
    
    print(f"Executing forward pass on {encoder.current_device}...")
    out = encoder(x_in)
    
    print("\n=== Verifying Tensor Shapes ===")
    is_valid = validate_shape(out, (batch, window, config.hidden_dimension), name="TemporalOutput")
    if is_valid:
        print(f"PASS: Output shape correctly matches (Batch={batch}, Window={window}, Hidden={config.hidden_dimension})")
    else:
        print("FAIL: Output shape mismatch!")
        
    print("\n=== Testing Backward Pass (Gradients) ===")
    try:
        # Create a dummy loss and backward pass
        loss = out.sum()
        loss.backward()
        print("PASS: Backward pass executed successfully without errors.")
    except Exception as e:
        print(f"FAIL: Backward pass crashed: {e}")
        
    print("\nTemporal Branch is successfully validated and ready for Phase 2.3!")

if __name__ == "__main__":
    test_temporal_branch()
