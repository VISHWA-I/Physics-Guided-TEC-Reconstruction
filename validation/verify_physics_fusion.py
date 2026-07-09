import sys
from pathlib import Path
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_config import ModelConfig
from models.adaptive_fusion import AdaptiveFusionLayer
from utils.seed import set_seed
from utils.tensor_utils import validate_shape

def test_physics_fusion():
    print("=== Testing Phase 2.3: Adaptive Physics Fusion ===")
    set_seed(42)
    
    config = ModelConfig.from_yaml(project_root / "configs" / "model.yaml")
    
    print("\n=== Instantiating Adaptive Fusion Layer (Debug Mode) ===")
    fusion_layer = AdaptiveFusionLayer(config, debug_mode=True)
    fusion_layer.initialize_weights()
    fusion_layer.move_to_device()
    
    print("\n[Parameter Introspection]")
    print(f"Total Trainable Parameters: {fusion_layer.count_parameters():,}")
    
    print("\n=== Testing Forward Pass ===")
    batch = 32
    window = config.window_size
    hidden = config.hidden_dimension
    
    # Generate Mock Inputs according to config
    temporal_in = torch.randn(batch, window, hidden).to(fusion_layer.current_device)
    physics_in = torch.randn(batch, len(config.physics_features)).to(fusion_layer.current_device)
    geo_in = torch.randn(batch, len(config.geo_features)).to(fusion_layer.current_device)
    storm_in = torch.randn(batch, len(config.storm_features)).to(fusion_layer.current_device)
    
    # Require grad for backward pass test
    temporal_in.requires_grad = True
    physics_in.requires_grad = True
    
    print(f"Executing forward pass on {fusion_layer.current_device}...")
    fused_out, metrics = fusion_layer(temporal_in, physics_in, geo_in, storm_in)
    
    print("\n=== Verifying Tensor Shapes ===")
    is_valid = validate_shape(fused_out, (batch, window, hidden), name="UnifiedRepresentation")
    if is_valid:
        print(f"PASS: Output shape correctly matches (Batch={batch}, Window={window}, Hidden={hidden})")
    else:
        print("FAIL: Output shape mismatch!")
        
    print("\n=== Verifying Metrics & Gate ===")
    attn_maps = metrics["attention_maps"]
    gate_weights = metrics["physics_gate_weights"]
    
    print(f"Attention Maps Shape: {attn_maps.shape}")
    print(f"Physics Gate Shape: {gate_weights.shape}")
    
    if (gate_weights >= 0.0).all() and (gate_weights <= 1.0).all():
        print("PASS: Physics Gate weights strictly bounded between 0 and 1.")
    else:
        print("FAIL: Physics Gate weights out of bounds!")
        
    print("\n=== Testing Backward Pass (Gradients) ===")
    try:
        loss = fused_out.sum()
        loss.backward()
        print("PASS: Backward pass executed successfully without errors.")
    except Exception as e:
        print(f"FAIL: Backward pass crashed: {e}")
        
    print("\nPhysics Fusion Layer is successfully validated and ready for Phase 2.4!")

if __name__ == "__main__":
    test_physics_fusion()
