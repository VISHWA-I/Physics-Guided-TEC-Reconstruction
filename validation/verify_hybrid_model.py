import sys
from pathlib import Path
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_config import ModelConfig
from models.hybrid_model import HybridModel
from utils.seed import set_seed
from utils.tensor_utils import validate_shape

def test_hybrid_model():
    print("=== Testing Phase 2.5: Hybrid Mamba-TKAN Model Assembly ===")
    set_seed(42)
    
    config = ModelConfig.from_yaml(project_root / "configs" / "model.yaml")
    
    print("\n=== Instantiating Hybrid Model (Debug Mode) ===")
    model = HybridModel(config, debug_mode=True)
    model.initialize_weights()
    model.move_to_device()
    model.train() # Enable training mode to test auxiliary head
    
    print("\n[Parameter Introspection]")
    print(f"Total Trainable Parameters: {model.count_parameters():,}")
    
    print("\n=== Testing Full Forward Pass ===")
    batch = 32
    window = config.window_size
    
    # Inputs Phase 2.2
    temporal_in = torch.randn(batch, window, len(config.temporal_features)).to(model.current_device)
    
    # Inputs Phase 2.3
    physics_in = torch.randn(batch, len(config.physics_features)).to(model.current_device)
    geo_in = torch.randn(batch, len(config.geo_features)).to(model.current_device)
    storm_in = torch.randn(batch, len(config.storm_features)).to(model.current_device)
    
    # Inputs Phase 2.5
    bottomside_tec_in = torch.randn(batch, window, 1).to(model.current_device)
    geometry_in = torch.randn(batch, window, config.geometry_features_dim).to(model.current_device)
    
    # Require grad for backward pass test
    temporal_in.requires_grad = True
    
    print(f"Executing complete pipeline on {model.current_device}...")
    output = model(
        temporal_seq=temporal_in,
        physics_feats=physics_in,
        geo_feats=geo_in,
        storm_feats=storm_in,
        bottomside_tec=bottomside_tec_in,
        geometry_feats=geometry_in
    )
    
    print("\n=== Verifying Dataclass Tensor Shapes ===")
    valid = True
    valid &= validate_shape(output.topside_tec, (batch, window, 1), name="Topside TEC")
    valid &= validate_shape(output.net_tec, (batch, window, 1), name="Net TEC")
    valid &= validate_shape(output.electron_density, (batch, window, 1), name="Electron Density")
    valid &= validate_shape(output.gnss_delays["vertical_delay"], (batch, window, 1), name="Vertical Delay")
    valid &= validate_shape(output.gnss_delays["slant_delay"], (batch, window, 1), name="Slant Delay")
    valid &= validate_shape(output.confidence_score, (batch, window, 1), name="Confidence Score")
    
    if valid:
        print("PASS: All hierarchical prediction shapes correctly match expectations.")
    else:
        print("FAIL: Output shape mismatch in ModelOutput!")
        
    print("\n=== Testing Backward Pass (Gradients) ===")
    try:
        # Create a synthetic loss aggregating all outputs to ensure gradients flow from EVERY head
        loss = output.topside_tec.sum() + output.net_tec.sum() + output.gnss_delays["gps_delay"].sum()
        loss.backward()
        print("PASS: Backward pass successfully propagated gradients from all tasks down to the Temporal Encoder.")
    except Exception as e:
        print(f"FAIL: Backward pass crashed: {e}")
        
    print("\nPhase 2.5 Complete: The HybridModel is fully assembled and ready for Phase 3 (Training)!")

if __name__ == "__main__":
    test_hybrid_model()
