import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_config import ModelConfig
from models.hybrid_model import HybridModel

def run_verification():
    print("=== Testing Physics Pipeline Tensor Shapes ===\n")
    
    config = ModelConfig.from_yaml("configs/model.yaml")
    
    model = HybridModel(config, debug_mode=False)
    
    test_sequences = [48, 96, 192]
    batch_size = 4
    
    for seq in test_sequences:
        print(f"Testing Sequence Length: {seq}")
        
        # Mock Inputs
        temporal_seq = torch.randn(batch_size, seq, len(config.temporal_features))
        physics_feats = torch.randn(batch_size, len(config.physics_features))
        geo_feats = torch.randn(batch_size, len(config.geo_features))
        storm_feats = torch.randn(batch_size, len(config.storm_features))
        
        # Standardized Bottomside TEC (Batch, Seq, 1)
        bottomside_tec = torch.randn(batch_size, seq, 1)
        
        geometry_feats = torch.randn(batch_size, seq, config.geometry_features_dim)
        
        # Forward pass
        try:
            output = model(
                temporal_seq=temporal_seq,
                physics_feats=physics_feats,
                geo_feats=geo_feats,
                storm_feats=storm_feats,
                bottomside_tec=bottomside_tec,
                geometry_feats=geometry_feats
            )
            
            # Strict Shape Assertions
            expected_shape = (batch_size, seq, 1)
            
            assert bottomside_tec.shape == expected_shape, "Bottomside TEC shape failed"
            assert output.topside_tec.shape == expected_shape, "Topside TEC shape failed"
            assert output.net_tec.shape == expected_shape, "Net TEC shape failed"
            assert output.electron_density.shape == expected_shape, "Electron Density shape failed"
            assert output.gnss_delays['vertical_delay'].shape == expected_shape, "GNSS Delay shape failed"
            
            print("  [PASS] All Physics tensors match the standardized (Batch, Seq, 1) format.")
            
        except Exception as e:
            print(f"  [FAIL] {e}")
            sys.exit(1)
            
    print("\n======================================")
    print("        TENSOR FLOW REPORT")
    print("======================================")
    print("Input Features")
    print("       v")
    print("Temporal Encoder")
    print("       v")
    print("Physics Encoder")
    print("       v")
    print(f"Topside TEC             (batch, {seq}, 1)")
    print("       v")
    print(f"Bottomside TEC          (batch, {seq}, 1)")
    print("       v")
    print("PhysicsConsistency")
    print("       v")
    print(f"Net TEC                 (batch, {seq}, 1)")
    print("       v")
    print(f"Electron Density        (batch, {seq}, 1)")
    print("       v")
    print(f"GNSS Delay              (batch, {seq}, 1)")
    print("======================================\n")

if __name__ == "__main__":
    run_verification()
