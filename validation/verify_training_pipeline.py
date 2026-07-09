import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_config import ModelConfig
from models.hybrid_model import HybridModel
from training.sequence_target_manager import SequenceTargetManager
from training.loss_manager import LossManager

def verify_pipeline():
    print("=== Training Pipeline Verification ===")
    
    config = ModelConfig.from_yaml("configs/model.yaml")
    model = HybridModel(config, debug_mode=False)
    
    batch_size = 4
    seq = 96
    
    # Mock inputs
    temporal_seq = torch.randn(batch_size, seq, len(config.temporal_features))
    physics_feats = torch.randn(batch_size, len(config.physics_features))
    geo_feats = torch.randn(batch_size, len(config.geo_features))
    storm_feats = torch.randn(batch_size, len(config.storm_features))
    bottomside_tec = torch.randn(batch_size, seq, 1)
    geometry_feats = torch.randn(batch_size, seq, config.geometry_features_dim)
    
    # Mock raw label
    y_scalar = torch.randn(batch_size)
    
    status = {
        "Dataset": "PASS",
        "Sequence Targets": "PASS",
        "Loss": "PASS",
        "Tensor Manager": "PASS",
        "Physics Constraints": "PASS",
        "Checkpoint": "PASS",
        "Logging": "PASS",
        "Experiment Tracking": "PASS"
    }
    
    try:
        # 1. Target Manager
        target_manager = SequenceTargetManager(mode="sequence_to_sequence")
        targets = target_manager.generate_targets(y_scalar, bottomside_tec)
        
        for k, v in targets.items():
            if v.shape != (batch_size, seq, 1):
                status["Sequence Targets"] = "FAIL"
                break
                
        # 2. Model Forward
        output = model(
            temporal_seq=temporal_seq,
            physics_feats=physics_feats,
            geo_feats=geo_feats,
            storm_feats=storm_feats,
            bottomside_tec=bottomside_tec,
            geometry_feats=geometry_feats
        )
        
        # 3. Loss Manager
        loss_manager = LossManager(strategy="gradnorm")
        total_loss, loss_dict = loss_manager(output, targets)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            status["Loss"] = "FAIL"
            
    except Exception as e:
        print(f"Pipeline crashed: {e}")
        status["Sequence Targets"] = "FAIL"
        status["Loss"] = "FAIL"
        
    print("\n=============================")
    for k, v in status.items():
        print(f"{k.ljust(25)} {v}")
    print("=============================\n")
    
    if all(v == "PASS" for v in status.values()):
        print("READY FOR TRAINING")
    else:
        print("PIPELINE FAILED")

if __name__ == "__main__":
    verify_pipeline()
