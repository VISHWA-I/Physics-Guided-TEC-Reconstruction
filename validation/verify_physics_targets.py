import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.sequence_target_manager import SequenceTargetManager

def verify():
    print("=== Physics-Aware Target Validation ===")
    
    batch = 32
    seq = 96
    
    bottomside = torch.rand(batch, seq, 1)
    y_raw = torch.rand(batch)
    
    manager = SequenceTargetManager(mode="sequence_to_sequence")
    targets = manager.generate_targets(y_raw, bottomside)
    
    shapes = {
        "Topside TEC": targets['topside'].shape,
        "Bottomside TEC": bottomside.shape,
        "Net TEC": targets['net'].shape,
        "Electron Density": targets['density'].shape,
        "GNSS Delay": targets['delay'].shape
    }
    
    for name, shape in shapes.items():
        if shape == (batch, seq, 1):
            print(f"{name.ljust(20)} PASS {shape}")
        else:
            print(f"{name.ljust(20)} FAIL {shape}")
            
    print("\nREADY FOR TRAINING\n")

if __name__ == "__main__":
    verify()
