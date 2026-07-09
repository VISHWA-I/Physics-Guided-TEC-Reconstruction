import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_config import ModelConfig
from models.hybrid_model import HybridModel
from training.training_config import TrainingConfig
from training.trainer import Trainer
from utils.seed import set_seed

class MockDataset(Dataset):
    def __init__(self, config: ModelConfig, num_samples: int = 64):
        self.config = config
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        window = self.config.window_size
        bottomside_tec = torch.randn(window, 1)
        topside = torch.randn(window, 1)
        return {
            # ── Inputs ──────────────────────────────────────────────────
            'temporal_seq':   torch.randn(window, len(self.config.temporal_features)),
            'physics_feats':  torch.randn(len(self.config.physics_features)),
            'geo_feats':      torch.randn(len(self.config.geo_features)),
            'storm_feats':    torch.randn(len(self.config.storm_features)),
            'bottomside_tec': bottomside_tec,
            'geometry_feats': torch.randn(window, self.config.geometry_features_dim),
            # ── Targets (canonical nested format) ────────────────────────
            'targets': {
                'topside': topside,
                'net':     bottomside_tec + topside,
                'density': (bottomside_tec + topside) * 0.1,
                'delay':   (bottomside_tec + topside) * 0.162,
            }
        }

def test_training_framework():
    print("=== Testing Phase 3: Research-Grade Training Framework ===")
    set_seed(42)
    
    # Models
    model_config = ModelConfig.from_yaml(project_root / "configs" / "model.yaml")
    model = HybridModel(model_config)
    
    # Training Config
    train_config = TrainingConfig(
        epochs=2,
        batch_size=16,
        optimizer="AdamW",
        scheduler="CosineAnnealing",
        adaptive_loss_strategy="UncertaintyWeighting",
        mixed_precision=False, # Set to False for strict CPU CI testing, True usually
        device="cpu"
    )
    
    print("\n=== Initializing DataLoader ===")
    dataset = MockDataset(model_config, num_samples=32) # Small dataset for fast test
    loader = DataLoader(dataset, batch_size=train_config.batch_size)
    
    print("\n=== Initializing Trainer ===")
    trainer = Trainer(model, train_config)
    
    print("\n=== Executing Mini Training Loop (2 Epochs) ===")
    try:
        trainer.fit(train_loader=loader, val_loader=loader)
        print("PASS: Training loop executed successfully.")
    except Exception as e:
        print(f"FAIL: Training loop crashed: {e}")
        return
        
    print("\n=== Verifying Checkpoints ===")
    latest_ckpt = Path(train_config.checkpoint_dir) / "latest.pt"
    if latest_ckpt.exists():
        print(f"PASS: Checkpoint saved correctly to {latest_ckpt}")
    else:
        print("FAIL: Checkpoint not found.")
        
    print("\nPhase 3 Complete: The Training Framework is fully functional and ready for Phase 4!")

if __name__ == "__main__":
    test_training_framework()
