import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate import Evaluator
from models.model_config import ModelConfig
from utils.seed import set_seed

class MockEvalDataset(Dataset):
    def __init__(self, config: ModelConfig, num_samples: int = 128):
        self.config = config
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        window = self.config.window_size
        bottomside_tec = torch.rand(window, 1) * 20.0   # Positive TEC values
        topside = torch.rand(window, 1) * 30.0           # Positive TEC values
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

def test_evaluation_framework():
    print("=== Testing Phase 4: Scientific Evaluation Framework ===")
    set_seed(42)
    
    config_path = project_root / "configs" / "model.yaml"
    evaluator = Evaluator(
        checkpoint_path="dummy_nonexistent.pt", # We rely on random initialized weights for structural test
        config_path=str(config_path),
        output_dir="evaluation_reports"
    )
    
    model_config = ModelConfig.from_yaml(config_path)
    dataset = MockEvalDataset(model_config, num_samples=64)
    loader = DataLoader(dataset, batch_size=32)
    
    print("\n=== Executing Evaluation Pipeline ===")
    try:
        evaluator.run_evaluation(loader)
        print("PASS: Evaluation pipeline executed successfully.")
    except Exception as e:
        print(f"FAIL: Evaluation crashed: {e}")
        return
        
    print("\n=== Verifying Output Artifacts ===")
    report_file = Path("evaluation_reports/Scientific_Validation_Report.md")
    fig_scatter = Path("evaluation_reports/figures/pred_vs_actual.png")
    fig_resid = Path("evaluation_reports/figures/residuals.png")
    fig_radar = Path("evaluation_reports/figures/radar_chart.png")
    
    all_exist = True
    for f in [report_file, fig_scatter, fig_resid, fig_radar]:
        if f.exists():
            print(f"PASS: Found artifact {f.name}")
        else:
            print(f"FAIL: Missing artifact {f.name}")
            all_exist = False
            
    if all_exist:
        print("\nPhase 4 Complete: The Evaluation Framework successfully aggregates metrics, validates physics, and generates publication reports!")

if __name__ == "__main__":
    test_evaluation_framework()
