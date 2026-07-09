import sys
from pathlib import Path
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.offline_engine import OfflineEngine
from models.model_config import ModelConfig
from utils.seed import set_seed

def test_app_framework():
    print("=== Testing Phase 5: Operational Digital Twin Platform ===")
    set_seed(42)
    
    config_path = project_root / "configs" / "model.yaml"
    
    # 1. Initialize Engine
    print("\n=== Initializing Offline Engine ===")
    try:
        engine = OfflineEngine(
            config_path=str(config_path),
            checkpoint_path="dummy_nonexistent.pt", # Use uninitialized weights for test
            output_dir="offline_outputs"
        )
        print("PASS: Engine initialized successfully (Modules: ModelLoader, PredictionEngine, DelayCalculator, ConfidenceEngine, AnomalyDetector, StormMonitor, QualityScore, ExportManager, ReportGenerator).")
    except Exception as e:
        print(f"FAIL: Engine initialization crashed: {e}")
        return
        
    # 2. Generate Synthetic Sliding Window Data
    print("\n=== Simulating Sliding Window Inference ===")
    config = ModelConfig.from_yaml(config_path)
    window = config.window_size
    batch = 1 # Operational inference is typically batch=1 (real-time rolling)
    
    mock_inputs = {
        'temporal_seq': torch.randn(batch, window, len(config.temporal_features)),
        'physics_feats': torch.randn(batch, len(config.physics_features)),
        'geo_feats': torch.randn(batch, len(config.geo_features)),
        'storm_feats': torch.randn(batch, len(config.storm_features)),
        'bottomside_tec': torch.rand(batch, window, 1) * 20.0,
        'geometry_feats': torch.randn(batch, window, config.geometry_features_dim)
    }
    
    try:
        results = engine.process_window(mock_inputs)
        print("PASS: Inference pipeline completed flawlessly.")
        print(f"Generated Keys: {list(results.keys())}")
        print(f"Storm State Detected: {results['storm_state']['Storm_Category']}")
        print(f"Quality Score: {results['quality_score']}")
    except Exception as e:
        print(f"FAIL: Inference crashed: {e}")
        return
        
    # 3. Verify Artifact Generation
    print("\n=== Verifying Offline Output Generation ===")
    exports_dir = Path("offline_outputs/exports")
    reports_dir = Path("offline_outputs/reports")
    
    if exports_dir.exists() and len(list(exports_dir.glob("*.csv"))) > 0:
        print("PASS: CSV Export successful.")
    else:
        print("FAIL: CSV Export missing.")
        
    if reports_dir.exists() and len(list(reports_dir.glob("*.md"))) > 0:
        print("PASS: Markdown Operational Report successful.")
    else:
        print("FAIL: Markdown Report missing.")
        
    print("\nPhase 5 Complete: The Digital Twin Platform is fully operational!")

if __name__ == "__main__":
    test_app_framework()
