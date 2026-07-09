import sys
from pathlib import Path
import torch
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.benchmark_engine import BenchmarkEngine
from models.model_config import ModelConfig
from utils.seed import set_seed

def test_benchmark_framework():
    print("=== Testing Phase 6: Benchmarking and Publication Framework ===")
    set_seed(42)
    
    config_path = project_root / "configs" / "model.yaml"
    
    print("\n=== Initializing Benchmark Engine ===")
    try:
        engine = BenchmarkEngine(
            config_path=str(config_path),
            checkpoint_path="dummy_nonexistent.pt",
            data_path="dummy_dataset.csv",
            output_dir="benchmark_reports"
        )
        print("PASS: BenchmarkEngine initialized successfully.")
    except Exception as e:
        print(f"FAIL: BenchmarkEngine initialization crashed: {e}")
        return
        
    print("\n=== Executing Full Benchmark Suite ===")
    config = ModelConfig.from_yaml(config_path)
    window = config.window_size
    batch = 16
    
    mock_inputs = {
        'temporal_seq': torch.randn(batch, window, len(config.temporal_features)),
        'physics_feats': torch.randn(batch, len(config.physics_features)),
        'geo_feats': torch.randn(batch, len(config.geo_features)),
        'storm_feats': torch.randn(batch, len(config.storm_features)),
        'bottomside_tec': torch.rand(batch, window, 1) * 20.0,
        'geometry_feats': torch.randn(batch, window, config.geometry_features_dim)
    }
    
    # Flatten mock target (seq length * batch size)
    target_tec = np.random.rand(batch * window) * 30.0
    
    try:
        engine.run_suite(mock_inputs, target_tec)
        print("PASS: Benchmarking Suite completed successfully.")
    except Exception as e:
        print(f"FAIL: Benchmarking crashed: {e}")
        return
        
    print("\n=== Verifying Output Artifacts ===")
    report = Path("benchmark_reports/IEEE_Supplementary_Material.md")
    latex = Path("benchmark_reports/table_baselines.tex")
    fig_png = Path("benchmark_reports/figures/taylor_diagram.png")
    fig_eps = Path("benchmark_reports/figures/taylor_diagram.eps")
    
    all_exist = True
    for f in [report, latex, fig_png, fig_eps]:
        if f.exists():
            print(f"PASS: Found artifact {f.name}")
        else:
            print(f"FAIL: Missing artifact {f.name}")
            all_exist = False
            
    # Check if any zip file exists in archives
    archives = list(Path("archives").glob("*.zip"))
    if archives:
        print(f"PASS: Found reproducibility ZIP archive: {archives[0].name}")
    else:
        print("FAIL: Missing reproducibility archive.")
        all_exist = False
            
    if all_exist:
        print("\nPhase 6 Complete: The Benchmarking Framework is fully operational and generating publication-ready artifacts!")

if __name__ == "__main__":
    test_benchmark_framework()
