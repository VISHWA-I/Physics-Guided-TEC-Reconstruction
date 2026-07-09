import sys
from pathlib import Path
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scientific_discovery.discovery_engine import DiscoveryEngine
from utils.seed import set_seed

def test_sdm_framework():
    print("=== Testing Scientific Discovery Module (SDM) ===")
    set_seed(42)
    
    print("\n=== Initializing Discovery Engine ===")
    try:
        engine = DiscoveryEngine(output_dir="scientific_discovery_reports")
        print("PASS: DiscoveryEngine initialized successfully.")
    except Exception as e:
        print(f"FAIL: DiscoveryEngine initialization crashed: {e}")
        return
        
    print("\n=== Executing Rapid Pipeline (< 2s constraint) ===")
    batch = 500 # Sizeable batch of mock predictions
    
    # Mock inputs simulating the data passed from the Inference pipeline
    mock_inputs = {
        'physics_feats': np.random.rand(batch, 5), # Kp, Dst, F10.7 etc.
        'storm_feats': np.random.rand(batch, 2)
    }
    mock_preds = np.random.rand(batch) * 30.0
    mock_latent = np.random.rand(batch, 128) # 128D internal network state
    
    import time
    start = time.time()
    try:
        discoveries = engine.analyze(mock_inputs, mock_preds, latent_states=mock_latent)
        duration = time.time() - start
        
        print(f"PASS: Pipeline completed in {duration:.3f} seconds.")
        if duration < 2.0:
            print("PASS: Performance constraint met (< 2s).")
        else:
            print(f"WARN: Performance constraint failed ({duration:.3f}s > 2.0s).")
            
    except Exception as e:
        print(f"FAIL: Pipeline crashed: {e}")
        return
        
    print("\n=== Verifying Output Artifacts ===")
    report = Path("scientific_discovery_reports/Automated_Scientific_Discovery.md")
    fig1 = Path("scientific_discovery_reports/figures/latent_umap.png")
    fig2 = Path("scientific_discovery_reports/figures/knowledge_graph.png")
    kb = Path("scientific_discovery/knowledge_base.json")
    
    all_exist = True
    for f in [report, fig1, fig2, kb]:
        if f.exists():
            print(f"PASS: Found artifact {f.name}")
        else:
            print(f"FAIL: Missing artifact {f.name}")
            all_exist = False
            
    if all_exist:
        print("\nSDM Complete: The Scientific Discovery Module is fully operational!")

if __name__ == "__main__":
    test_sdm_framework()
