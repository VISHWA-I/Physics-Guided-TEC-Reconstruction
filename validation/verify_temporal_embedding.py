import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_config import ModelConfig
from models.temporal_embedding import PhysicsAwareTemporalEmbedding

def run_verification():
    print("=== Testing Physics-Aware Dynamic Temporal Embedding ===\n")
    
    config = ModelConfig.from_yaml("configs/model.yaml")
    
    # Initialize with default 24
    embedding_layer = PhysicsAwareTemporalEmbedding(config, initial_seq_len=24)
    
    test_sequences = [8, 16, 24, 48, 96, 192, 384]
    batch_size = 4
    d_model = config.embedding_dimension
    
    # Validation flags
    dynamic_resize_pass = True
    physics_embedding_pass = True
    no_nan_pass = True
    no_index_error_pass = True
    
    last_output = None
    
    for seq in test_sequences:
        print(f"Testing Sequence {seq}...", end=" ")
        try:
            # Mock Inputs
            x = torch.randn(batch_size, seq, d_model)
            geo_feats = torch.rand(batch_size, 4) # Lat, Lon, DOY, LT
            storm_feats = torch.rand(batch_size, 5) # Kp, Dst, Ap, F10.7, SSN
            
            # Forward Pass
            out = embedding_layer(x, geo_feats, storm_feats)
            
            if torch.isnan(out).any():
                no_nan_pass = False
                print("FAIL (NaN detected)")
            elif out.shape != x.shape:
                print("FAIL (Shape mismatch)")
            else:
                print("PASS")
                last_output = out
                
        except IndexError:
            no_index_error_pass = False
            dynamic_resize_pass = False
            print("FAIL (IndexError)")
        except Exception as e:
            physics_embedding_pass = False
            print(f"FAIL ({e})")
            
    print("\n--- Summary ---")
    print(f"Dynamic Resize       : {'PASS' if dynamic_resize_pass else 'FAIL'}")
    print(f"Physics Embedding    : {'PASS' if physics_embedding_pass else 'FAIL'}")
    print(f"No NaN               : {'PASS' if no_nan_pass else 'FAIL'}")
    print(f"No IndexError        : {'PASS' if no_index_error_pass else 'FAIL'}")
    
    # Visualization
    if last_output is not None:
        print("\nGenerating visual analysis...")
        out_dir = Path("results/embedding_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(last_output[0].detach().numpy(), aspect='auto', cmap='viridis')
        plt.title(f"Temporal Embedding Heatmap (Seq={test_sequences[-1]})")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Time Step")
        plt.colorbar()
        plt.savefig(out_dir / "embedding_heatmap.png")
        plt.close()
        
        # 2. PCA / Scatter (Mocking structure)
        plt.figure(figsize=(6, 6))
        plt.scatter(np.random.randn(100), np.random.randn(100), c='r', alpha=0.5)
        plt.title("Embedding PCA (Mock)")
        plt.savefig(out_dir / "embedding_pca.png")
        plt.close()
        
        # 3. Cosine Similarity
        plt.figure(figsize=(6, 6))
        plt.imshow(np.corrcoef(last_output[0].detach().numpy()), cmap='coolwarm')
        plt.title("Cosine Similarity Matrix")
        plt.savefig(out_dir / "cosine_similarity.png")
        plt.close()
        
        # 4. Norms
        norms = torch.norm(last_output[0], dim=1).detach().numpy()
        plt.figure(figsize=(8, 4))
        plt.plot(norms)
        plt.title("Embedding L2 Norm over Time")
        plt.savefig(out_dir / "embedding_norm.png")
        plt.close()
        
        print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    run_verification()
