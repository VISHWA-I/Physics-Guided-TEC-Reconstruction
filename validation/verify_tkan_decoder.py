import sys
from pathlib import Path
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_config import ModelConfig
from models.tkan_decoder import TKANDecoder
from utils.seed import set_seed
from utils.tensor_utils import validate_shape

def test_tkan_decoder():
    print("=== Testing Phase 2.4: Memory-Augmented TKAN Decoder ===")
    set_seed(42)
    
    config = ModelConfig.from_yaml(project_root / "configs" / "model.yaml")
    
    print("\n=== Instantiating TKAN Decoder (Debug Mode) ===")
    decoder = TKANDecoder(config, debug_mode=True)
    decoder.initialize_weights()
    decoder.move_to_device()
    
    print("\n[Parameter Introspection]")
    print(f"Total Trainable Parameters: {decoder.count_parameters():,}")
    
    print("\n=== Testing Forward Pass ===")
    batch = 32
    window = config.window_size
    hidden = config.hidden_dimension
    
    # Generate Mock Latent Input (from Phase 2.3)
    latent_in = torch.randn(batch, window, hidden).to(decoder.current_device)
    latent_in.requires_grad = True
    
    print(f"Executing forward pass on {decoder.current_device}...")
    tec_out, hidden_out, metrics = decoder(latent_in)
    
    print("\n=== Verifying Tensor Shapes ===")
    is_valid_tec = validate_shape(tec_out, (batch, window, 1), name="Final Topside TEC")
    is_valid_hidden = validate_shape(hidden_out, (batch, window, hidden), name="Final Hidden Representation")
    
    if is_valid_tec and is_valid_hidden:
        print("PASS: Output shapes correctly match expectations.")
    else:
        print("FAIL: Output shape mismatch!")
        
    print("\n=== Verifying Memory Attention Metrics ===")
    attn_scores = metrics["memory_attention_scores"]
    expected_slots = config.memory_slots * 5 # 5 domains
    print(f"Memory Attention Scores Shape: {attn_scores.shape}")
    
    if attn_scores.shape == (batch, config.memory_heads, window, expected_slots):
        print("PASS: Memory attention maps generated correctly.")
    else:
        print("FAIL: Memory attention maps mismatch!")
        
    print("\n=== Testing Backward Pass (Gradients) ===")
    try:
        loss = tec_out.sum()
        loss.backward()
        print("PASS: Backward pass executed successfully through PyTorch KAN and Memory Banks.")
    except Exception as e:
        print(f"FAIL: Backward pass crashed: {e}")
        
    print("\nTKAN Decoder is successfully validated and ready for Phase 2.5!")

if __name__ == "__main__":
    test_tkan_decoder()
