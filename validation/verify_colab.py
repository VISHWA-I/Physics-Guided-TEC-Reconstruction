"""
validation/verify_colab.py
==========================
Automated verification script for Google Colab compatibility.
Runs a series of tests to ensure the environment, paths, AMP, and imports
are fully functional before starting a long training run.
"""

import sys
import os
from pathlib import Path
import traceback

def run_check(name: str, check_fn) -> int:
    # returns 2 for PASS, 1 for WARNING, 0 for FAIL
    print(f"[{' ' * 4}] {name:<50}", end="", flush=True)
    try:
        status, details = check_fn()
        if status == 2:
            print(f"[\033[92mPASS\033[0m] {details}")
        elif status == 1:
            print(f"[\033[93mWARN\033[0m] {details}")
        else:
            print(f"[\033[91mFAIL\033[0m] {details}")
        return status
    except Exception as e:
        print(f"[\033[91mERROR\033[0m]")
        traceback.print_exc(limit=1, file=sys.stdout)
        return 0

def check_imports():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.env_config import is_colab
    try:
        import torch
        import pandas
        import numpy
        import tensorboard
        import onnx
        return 2, "All critical packages imported."
    except ImportError as e:
        if not is_colab():
            return 1, f"ImportError: {e} (Expected on local without Colab requirements)"
        return 0, f"ImportError: {e}"

def check_gpu():
    from utils.env_config import is_colab
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return 2, f"Found CUDA GPU: {name}"
    else:
        if not is_colab():
            return 1, "No CUDA GPU available. (Expected on local CPU)"
        return 0, "No CUDA GPU available. Are you on a T4/A100 runtime?"

def check_env_config():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.env_config import configure, is_colab
    
    # We force Colab mode for the test
    os.environ["FORCE_COLAB"] = "1"
    paths = configure(drive_root="/tmp/fake_drive")
    
    colab_status = is_colab()
    checkpoints_path = str(paths.checkpoints)
    
    # Cleanup env
    os.environ.pop("FORCE_COLAB", None)
    
    if not colab_status:
        return 0, "is_colab() returned False despite FORCE_COLAB=1"
        
    if checkpoints_path != str(Path("/tmp/fake_drive/checkpoints")):
        return 0, f"Paths not routing correctly: {checkpoints_path}"
        
    return 2, "env_config successfully routes paths to drive_root."

def check_tensorboard():
    from utils.env_config import is_colab
    from utils.tensorboard_logger import get_tensorboard_logger
    tb = get_tensorboard_logger("/tmp/fake_drive/tb", enabled=True)
    if not tb.is_active:
        if not is_colab():
            return 1, "TensorBoard logger inactive (Expected if tensorboard is not installed locally)."
        return 0, "TensorBoard logger failed to activate."
    tb.log_scalars(0, test_val=1.0)
    tb.close()
    return 2, "TensorBoard writer initialized and logged successfully."

def check_amp():
    from utils.env_config import get_amp_enabled
    amp = get_amp_enabled()
    import torch
    if torch.cuda.is_available() and not amp:
        return 0, "AMP is disabled despite CUDA being available."
    if not torch.cuda.is_available() and amp:
        return 0, "AMP is enabled despite no CUDA (not supported for CPU in this script)."
    return 2, f"AMP state correct (Enabled: {amp})"

def main():
    print("======================================================")
    print(" Mamba-TKAN Google Colab Compatibility Verification")
    print("======================================================")
    
    checks = [
        ("Package Imports", check_imports),
        ("GPU & CUDA Availability", check_gpu),
        ("Environment Config (Path Routing)", check_env_config),
        ("TensorBoard Initialization", check_tensorboard),
        ("Mixed Precision (AMP) Logic", check_amp),
    ]
    
    all_passed = True
    for name, fn in checks:
        if run_check(name, fn) == 0:
            all_passed = False
            
    print("======================================================")
    if all_passed:
        print("\n\033[92mALL CHECKS PASSED!\033[0m The project is fully Colab-ready.")
        print("You can safely run `train_colab.ipynb` on Google Colab.\n")
        sys.exit(0)
    else:
        print("\n\033[91mSOME CHECKS FAILED.\033[0m Please review the output above.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
