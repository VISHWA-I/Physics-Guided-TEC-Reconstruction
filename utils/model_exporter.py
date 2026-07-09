"""
utils/model_exporter.py
=======================
Automated Model Export Pipeline

Exports the trained PyTorch model into three formats for deployment:
1. PyTorch native (State Dict)
2. TorchScript (JIT compiled / traced)
3. ONNX (Open Neural Network Exchange)

Saves outputs to the configured `exports` directory from `env_config`.
Handles the dynamic dummy inputs required for tracing and ONNX export.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.logger import get_model_logger
from utils.env_config import get_paths

logger = get_model_logger("ModelExporter")


def get_dummy_inputs(
    batch_size: int = 2,
    window_size: int = 24,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate dummy tensors matching the exact input signature of the HybridModel
    forward pass.

    Returns:
        (temporal_feats, physics_feats, geo_feats, storm_feats, targets, geometry_feats)
    """
    # Feature dimensions based on standard configuration
    # These match the dimensions passed to HybridModel in the pipeline
    f_temporal = 6
    f_physics = 9
    f_geo = 4
    f_storm = 5
    f_target = 1
    f_geom = 2

    temporal_feats = torch.randn(batch_size, window_size, f_temporal, device=device)
    physics_feats = torch.randn(batch_size, window_size, f_physics, device=device)
    geo_feats = torch.randn(batch_size, f_geo, device=device)
    storm_feats = torch.randn(batch_size, f_storm, device=device)
    targets = torch.randn(batch_size, window_size, f_target, device=device)
    
    # geometry_feats is optional, but for ONNX/TorchScript tracing we must provide a fixed tensor
    geometry_feats = torch.zeros(batch_size, f_geom, device=device)

    return (temporal_feats, physics_feats, geo_feats, storm_feats, targets, geometry_feats)


def export_model(
    model: nn.Module,
    epoch: int,
    export_dir: Optional[str | Path] = None,
    export_torchscript: bool = True,
    export_onnx: bool = True,
) -> None:
    """
    Export the model to TorchScript and ONNX.

    Args:
        model: The trained PyTorch model.
        epoch: Current epoch (for naming).
        export_dir: Output directory (defaults to env_config exports).
        export_torchscript: Whether to export to TorchScript.
        export_onnx: Whether to export to ONNX.
    """
    if export_dir is None:
        paths = get_paths()
        out_dir = paths.exports
    else:
        out_dir = Path(export_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    logger.info(f"Starting model export (Epoch {epoch}) to {out_dir}")
    
    # Set to eval mode for tracing/export
    model.eval()

    # Get dummy inputs on the same device as the model
    dummy_inputs = get_dummy_inputs(batch_size=2, window_size=24, device=str(device))
    
    base_name = f"hybrid_model_epoch_{epoch}"

    # 1. PyTorch State Dict (Full)
    # Note: CheckpointManager already saves checkpoints, but we save a clean one here
    # without optimizer state for purely inference purposes.
    pt_path = out_dir / f"{base_name}.pt"
    try:
        torch.save(model.state_dict(), pt_path)
        logger.info(f"Exported PyTorch state dict: {pt_path}")
    except Exception as e:
        logger.error(f"Failed to save PyTorch state dict: {e}")

    # 2. TorchScript Export
    if export_torchscript:
        ts_path = out_dir / f"{base_name}_scripted.pt"
        try:
            # We attempt scripting first, but complex dynamic control flow often fails.
            # If so, we fall back to tracing.
            try:
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(ts_path))
                logger.info(f"Exported TorchScript (scripted): {ts_path}")
            except Exception as e:
                logger.warning(f"JIT scripting failed, falling back to tracing. Reason: {e}")
                traced_model = torch.jit.trace(model, dummy_inputs, strict=False)
                traced_model.save(str(ts_path))
                logger.info(f"Exported TorchScript (traced): {ts_path}")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")

    # 3. ONNX Export
    if export_onnx:
        onnx_path = out_dir / f"{base_name}.onnx"
        try:
            input_names = [
                "temporal_feats",
                "physics_feats",
                "geo_feats",
                "storm_feats",
                "targets",
                "geometry_feats"
            ]
            output_names = ["prediction", "aux_tkan", "aux_physics"]
            
            # Dynamic axes to allow variable batch sizes
            dynamic_axes = {
                name: {0: "batch_size"} for name in input_names
            }
            for out_name in output_names:
                dynamic_axes[out_name] = {0: "batch_size"}

            # Hide warnings during export for cleaner logs
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    model,
                    dummy_inputs,
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,  # Recommended stable version
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
            logger.info(f"Exported ONNX model: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

    # Restore training mode if needed
    model.train()
    logger.info("Model export completed.")
