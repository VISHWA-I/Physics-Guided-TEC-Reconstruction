"""
training/batch_manager.py
=========================
Unified Batch Interface for the Hybrid Mamba-TKAN TEC Reconstruction Project.

This module is the ONLY canonical interface for reading batch dictionaries.
All modules (Trainer, Validator, Evaluator, LossManager, etc.) must access
batch contents exclusively through BatchManager static methods.

Supported Batch Formats
-----------------------
Both formats are automatically recognized and transparently normalized:

Format A — Nested (canonical, produced by DictWrapperDataset / train.py):
    batch = {
        'temporal_seq': Tensor,
        'physics_feats': Tensor,
        'geo_feats':     Tensor,
        'storm_feats':   Tensor,
        'bottomside_tec': Tensor,
        'geometry_feats': Tensor (optional),
        'targets': {
            'topside':  Tensor,   # shape (Batch, Seq, 1)
            'net':      Tensor,
            'density':  Tensor,
            'delay':    Tensor,
        }
    }

Format B — Flat (legacy, produced by mock datasets and old verify scripts):
    batch = {
        'temporal_seq':   Tensor,
        'physics_feats':  Tensor,
        'geo_feats':      Tensor,
        'storm_feats':    Tensor,
        'bottomside_tec': Tensor,
        'geometry_feats': Tensor (optional),
        'targets_topside': Tensor,
        'targets_net':     Tensor  (optional),
        'targets_density': Tensor  (optional),
        'targets_delay':   Tensor  (optional),
        'targets_gnss':    Tensor  (optional),
        'targets_uncertainty': Tensor (optional),
    }

Usage
-----
    from training.batch_manager import BatchManager

    inputs = BatchManager.get_inputs(batch, device)
    topside = BatchManager.get_topside_target(batch, device)
    targets = BatchManager.get_all_targets(batch, device)

Error Handling
--------------
    If a required key is missing in BOTH formats,
    BatchManager raises BatchInterfaceError with:
        - Missing key
        - Available keys
        - Expected keys
        - Suggested fix

Author  : Antigravity / Final Year Project
Phase   : 3 — Training Framework Standardization
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.logger import get_model_logger

logger = get_model_logger("BatchManager")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class BatchInterfaceError(Exception):
    """
    Raised when a required batch key is missing.
    Provides full diagnostic context to prevent silent failures.
    """

    def __init__(
        self,
        missing_key: str,
        available_keys: List[str],
        expected_keys: List[str],
        suggested_fix: str = "",
    ) -> None:
        self.missing_key = missing_key
        self.available_keys = available_keys
        self.expected_keys = expected_keys
        self.suggested_fix = suggested_fix

        message = (
            f"\n{'='*60}\n"
            f"  BatchInterfaceError\n"
            f"{'='*60}\n"
            f"  Missing Key    : '{missing_key}'\n"
            f"  Available Keys : {available_keys}\n"
            f"  Expected Keys  : {expected_keys}\n"
            f"  Suggested Fix  : {suggested_fix or 'Ensure your Dataset returns a dict with the canonical batch format.'}\n"
            f"{'='*60}"
        )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Key registries
# ---------------------------------------------------------------------------

# All keys that must be present as inputs (geometry_feats is optional)
_REQUIRED_INPUT_KEYS: List[str] = [
    "temporal_seq",
    "physics_feats",
    "geo_feats",
    "storm_feats",
    "bottomside_tec",
]
_OPTIONAL_INPUT_KEYS: List[str] = ["geometry_feats"]

# Canonical nested targets dict key
_NESTED_TARGETS_KEY = "targets"

# Target sub-keys under nested format
_NESTED_TARGET_SUBKEYS: Dict[str, str] = {
    "topside":     "topside",
    "net":         "net",
    "density":     "density",
    "delay":       "delay",
    "gnss":        "gnss",
    "uncertainty": "uncertainty",
}

# Flat format fallback mappings  (flat key → logical name)
_FLAT_TARGET_KEYS: Dict[str, str] = {
    "topside":     "targets_topside",
    "bottomside":  "bottomside_tec",       # bottomside is an input; for target symmetry
    "net":         "targets_net",
    "density":     "targets_density",
    "delay":       "targets_delay",
    "gnss":        "targets_gnss",
    "uncertainty": "targets_uncertainty",
}

# Additional legacy aliases that have appeared in the wild
_LEGACY_ALIASES: Dict[str, str] = {
    "topside_tec":        "topside",
    "net_tec":            "net",
    "electron_density":   "density",
    "gnss_delay":         "delay",
    "gnss_delays":        "gnss",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_format(batch: dict) -> str:
    """
    Detect whether the batch is in nested ('nested') or flat ('flat') format.
    Returns 'nested', 'flat', or 'unknown'.
    """
    if _NESTED_TARGETS_KEY in batch and isinstance(batch[_NESTED_TARGETS_KEY], dict):
        return "nested"
    # Check for any flat target key
    for flat_key in _FLAT_TARGET_KEYS.values():
        if flat_key in batch:
            return "flat"
    # Check legacy aliases
    for alias in _LEGACY_ALIASES:
        if alias in batch:
            return "flat"
    return "unknown"


def _resolve_target(batch: dict, logical_name: str) -> Optional[torch.Tensor]:
    """
    Resolve a logical target name from the batch, trying all format variants.
    Returns the Tensor if found, else None.

    Order of resolution:
        1. Nested:  batch['targets'][logical_name]
        2. Flat:    batch[_FLAT_TARGET_KEYS[logical_name]]
        3. Legacy:  batch[alias] where alias maps to logical_name
    """
    # 1. Nested format
    if _NESTED_TARGETS_KEY in batch and isinstance(batch[_NESTED_TARGETS_KEY], dict):
        targets_dict = batch[_NESTED_TARGETS_KEY]
        if logical_name in targets_dict:
            return targets_dict[logical_name]

    # 2. Flat format
    flat_key = _FLAT_TARGET_KEYS.get(logical_name)
    if flat_key and flat_key in batch:
        return batch[flat_key]

    # 3. Legacy aliases: scan batch keys for any alias that maps to this logical name
    for alias, mapped in _LEGACY_ALIASES.items():
        if mapped == logical_name and alias in batch:
            return batch[alias]

    return None


def _require_target(
    batch: dict,
    logical_name: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Resolve a required target. Raises BatchInterfaceError if not found.
    Optionally moves to device.
    """
    tensor = _resolve_target(batch, logical_name)
    if tensor is None:
        # Build helpful diagnostics
        available = list(batch.keys())
        if _NESTED_TARGETS_KEY in batch and isinstance(batch[_NESTED_TARGETS_KEY], dict):
            available += [f"targets.{k}" for k in batch[_NESTED_TARGETS_KEY].keys()]

        expected = [
            f"batch['targets']['{logical_name}']  (nested format)",
            f"batch['{_FLAT_TARGET_KEYS.get(logical_name, logical_name)}']  (flat format)",
        ]
        fix = (
            f"Either add 'targets' as a nested dict with key '{logical_name}', "
            f"or add a top-level key '{_FLAT_TARGET_KEYS.get(logical_name, logical_name)}'."
        )
        raise BatchInterfaceError(
            missing_key=logical_name,
            available_keys=available,
            expected_keys=expected,
            suggested_fix=fix,
        )
    if device is not None:
        return tensor.to(device)
    return tensor


def _optional_target(
    batch: dict,
    logical_name: str,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Resolve an optional target. Returns None if not present."""
    tensor = _resolve_target(batch, logical_name)
    if tensor is None:
        return None
    if device is not None:
        return tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# BatchManager
# ---------------------------------------------------------------------------

class BatchManager:
    """
    The single canonical interface for reading batch contents.

    All modules in the training/validation/evaluation pipeline must
    read batch data exclusively through these static methods.

    No module should ever index a batch dict directly with a string key.
    """

    # Track whether we've already printed the batch diagnostics for this run
    _validated_batch_fingerprint: Optional[int] = None

    # -----------------------------------------------------------------------
    # Input accessors
    # -----------------------------------------------------------------------

    @staticmethod
    def get_inputs(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract all input tensors from a batch.

        Returns
        -------
        dict with keys:
            temporal_seq, physics_feats, geo_feats, storm_feats,
            bottomside_tec, geometry_feats (may be None)
        """
        result: Dict[str, Optional[torch.Tensor]] = {}

        for key in _REQUIRED_INPUT_KEYS:
            if key not in batch:
                raise BatchInterfaceError(
                    missing_key=key,
                    available_keys=list(batch.keys()),
                    expected_keys=_REQUIRED_INPUT_KEYS,
                    suggested_fix=(
                        f"Ensure your Dataset.__getitem__ returns a dict containing '{key}'."
                    ),
                )
            t = batch[key]
            result[key] = t.to(device) if device is not None else t

        for key in _OPTIONAL_INPUT_KEYS:
            t = batch.get(key)
            if t is not None and device is not None:
                t = t.to(device)
            result[key] = t

        return result

    # -----------------------------------------------------------------------
    # Individual target accessors
    # -----------------------------------------------------------------------

    @staticmethod
    def get_topside_target(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the topside TEC target tensor.
        Resolves both batch['targets']['topside'] and batch['targets_topside'].
        """
        return _require_target(batch, "topside", device)

    @staticmethod
    def get_bottomside_target(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the bottomside TEC target tensor.
        In most cases this is the same as the input bottomside_tec.
        """
        # Try standard target resolution first
        tensor = _resolve_target(batch, "bottomside")
        if tensor is None:
            # Fall back to the bottomside_tec input key (it IS the target for autoencoder use)
            if "bottomside_tec" in batch:
                tensor = batch["bottomside_tec"]
            else:
                raise BatchInterfaceError(
                    missing_key="bottomside",
                    available_keys=list(batch.keys()),
                    expected_keys=["batch['targets']['bottomside']", "batch['bottomside_tec']"],
                    suggested_fix="Ensure batch contains 'bottomside_tec' or 'targets.bottomside'.",
                )
        return tensor.to(device) if device is not None else tensor

    @staticmethod
    def get_nettec_target(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the Net TEC target tensor."""
        return _require_target(batch, "net", device)

    @staticmethod
    def get_electron_density_target(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the electron density target tensor."""
        return _require_target(batch, "density", device)

    @staticmethod
    def get_gnss_delay_target(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the GNSS delay target tensor."""
        return _require_target(batch, "delay", device)

    @staticmethod
    def get_uncertainty_target(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Return the uncertainty target tensor (optional).
        Returns None if not present in the batch — this is not an error.
        """
        return _optional_target(batch, "uncertainty", device)

    # -----------------------------------------------------------------------
    # Bulk target accessor
    # -----------------------------------------------------------------------

    @staticmethod
    def get_all_targets(
        batch: dict,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return all available targets as a canonical dict with keys:
            topside, net, density, delay
        plus optional keys if present:
            gnss, uncertainty

        This is the dict expected by LossManager.forward().
        """
        targets: Dict[str, torch.Tensor] = {
            "topside": BatchManager.get_topside_target(batch, device),
            "net":     BatchManager.get_nettec_target(batch, device),
            "density": BatchManager.get_electron_density_target(batch, device),
            "delay":   BatchManager.get_gnss_delay_target(batch, device),
        }

        # Optional targets — silently omitted if not present
        gnss = _optional_target(batch, "gnss", device)
        if gnss is not None:
            targets["gnss"] = gnss

        uncertainty = _optional_target(batch, "uncertainty", device)
        if uncertainty is not None:
            targets["uncertainty"] = uncertainty

        return targets

    # -----------------------------------------------------------------------
    # Validation & diagnostics
    # -----------------------------------------------------------------------

    @staticmethod
    def validate_batch(batch: dict, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate a batch dictionary against the expected schema.
        Prints a structured diagnostic report.

        Parameters
        ----------
        batch : dict
            The batch to validate.
        verbose : bool
            If True, print the full report to stdout.

        Returns
        -------
        dict
            Validation result with keys:
                format, batch_keys, missing_input_keys,
                available_target_keys, missing_required_targets,
                unexpected_keys, status
        """
        fmt = _detect_format(batch)
        batch_keys = list(batch.keys())

        # Input key check
        missing_inputs = [k for k in _REQUIRED_INPUT_KEYS if k not in batch]

        # Target availability check
        available_targets: Dict[str, bool] = {}
        for name in ["topside", "net", "density", "delay", "gnss", "uncertainty"]:
            available_targets[name] = _resolve_target(batch, name) is not None

        required_targets = ["topside", "net", "density", "delay"]
        missing_required = [t for t in required_targets if not available_targets[t]]

        all_pass = (len(missing_inputs) == 0) and (len(missing_required) == 0)
        status = "PASS" if all_pass else "FAIL"

        report = {
            "format":                   fmt,
            "batch_keys":               batch_keys,
            "missing_input_keys":       missing_inputs,
            "available_targets":        available_targets,
            "missing_required_targets": missing_required,
            "status":                   status,
        }

        if verbose:
            sep = "-" * 55
            print(f"\n{sep}")
            print(f"  Batch Validation Report")
            print(sep)
            print(f"  Detected Format     : {fmt.upper()}")
            print(f"  Batch Keys          : {batch_keys}")
            print(f"  Missing Input Keys  : {missing_inputs or 'None'}")
            print(sep)
            print(f"  Target              Status")
            print(sep)
            for name, found in available_targets.items():
                tag = "PASS" if found else "MISSING"
                print(f"    {name:<20} {tag}")
            print(sep)
            print(f"  Overall Status      : {status}")
            print(sep)

        return report

    @staticmethod
    def generate_debug_report(
        batch: dict,
        output_path: str = "results/debug/batch_interface_report.json",
    ) -> Dict[str, Any]:
        """
        Generate a detailed JSON report describing the batch structure,
        tensor shapes, target names, and compatibility status.

        Parameters
        ----------
        batch : dict
            The batch to inspect.
        output_path : str
            Path where the JSON report will be written.

        Returns
        -------
        dict
            The full report dictionary.
        """
        fmt = _detect_format(batch)

        # Build tensor shape map
        def _shape(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return list(v.shape)
            if isinstance(v, dict):
                return {kk: _shape(vv) for kk, vv in v.items()}
            return str(type(v).__name__)

        tensor_shapes: Dict[str, Any] = {}
        for k, v in batch.items():
            tensor_shapes[k] = _shape(v)

        # Resolve which targets are available
        target_names: List[str] = []
        for name in ["topside", "net", "density", "delay", "gnss", "uncertainty"]:
            if _resolve_target(batch, name) is not None:
                target_names.append(name)

        # Compatibility status
        flat_ok = _detect_format(batch) in ("flat", "nested")
        nested_ok = _NESTED_TARGETS_KEY in batch and isinstance(batch[_NESTED_TARGETS_KEY], dict)

        # Missing / unexpected keys
        all_known = (
            _REQUIRED_INPUT_KEYS
            + _OPTIONAL_INPUT_KEYS
            + [_NESTED_TARGETS_KEY]
            + list(_FLAT_TARGET_KEYS.values())
        )
        unexpected = [k for k in batch.keys() if k not in all_known]

        val_result = BatchManager.validate_batch(batch, verbose=False)

        report: Dict[str, Any] = {
            "batch_structure": {
                "format": fmt,
                "top_level_keys": list(batch.keys()),
            },
            "tensor_shapes": tensor_shapes,
            "target_names": target_names,
            "metadata": {
                "nested_targets_present": nested_ok,
                "flat_targets_present": fmt == "flat",
                "optional_geometry_present": "geometry_feats" in batch,
                "optional_uncertainty_present": "uncertainty" in target_names,
            },
            "compatibility_status": {
                "flat_format_supported":   "PASS",
                "nested_format_supported": "PASS",
                "missing_input_keys":     val_result["missing_input_keys"],
                "missing_target_keys":    val_result["missing_required_targets"],
                "unexpected_keys":        unexpected,
                "overall":               val_result["status"],
            },
        }

        # Write to disk
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Batch interface debug report written to: {out_path}")

        return report

    @staticmethod
    def print_expected_schema() -> None:
        """Print the canonical batch schema to stdout for documentation purposes."""
        print("""
Canonical Batch Schema (Nested Format — Required by all modules)
================================================================
batch = {
    # ── Inputs ──────────────────────────────────────────────────
    'temporal_seq':   Tensor  (Batch, SeqLen, n_temporal_features),
    'physics_feats':  Tensor  (Batch, n_physics_features),
    'geo_feats':      Tensor  (Batch, n_geo_features),
    'storm_feats':    Tensor  (Batch, n_storm_features),
    'bottomside_tec': Tensor  (Batch, SeqLen, 1),
    'geometry_feats': Tensor  (Batch, SeqLen, geometry_dim),   # OPTIONAL

    # ── Targets ─────────────────────────────────────────────────
    'targets': {
        'topside':  Tensor  (Batch, SeqLen, 1),   # REQUIRED
        'net':      Tensor  (Batch, SeqLen, 1),   # REQUIRED
        'density':  Tensor  (Batch, SeqLen, 1),   # REQUIRED
        'delay':    Tensor  (Batch, SeqLen, 1),   # REQUIRED
        'gnss':     Tensor  (Batch, SeqLen, 7),   # OPTIONAL
        'uncertainty': Tensor  (Batch, SeqLen, 1), # OPTIONAL
    }
}

Legacy flat format is also auto-detected:
    batch['targets_topside'], batch['targets_net'], etc.
""")
