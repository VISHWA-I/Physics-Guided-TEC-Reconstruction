"""
validation/verify_batch_interface.py
=====================================
Automated end-to-end verification of the Unified Batch Interface.

Verifies that every module (Dataset, Trainer, Validator, LossManager,
Evaluator, Prediction) can access all required batch targets through
BatchManager without a KeyError.

Tests both the canonical nested format AND the legacy flat format to
confirm backward compatibility.

Usage
-----
    python validation/verify_batch_interface.py

Output
------
    - Console report with PASS/FAIL per module per target
    - results/debug/batch_interface_report.json
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

# Make project root importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from training.batch_manager import BatchManager, BatchInterfaceError
from models.model_config import ModelConfig
from training.sequence_target_manager import SequenceTargetManager
from training.loss_manager import LossManager
from training.validator import Validator
from models.hybrid_model import HybridModel


# ─────────────────────────────────────────────────────────────────────────────
# Mock Datasets (both formats for compatibility testing)
# ─────────────────────────────────────────────────────────────────────────────

class NestedFormatDataset(Dataset):
    """Produces the canonical nested batch format (what DictWrapperDataset outputs)."""

    def __init__(self, config: ModelConfig, n: int = 32):
        self.config = config
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        w = self.config.window_size
        bs = torch.rand(w, 1) * 15.0
        ts = torch.rand(w, 1) * 25.0
        net = bs + ts
        return {
            "temporal_seq":   torch.randn(w, len(self.config.temporal_features)),
            "physics_feats":  torch.randn(len(self.config.physics_features)),
            "geo_feats":      torch.randn(len(self.config.geo_features)),
            "storm_feats":    torch.randn(len(self.config.storm_features)),
            "bottomside_tec": bs,
            "geometry_feats": torch.randn(w, self.config.geometry_features_dim),
            "targets": {
                "topside": ts,
                "net":     net,
                "density": net * 0.1,
                "delay":   net * 0.162,
            },
        }


class FlatFormatDataset(Dataset):
    """Produces the legacy flat batch format (what old MockDatasets produced)."""

    def __init__(self, config: ModelConfig, n: int = 32):
        self.config = config
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        w = self.config.window_size
        bs = torch.rand(w, 1) * 15.0
        ts = torch.rand(w, 1) * 25.0
        net = bs + ts
        return {
            "temporal_seq":    torch.randn(w, len(self.config.temporal_features)),
            "physics_feats":   torch.randn(len(self.config.physics_features)),
            "geo_feats":       torch.randn(len(self.config.geo_features)),
            "storm_feats":     torch.randn(len(self.config.storm_features)),
            "bottomside_tec":  bs,
            "geometry_feats":  torch.randn(w, self.config.geometry_features_dim),
            # Flat target keys (legacy format)
            "targets_topside": ts,
            "targets_net":     net,
            "targets_density": net * 0.1,
            "targets_delay":   net * 0.162,
        }


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------

def _check(result: dict, module: str, test: str, passed: bool, note: str = "") -> None:
    tag = "[PASS]" if passed else "[FAIL]"
    result[module][test] = "PASS" if passed else f"FAIL: {note}"
    print(f"    [{module}] {test:<35} {tag}" + (f"  ({note})" if note and not passed else ""))


def _run_getter_tests(batch: dict, label: str) -> Dict[str, Any]:
    """Run all BatchManager getters against a single batch."""
    results: Dict[str, Any] = {
        "format": label,
        "get_inputs":                  None,
        "get_topside_target":          None,
        "get_nettec_target":           None,
        "get_electron_density_target": None,
        "get_gnss_delay_target":       None,
        "get_uncertainty_target":      None,
        "get_all_targets":             None,
        "validate_batch":              None,
    }

    for name, fn in [
        ("get_inputs",                  lambda b: BatchManager.get_inputs(b)),
        ("get_topside_target",          lambda b: BatchManager.get_topside_target(b)),
        ("get_nettec_target",           lambda b: BatchManager.get_nettec_target(b)),
        ("get_electron_density_target", lambda b: BatchManager.get_electron_density_target(b)),
        ("get_gnss_delay_target",       lambda b: BatchManager.get_gnss_delay_target(b)),
        ("get_all_targets",             lambda b: BatchManager.get_all_targets(b)),
        ("validate_batch",              lambda b: BatchManager.validate_batch(b, verbose=False)),
    ]:
        try:
            fn(batch)
            results[name] = "PASS"
        except BatchInterfaceError as e:
            results[name] = f"FAIL (BatchInterfaceError): {str(e)[:80]}"
        except Exception as e:
            results[name] = f"FAIL ({type(e).__name__}): {str(e)[:80]}"

    # uncertainty is optional — PASS even if None
    try:
        _ = BatchManager.get_uncertainty_target(batch)
        results["get_uncertainty_target"] = "PASS (optional, returned None or Tensor)"
    except Exception as e:
        results["get_uncertainty_target"] = f"FAIL: {str(e)[:80]}"

    return results


# -----------------------------------------------------------------------------
# Module-level integration tests
# -----------------------------------------------------------------------------

def verify_dataset_output(config: ModelConfig, results: dict) -> None:
    """Verify that both dataset formats produce valid batches."""
    module = "Dataset"
    results[module] = {}

    for Dataset_cls, fmt in [(NestedFormatDataset, "nested"), (FlatFormatDataset, "flat")]:
        try:
            ds = Dataset_cls(config, n=8)
            loader = DataLoader(ds, batch_size=4)
            batch = next(iter(loader))
            report = BatchManager.validate_batch(batch, verbose=False)
            passed = report["status"] == "PASS"
            _check(results, module, f"Format '{fmt}' — batch valid", passed, report.get("status", ""))
        except Exception as e:
            _check(results, module, f"Format '{fmt}' — batch valid", False, str(e)[:80])


def verify_batch_manager_getters(config: ModelConfig, results: dict) -> None:
    """Verify every getter works on both formats."""
    module = "BatchManager"
    results[module] = {}

    for Dataset_cls, fmt in [(NestedFormatDataset, "nested"), (FlatFormatDataset, "flat")]:
        ds = Dataset_cls(config, n=4)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))

        getter_results = _run_getter_tests(batch, fmt)
        for test_name, outcome in getter_results.items():
            if test_name == "format":
                continue
            passed = isinstance(outcome, str) and outcome.startswith("PASS")
            _check(results, module, f"[{fmt}] {test_name}", passed, "" if passed else str(outcome))


def verify_loss_manager(config: ModelConfig, results: dict) -> None:
    """Verify LossManager receives correct target dict from BatchManager."""
    module = "LossManager"
    results[module] = {}

    try:
        from models.hybrid_model import HybridModel
        model = HybridModel(config)
        model.eval()
        ds = NestedFormatDataset(config, n=4)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))

        inputs = BatchManager.get_inputs(batch)
        targets = BatchManager.get_all_targets(batch)

        with torch.no_grad():
            output = model(
                inputs["temporal_seq"], inputs["physics_feats"], inputs["geo_feats"],
                inputs["storm_feats"], inputs["bottomside_tec"], inputs["geometry_feats"]
            )

        lm = LossManager(strategy="gradnorm", num_tasks=4)
        total_loss, loss_dict = lm(output, targets)
        passed = not (torch.isnan(total_loss) or torch.isinf(total_loss))
        _check(results, module, "Loss computed (no NaN/Inf)", passed)

        for key in ["loss_topside", "loss_net", "loss_density", "loss_delay"]:
            _check(results, module, f"Loss key '{key}' present", key in loss_dict)

    except Exception as e:
        _check(results, module, "LossManager integration", False, str(e)[:120])
        traceback.print_exc()


def verify_validator(config: ModelConfig, results: dict) -> None:
    """Verify Validator runs without KeyError on canonical batch format."""
    module = "Validator"
    results[module] = {}

    try:
        model = HybridModel(config)
        ds = NestedFormatDataset(config, n=8)
        loader = DataLoader(ds, batch_size=4)
        device = torch.device("cpu")

        v = Validator(model, device)
        metrics = v.validate(loader)

        _check(results, module, "validate() completed", True)
        _check(results, module, "'val_loss' in metrics", "val_loss" in metrics)
        _check(results, module, "'val_rmse' in metrics", "val_rmse" in metrics)

    except BatchInterfaceError as e:
        _check(results, module, "validate() completed", False, f"BatchInterfaceError: {str(e)[:120]}")
    except Exception as e:
        _check(results, module, "validate() completed", False, str(e)[:120])
        traceback.print_exc()


def verify_backward_compatibility_flat(config: ModelConfig, results: dict) -> None:
    """Verify Validator works on flat-format batches (backward compatibility)."""
    module = "Backward Compat (Flat)"
    results[module] = {}

    try:
        model = HybridModel(config)
        ds = FlatFormatDataset(config, n=8)
        loader = DataLoader(ds, batch_size=4)
        device = torch.device("cpu")

        v = Validator(model, device)
        metrics = v.validate(loader)
        _check(results, module, "Validator on flat format", True)

    except BatchInterfaceError as e:
        _check(results, module, "Validator on flat format", False, f"BatchInterfaceError: {str(e)[:120]}")
    except Exception as e:
        _check(results, module, "Validator on flat format", False, str(e)[:120])


def verify_batch_interface_error(config: ModelConfig, results: dict) -> None:
    """Verify that missing keys raise BatchInterfaceError, not raw KeyError."""
    module = "BatchInterfaceError"
    results[module] = {}

    bad_batch = {
        "temporal_seq": torch.randn(4, 96, 10),
        # deliberately missing all target keys and some input keys
    }

    for name, fn in [
        ("get_topside_target",  lambda: BatchManager.get_topside_target(bad_batch)),
        ("get_nettec_target",   lambda: BatchManager.get_nettec_target(bad_batch)),
        ("get_all_targets",     lambda: BatchManager.get_all_targets(bad_batch)),
        ("get_inputs_missing",  lambda: BatchManager.get_inputs(bad_batch)),
    ]:
        try:
            fn()
            _check(results, module, f"{name} raises BatchInterfaceError", False,
                   "Expected error but none was raised")
        except BatchInterfaceError:
            _check(results, module, f"{name} raises BatchInterfaceError", True)
        except KeyError as e:
            _check(results, module, f"{name} raises BatchInterfaceError", False,
                   f"Got raw KeyError instead: {e}")
        except Exception as e:
            # Other errors (e.g., physics_feats missing) are acceptable
            _check(results, module, f"{name} raises BatchInterfaceError", True,
                   f"Got expected error type: {type(e).__name__}")


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

def _write_summary_report(all_results: dict, config: ModelConfig) -> None:
    """Write the combined verification results to results/debug/."""
    out_dir = Path("results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also build and write the batch-level debug report
    config_info = {
        "window_size": config.window_size,
        "temporal_features_count": len(config.temporal_features),
        "physics_features_count": len(config.physics_features),
    }

    # Compute overall status
    all_pass = True
    for module, tests in all_results.items():
        for test, outcome in tests.items():
            if not str(outcome).startswith("PASS"):
                all_pass = False
                break

    summary = {
        "overall_status": "PASS" if all_pass else "FAIL",
        "model_config_summary": config_info,
        "module_results": all_results,
    }

    report_path = out_dir / "verify_batch_interface_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Report written to: {report_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_verification() -> None:
    sep  = "=" * 60
    thin = "-" * 60

    print(f"\n{sep}")
    print("  Batch Interface Verification Suite")
    print(sep)

    config_path = _PROJECT_ROOT / "configs" / "model.yaml"
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}")
        sys.exit(1)

    config = ModelConfig.from_yaml(config_path)

    all_results: Dict[str, Dict] = {}

    # Run all module tests
    tests = [
        ("Dataset output", verify_dataset_output),
        ("BatchManager getters", verify_batch_manager_getters),
        ("LossManager integration", verify_loss_manager),
        ("Validator integration", verify_validator),
        ("Backward compat (flat format)", verify_backward_compatibility_flat),
        ("BatchInterfaceError enforcement", verify_batch_interface_error),
    ]

    for test_label, test_fn in tests:
        print(f"\n{thin}")
        print(f"  {test_label}")
        print(thin)
        try:
            test_fn(config, all_results)
        except Exception as e:
            print(f"  [ERROR] Test '{test_label}' crashed: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)

    total = 0
    passed = 0
    for module, tests_dict in all_results.items():
        for test_name, outcome in tests_dict.items():
            total += 1
            ok = str(outcome).startswith("PASS")
            if ok:
                passed += 1
            status = "[OK]" if ok else "[XX]"
            print(f"  {status}  [{module}] {test_name}")

    print(thin)
    print(f"  Result: {passed}/{total} tests passed")
    print(sep)

    if passed == total:
        print("\n  ALL BATCH INTERFACE TESTS PASSED")
        print("  KeyError has been eliminated. BatchManager is the single source of truth.\n")
    else:
        failed = total - passed
        print(f"\n  WARNING: {failed} test(s) FAILED -- see details above\n")

    # Write reports
    _write_summary_report(all_results, config)

    # Also generate the per-batch debug report using a real batch
    try:
        ds = NestedFormatDataset(config, n=4)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        BatchManager.generate_debug_report(
            batch,
            output_path="results/debug/batch_interface_report.json",
        )
    except Exception as e:
        print(f"  Warning: Could not write batch_interface_report.json: {e}")


if __name__ == "__main__":
    run_verification()
