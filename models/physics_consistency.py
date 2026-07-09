import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from utils.logger import get_model_logger
from models.tensor_manager import TensorManager

logger = get_model_logger("PhysicsConstraintEngine")

class PhysicsConstraintEngine(nn.Module):
    """
    Orchestrates the physical evaluation pipeline.
    Responsible for validating, tracking, and executing the dependent physics heads 
    (Net TEC -> Electron Density -> GNSS Delay) securely and ensuring physical compliance.
    """

    def __init__(self, nettec_head: nn.Module, electron_density_head: nn.Module, gnss_delay_head: nn.Module, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode
        self.tensor_manager = TensorManager(debug_mode=debug_mode)
        
        # Wrapped Physics Heads
        self.nettec_head = nettec_head
        self.electron_density_head = electron_density_head
        self.gnss_delay_head = gnss_delay_head
        
        logger.info("PhysicsConstraintEngine (Validation Framework) initialized.")

    def forward(self, 
                bottomside_tec: torch.Tensor, 
                predicted_topside_tec: torch.Tensor, 
                hidden_rep: torch.Tensor, 
                geometry_feats: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """
        Executes the hierarchical prediction while mathematically enforcing physics constraints.
        
        Args:
            bottomside_tec (torch.Tensor): Observed/Processed Bottomside TEC.
            predicted_topside_tec (torch.Tensor): Output from TopsideHead.
            hidden_rep (torch.Tensor): TKAN Latent Space.
            geometry_feats (torch.Tensor): Satellite geometry for GNSS delays.
            
        Returns:
            Tuple containing:
            - Net TEC
            - Electron Density
            - GNSS Delays (Dict)
            - Physics Intermediates (Dict)
        """
        seq_len = hidden_rep.size(1)
        
        # =========================================================================
        # 1. Shape & Tensor Alignment Validation
        # =========================================================================
        bottomside_tec = self.tensor_manager.validate_and_align(bottomside_tec, "Bottomside TEC", seq_len)
        predicted_topside_tec = self.tensor_manager.validate_and_align(predicted_topside_tec, "Topside TEC", seq_len)
        
        # =========================================================================
        # 2. Base Physics Verification (Topside >= 0)
        # =========================================================================
        predicted_topside_tec = self.tensor_manager.ensure_non_negative(predicted_topside_tec, "Topside TEC")
        
        # =========================================================================
        # 3. Base Net TEC Reconstruction
        # =========================================================================
        base_net_tec = bottomside_tec + predicted_topside_tec
        
        # =========================================================================
        # 4. Neural Net TEC Correction
        # =========================================================================
        net_tec = self.nettec_head(base_net_tec, hidden_rep)
        net_tec = self.tensor_manager.validate_and_align(net_tec, "Net TEC", seq_len)
        
        # Constraint: Net TEC must be greater than or equal to Bottomside TEC
        net_tec = self.tensor_manager.enforce_monotonic_increase(bottomside_tec, net_tec, "Net TEC (vs Bottomside)")
        
        # =========================================================================
        # 5. Electron Density Estimation
        # =========================================================================
        electron_density = self.electron_density_head(net_tec, hidden_rep)
        electron_density = self.tensor_manager.validate_and_align(electron_density, "Electron Density", seq_len)
        electron_density = self.tensor_manager.ensure_non_negative(electron_density, "Electron Density")
        
        # =========================================================================
        # 6. GNSS Delay Estimation
        # =========================================================================
        gnss_delays = self.gnss_delay_head(net_tec, electron_density, hidden_rep, geometry_feats)
        for key, delay_tensor in gnss_delays.items():
            delay_tensor = self.tensor_manager.validate_and_align(delay_tensor, f"GNSS {key}", seq_len)
            gnss_delays[key] = self.tensor_manager.ensure_non_negative(delay_tensor, f"GNSS {key}")
            
        # =========================================================================
        # 7. Intermediate Reporting & Debug Mode
        # =========================================================================
        if self.debug_mode:
            self._print_debug_report(bottomside_tec, predicted_topside_tec, net_tec, electron_density, gnss_delays['vertical_delay'])
            
        physics_intermediates = {
            "bottomside_tec_input": bottomside_tec,
            "predicted_topside_tec": predicted_topside_tec,
            "base_net_tec": base_net_tec
        }
        
        return net_tec, electron_density, gnss_delays, physics_intermediates

    def _print_debug_report(self, b_tec, t_tec, n_tec, e_den, d_delay):
        print("\n=============================")
        print("Physics Constraint Engine Report")
        print("=============================")
        print(f"Bottomside TEC    | Shape {tuple(b_tec.shape)}")
        print(f"Topside TEC       | Shape {tuple(t_tec.shape)}")
        print(f"Net TEC           | Shape {tuple(n_tec.shape)}")
        print(f"Electron Density  | Shape {tuple(e_den.shape)}")
        print(f"Delay             | Shape {tuple(d_delay.shape)}")
        print("=============================\n")
