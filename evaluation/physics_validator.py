import numpy as np
from typing import Dict

class PhysicsValidator:
    """
    Validates model predictions against physical laws of the Ionosphere/Plasmasphere.
    Computes a 'Physics Consistency Score' reflecting how well the model obeys physics.
    """
    
    @staticmethod
    def validate(topside_tec: np.ndarray, 
                 bottomside_tec: np.ndarray, 
                 net_tec: np.ndarray, 
                 electron_density: np.ndarray) -> Dict[str, float]:
        """
        Runs validation checks across all physical constraints.
        Returns percentage (0-100) of predictions that satisfy the constraints.
        """
        topside = np.asarray(topside_tec).flatten()
        bottomside = np.asarray(bottomside_tec).flatten()
        net = np.asarray(net_tec).flatten()
        density = np.asarray(electron_density).flatten()
        
        total_samples = len(topside)
        if total_samples == 0:
            return {}
            
        results = {}
        
        # 1. Positivity Constraints
        # Physical quantities cannot be negative
        topside_positive_pct = np.sum(topside >= 0) / total_samples * 100
        density_positive_pct = np.sum(density >= 0) / total_samples * 100
        
        results["Topside_Positivity_Pct"] = float(topside_positive_pct)
        results["Density_Positivity_Pct"] = float(density_positive_pct)
        
        # 2. Net TEC Mass Conservation
        # Net TEC should approximately equal Topside + Bottomside.
        # Allow a 15% deviation margin for unmodeled plasmaspheric tails.
        analytical_net = topside + bottomside
        deviation = np.abs(net - analytical_net) / (np.abs(analytical_net) + 1e-8)
        mass_conservation_pct = np.sum(deviation <= 0.15) / total_samples * 100
        
        results["Mass_Conservation_Pct"] = float(mass_conservation_pct)
        
        # 3. Smoothness (Temporal First Derivative)
        # Highly unrealistic jumps (e.g. > 50% change in one timestep) are penalized.
        # Assuming sequential data for simplicity of approximation here.
        if len(topside) > 1:
            diffs = np.abs(topside[1:] - topside[:-1])
            max_allowed_jump = np.abs(topside[:-1]) * 0.5 + 5.0 # 50% or 5 TECU absolute
            smoothness_pct = np.sum(diffs <= max_allowed_jump) / (total_samples - 1) * 100
        else:
            smoothness_pct = 100.0
            
        results["Temporal_Smoothness_Pct"] = float(smoothness_pct)
        
        # Overall Physics Consistency Score (Weighted Average)
        overall_score = (
            0.30 * topside_positive_pct +
            0.30 * density_positive_pct +
            0.25 * mass_conservation_pct +
            0.15 * smoothness_pct
        )
        results["Overall_Physics_Score"] = float(overall_score)
        
        return results
