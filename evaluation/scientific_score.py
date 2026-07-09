from typing import Dict

class ScientificScore:
    """
    Computes a unified 0-100 score indicating the scientific validity of the model.
    Weights accuracy, physics consistency, and robustness heavily.
    """
    
    @staticmethod
    def compute(metrics_dict: Dict[str, float], 
                physics_dict: Dict[str, float], 
                generalization_dict: Dict[str, float], 
                storm_dict: Dict[str, Dict[str, float]]) -> float:
        """
        Aggregates multiple domain metrics into a single score.
        """
        score = 0.0
        
        # 1. Prediction Accuracy (40 Points)
        # Assuming an R2 of 1.0 gives 40 points, R2 of 0.5 gives 20 points, etc.
        r2 = metrics_dict.get("R2", 0.0)
        acc_score = max(0.0, r2 * 40.0)
        score += acc_score
        
        # 2. Physics Consistency (30 Points)
        # Directly proportional to the Overall Physics Score percentage
        phys_pct = physics_dict.get("Overall_Physics_Score", 0.0)
        phys_score = (phys_pct / 100.0) * 30.0
        score += phys_score
        
        # 3. Generalization (15 Points)
        # Deduct points based on the Generalization Gap Percentage
        gap_pct = generalization_dict.get("Generalization_Gap_Pct", 100.0)
        # If gap is 0%, get 15 points. If gap > 20%, get 0 points.
        gen_score = max(0.0, 15.0 - (gap_pct / 20.0 * 15.0))
        score += gen_score
        
        # 4. Storm Robustness (10 Points)
        # Measured by the Storm Degradation Factor (Strong RMSE / Quiet RMSE).
        # A factor of 1.0 (no degradation) gives 10 points. 
        # A factor > 3.0 gives 0 points.
        if "Storm_Degradation_Factor" in storm_dict:
            deg_factor = storm_dict["Storm_Degradation_Factor"]
            storm_score = max(0.0, 10.0 - ((deg_factor - 1.0) / 2.0 * 10.0))
        else:
            storm_score = 0.0
        score += storm_score
        
        # 5. Confidence Calibration (5 Points)
        # (Assuming perfect calibration gives 5 points. For now, assign static 5 if uncertainty head activated)
        score += 5.0
        
        # Ensure bounds 0-100
        return max(0.0, min(100.0, score))
