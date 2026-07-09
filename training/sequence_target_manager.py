import torch

class ShapeMismatchError(Exception):
    """Raised when prediction and target shapes do not perfectly match."""
    pass

class SequenceTargetManager:
    """
    Dynamically generates sequence targets from scalar or partial sequence labels.
    Ensures all prediction targets conform to (batch, sequence_length, 1).
    Prepares targets for future Multi-Horizon forecasting without breaking existing models.
    """
    
    def __init__(self, mode: str = "sequence_to_sequence", forecast_horizon_steps: int = 0):
        """
        Args:
            mode (str): "sequence_to_sequence" or "sequence_to_one".
            forecast_horizon_steps (int): Infrastructure for future forecasting. Defaults to 0 (nowcast).
        """
        self.mode = mode
        self.forecast_horizon_steps = forecast_horizon_steps
        
        if self.mode not in ["sequence_to_sequence", "sequence_to_one"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sequence_to_sequence' or 'sequence_to_one'.")

    def generate_targets(self, y_raw: torch.Tensor, bottomside_tec_seq: torch.Tensor) -> dict:
        """
        Generates sequence targets for all tasks given a raw label (Topside TEC) and inputs.
        
        Args:
            y_raw (torch.Tensor): The raw dataset label. Usually shape (Batch,) or (Batch, 1).
            bottomside_tec_seq (torch.Tensor): The sequence bottomside TEC (Batch, Seq, 1).
            
        Returns:
            dict: Targets mapped to "topside", "net", "density", "delays". 
                  All shapes are guaranteed to be (Batch, Seq, 1) in sequence_to_sequence mode.
        """
        batch_size, seq_len, _ = bottomside_tec_seq.shape
        
        # Standardize y_raw to (Batch, 1)
        if y_raw.dim() == 1:
            y_raw = y_raw.unsqueeze(1)
            
        if self.mode == "sequence_to_one":
            # Target is just the last step
            topside_target = y_raw
            # We must map bottomside scalar to reconstruct net tec scalar
            bottomside_scalar = bottomside_tec_seq[:, -1, :]
            net_target = bottomside_scalar + topside_target
        else:
            # sequence_to_sequence mode (Default)
            # We broadcast the label backward through the sequence window as the target sequence
            # In a true seq-to-seq dataset, y_raw would be (Batch, Seq). 
            # Since our dataset gives a single scalar label per window, we broadcast it to simulate sequence loss.
            topside_target = y_raw.unsqueeze(1).expand(-1, seq_len, 1)
            net_target = bottomside_tec_seq + topside_target
            
        # Mock physics correlations for auxiliary targets (For model graph completion)
        # In a fully labeled dataset, we would extract these directly.
        density_target = net_target * 0.1
        delay_target = net_target * 0.162
        
        return {
            "topside": topside_target.float(),
            "net": net_target.float(),
            "density": density_target.float(),
            "delay": delay_target.float()
        }
