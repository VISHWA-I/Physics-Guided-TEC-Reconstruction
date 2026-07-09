import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

class BaselineLoader:
    """
    Ingests and standardizes baseline predictions from various models (empirical and ML)
    for direct benchmarking against the Hybrid Mamba-TKAN model.
    """
    
    def __init__(self, data_dir: str = "baselines"):
        self.data_dir = Path(data_dir)
        # Dictionary mapping model names to their file paths (mock setup for offline evaluation)
        self.baseline_registry = {
            "IRI-Plas": "iri_plas_preds.csv",
            "NeQuick": "nequick_preds.csv",
            "IRI-2016": "iri_2016_preds.csv",
            "Persistence": "persistence_preds.csv",
            "Linear_Regression": "lr_preds.csv",
            "Random_Forest": "rf_preds.csv",
            "XGBoost": "xgboost_preds.csv",
            "LSTM": "lstm_preds.csv",
            "GRU": "gru_preds.csv",
            "BiLSTM": "bilstm_preds.csv",
            "TCN": "tcn_preds.csv",
            "Transformer": "transformer_preds.csv",
            "PatchTST": "patchtst_preds.csv",
            "TiDE": "tide_preds.csv",
            "N-HiTS": "nhits_preds.csv",
            "Mamba": "mamba_preds.csv",
            "iTransformer": "itransformer_preds.csv",
            "TKAN": "tkan_preds.csv",
            "Physics_free_Mamba": "pf_mamba_preds.csv",
            "Physics_free_Transformer": "pf_transformer_preds.csv"
        }
        
    def load_baseline(self, model_name: str) -> np.ndarray:
        """
        Loads baseline prediction array. Mocks the array generation if file does not exist 
        to ensure the benchmarking framework compiles and runs for testing.
        """
        if model_name not in self.baseline_registry:
            raise ValueError(f"Unknown baseline: {model_name}")
            
        filepath = self.data_dir / self.baseline_registry[model_name]
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            return df['predicted_tec'].values
        else:
            # For testing/compilation purposes, return a mock random array
            # In a real environment, this would raise FileNotFoundError
            print(f"[BaselineLoader] Warning: {filepath} not found. Returning mock data.")
            return np.random.rand(128) * 30.0 # Mock TEC values
            
    def load_all(self, target_length: int) -> Dict[str, np.ndarray]:
        """
        Loads all configured baselines and enforces shape matching.
        """
        baselines = {}
        for name in self.baseline_registry.keys():
            data = self.load_baseline(name)
            if len(data) >= target_length:
                baselines[name] = data[:target_length]
            else:
                # Pad with NaNs if too short (handled by metrics engine)
                padded = np.pad(data, (0, target_length - len(data)), constant_values=np.nan)
                baselines[name] = padded
        return baselines
