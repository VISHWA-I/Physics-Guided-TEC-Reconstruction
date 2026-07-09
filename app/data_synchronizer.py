import pandas as pd
import numpy as np
from typing import Dict, Any

from utils.logger import get_model_logger

logger = get_model_logger("DataSynchronizer")

class DataSynchronizer:
    """
    Synchronizes heterogeneous offline datasets (GIRO observations + NASA OMNIWeb).
    Handles alignment, missing data interpolation, and windowing.
    """
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        
    def synchronize(self, giro_df: pd.DataFrame, omni_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges GIRO ionosonde data with OMNI solar wind data on timestamps.
        """
        logger.info("Synchronizing GIRO and OMNI datasets...")
        
        # Ensure timestamp is datetime and set as index
        if 'timestamp' in giro_df.columns:
            giro_df['timestamp'] = pd.to_datetime(giro_df['timestamp'])
            giro_df.set_index('timestamp', inplace=True)
            
        if 'timestamp' in omni_df.columns:
            omni_df['timestamp'] = pd.to_datetime(omni_df['timestamp'])
            omni_df.set_index('timestamp', inplace=True)
            
        # Outer join to align timestamps perfectly
        merged = giro_df.join(omni_df, how='outer')
        
        # Interpolate minor gaps linearly (limit to prevent hallucinating huge gaps)
        merged.interpolate(method='time', limit=3, inplace=True)
        
        # Forward fill remaining small gaps
        merged.ffill(limit=2, inplace=True)
        
        # Drop rows that still have NaNs
        final_df = merged.dropna()
        
        logger.info(f"Synchronization complete. Output shape: {final_df.shape}")
        return final_df
        
    def create_sliding_windows(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        Creates rolling sequences for the temporal encoder.
        """
        data = df[feature_cols].values
        num_windows = len(data) - self.window_size + 1
        
        if num_windows <= 0:
            raise ValueError("Dataset is smaller than the required window size.")
            
        windows = np.array([data[i:i+self.window_size] for i in range(num_windows)])
        return windows
