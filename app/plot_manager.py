import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

class PlotManager:
    """
    Generates standard operational plots for the Dashboard and Reports.
    """
    
    @staticmethod
    def create_timeline_plot(timestamps: np.ndarray, 
                             predictions: np.ndarray, 
                             lower_bound: np.ndarray, 
                             upper_bound: np.ndarray,
                             title: str = "TEC Prediction Timeline"):
        """
        Creates a time-series plot with confidence intervals.
        Returns the matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot Confidence Band
        ax.fill_between(timestamps, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence')
        
        # Plot Prediction
        ax.plot(timestamps, predictions, color='red', lw=2, label='Predicted')
        
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("TEC (TECU)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        return fig
        
    @staticmethod
    def create_delay_plot(delays: Dict[str, np.ndarray], timestamps: np.ndarray):
        """
        Plots multi-constellation delays.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for name, delay_arr in delays.items():
            if "Delay" in name:
                ax.plot(timestamps, delay_arr, label=name.replace('_', ' '))
                
        ax.set_title("Multi-Constellation GNSS Delays")
        ax.set_xlabel("Time")
        ax.set_ylabel("Delay")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        return fig
