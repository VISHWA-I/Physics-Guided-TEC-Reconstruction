import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

class VisualizationEngine:
    """
    Generates publication-ready DPI outputs.
    Avoids heavy external dependencies like specialized Taylor Diagram packages
    by implementing custom matplotlib logic where necessary.
    """
    
    def __init__(self, output_dir: str = "evaluation_reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set publication styling
        sns.set_theme(style="whitegrid", context="paper")
        
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Prediction vs Actual", filename: str = "pred_vs_actual.png"):
        """Scatter plot with a perfect 1:1 line."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, s=10, c='blue')
        
        # 1:1 Line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="1:1 Perfect Fit")
        
        plt.xlabel("Observed TEC (TECU)")
        plt.ylabel("Predicted TEC (TECU)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
        
    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str = "residuals.png"):
        """Histogram and KDE of residuals."""
        residuals = y_pred - y_true
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, bins=50, color='purple')
        plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
        plt.xlabel("Prediction Error (TECU)")
        plt.ylabel("Frequency")
        plt.title("Residual Error Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
        
    def plot_radar_chart(self, categories: list, values: list, baseline_values: Optional[list] = None, filename: str = "radar_chart.png"):
        """Radar chart for multi-dimensional scientific scoring."""
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories)
        
        # Plot Data
        val = values + values[:1]
        ax.plot(angles, val, linewidth=2, linestyle='solid', label="Hybrid Mamba-TKAN")
        ax.fill(angles, val, 'b', alpha=0.1)
        
        if baseline_values:
            b_val = baseline_values + baseline_values[:1]
            ax.plot(angles, b_val, linewidth=2, linestyle='dashed', label="Baseline", color='red')
            
        plt.title("Model Robustness Profile")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
