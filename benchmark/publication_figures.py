import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PublicationFigures:
    """
    Generates 300 DPI EPS/PNG figures suitable for strict journal submission requirements.
    Implements a custom Polar coordinate Taylor Diagram approximation.
    """
    
    def __init__(self, output_dir: str = "benchmark_reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use a style available everywhere
        plt.style.use('default')
        
    def plot_taylor_diagram_approximation(self, std_devs: dict, correlations: dict, ref_std: float, filename: str = "taylor_diagram.png"):
        """
        Creates an approximate Taylor Diagram using raw Polar projections.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        
        # Transform correlation to polar angle (0 to pi/2)
        # R = 1 implies angle = 0, R = 0 implies angle = pi/2
        angles = {k: np.arccos(v) for k, v in correlations.items()}
        
        # Plot reference point on x-axis (angle 0)
        ax.plot(0, ref_std, 'k*', markersize=15, label='Observation')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(std_devs)))
        for idx, (model_name, std) in enumerate(std_devs.items()):
            angle = angles.get(model_name, np.pi/2)
            ax.plot(angle, std, 'o', markersize=10, label=model_name.replace("_", " "), color=colors[idx])
            
        # Format polar axes to look like a Taylor Diagram (quarter circle)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        
        # Add correlation grid lines manually (angles)
        ticks = [0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        ax.set_xticks(np.arccos(ticks))
        ax.set_xticklabels([str(t) for t in ticks])
        
        ax.set_title("Taylor Diagram", pad=20, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, format='png', bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace(".png", ".eps"), format='eps', bbox_inches='tight')
        plt.close()
        
    def plot_box_whisker(self, error_dict: dict, title: str = "Error Distribution", filename: str = "boxplot.png"):
        """
        Standard scientific box-and-whisker plot for RMSE or MAE comparisons.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(error_dict.keys())
        data = [error_dict[k] for k in labels]
        
        ax.boxplot(data, patch_artist=True, labels=[l.replace("_", "\n") for l in labels])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel("Absolute Error (TECU)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, format='png')
        plt.close()
