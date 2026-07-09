import pandas as pd
from typing import Dict

class PublicationTables:
    """
    Generates LaTeX and Markdown tables directly from benchmark metric dictionaries.
    Conforms to IEEE TGRS format.
    """
    
    @staticmethod
    def generate_latex_table(results_dict: Dict[str, Dict[str, float]], caption: str = "Performance Comparison") -> str:
        """
        Converts a dictionary {ModelName: {Metric: Value}} into a LaTeX tabular string.
        """
        models = list(results_dict.keys())
        if not models:
            return ""
            
        metrics = list(results_dict[models[0]].keys())
        
        latex = f"\\begin{{table*}}[htbp]\n"
        latex += f"\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\begin{{tabular}}{{|l|{'c|'*len(metrics)}}}\n"
        latex += f"\\hline\n"
        
        # Header
        latex += "\\textbf{Model} & " + " & ".join([f"\\textbf{{{m}}}" for m in metrics]) + " \\\\\n"
        latex += f"\\hline\n"
        
        # Rows
        for model in models:
            row = [model.replace("_", "\\_")]
            for m in metrics:
                val = results_dict[model].get(m, 0.0)
                # Format smartly
                if val < 0.001:
                    row.append(f"{val:.2e}")
                else:
                    row.append(f"{val:.4f}")
            latex += " & ".join(row) + " \\\\\n"
            
        latex += f"\\hline\n"
        latex += f"\\end{{tabular}}\n"
        latex += f"\\end{{table*}}\n"
        
        return latex
