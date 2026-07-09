from pathlib import Path
from typing import Dict, Any

class PaperReportGenerator:
    """
    Compiles the Markdown appendix for IEEE/Nature supplementary material.
    """
    
    @staticmethod
    def generate(metrics: dict, stats: dict, comp: dict, reprod: dict, output_dir: str = "benchmark_reports"):
        """
        Writes the final overarching paper supplementary document.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "IEEE_Supplementary_Material.md"
        
        md = f"""# Supplementary Material: Benchmarking Results
**Physics-Guided Multi-Branch Memory-Augmented Mamba–TKAN Network**

## 1. Reproducibility Provenance
* **Git Commit:** `{reprod.get("Git_Commit_Hash")}`
* **Model Checkpoint SHA-256:** `{reprod.get("Model_Weights_Hash")}`
* **Dataset SHA-256:** `{reprod.get("Dataset_Hash")}`
* **Environment:** {reprod.get("OS_Platform")} / Python {reprod.get("Python_Version").split(' ')[0]}

## 2. Computational Complexity & Latency
* **Total Parameters:** {comp.get("Total_Parameters_Millions", 0):.2f} M
* **Estimated GFLOPs:** {comp.get("Estimated_GFLOPs", 0):.2f}
* **Peak GPU Memory:** {comp.get("Peak_GPU_Memory_MB", 0):.2f} MB
* **Average Inference Latency:** {comp.get("Average_Inference_ms", 0):.2f} ms
* **Suitable for 4-minute operational cadence:** {'Yes' if comp.get("Suitable_for_4min_cadence") else 'No'}

## 3. Statistical Significance (vs. Best Baseline)
* **Paired t-test p-value:** {stats.get("T_Test_p_value", 1.0):.2e}
* **Wilcoxon signed-rank p-value:** {stats.get("Wilcoxon_p_value", 1.0):.2e}
* **Cohen's d (Effect Size):** {stats.get("Cohens_D", 0.0):.2f}
* **Statistically Significant ($\alpha=0.05$):** {'Yes' if stats.get("Significant_05") else 'No'}

---
*Refer to the attached `.tex` and `.eps` files in this archive for publication tables and figures.*
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
            
        return path
