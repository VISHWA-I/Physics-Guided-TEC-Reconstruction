from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

class ReportGenerator:
    """
    Compiles all scientific discoveries into human-readable HTML/Markdown formats.
    """
    
    @staticmethod
    def generate(discoveries: Dict[str, Any], hypotheses: List[Dict[str, Any]], output_dir: str = "scientific_discovery_reports"):
        """
        Writes the final automated discovery report.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "Automated_Scientific_Discovery.md"
        
        md = f"""# Automated Scientific Discovery Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Automated Hypotheses
The SDM AI has generated the following scientific hypotheses based on raw inference data:
"""
        for i, hyp in enumerate(hypotheses, 1):
            md += f"\n### Hypothesis {i}\n"
            md += f"> **{hyp['statement']}**\n\n"
            md += f"- **Confidence Score:** {hyp['confidence_score']:.1f} / 100\n"
            md += f"- **Supporting Evidence:** {hyp['supporting_evidence']}\n"
            
        md += """
## 2. Latent Topology Analysis
* **Distinct Physical States Found:** {distinct_states}
* **Novel / Unprecedented Events:** {novel_events}

## 3. Storm & Anomaly Classification
* **Concept Drift Detected:** {drift}

---
*Refer to the attached figures directory for UMAP latent topologies and relationship networks.*
"""
        # Format safely
        md = md.format(
            distinct_states=discoveries.get("latent_clusters", {}).get("num_distinct_states", 0),
            novel_events=discoveries.get("novelty", {}).get("num_novel_events", 0),
            drift='Yes' if discoveries.get("concept_drift", {}).get("is_drifting", False) else 'No'
        )
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
            
        return path
