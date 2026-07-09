from pathlib import Path
from typing import Dict, Any

class OperationalReportGenerator:
    """
    Generates structured Markdown/HTML operational reports summarizing
    the current Digital Twin state, storm categorization, and anomalies.
    """
    
    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, 
                 timestamp: str,
                 quality_score: float,
                 storm_state: Dict[str, str],
                 anomalies: Dict[str, Any],
                 filename: str = "operational_report.md") -> Path:
        """
        Compiles the report.
        """
        path = self.report_dir / filename
        
        md = f"""# Operational Digital Twin Report
**Timestamp:** {timestamp}

## Overall Status
**Operational Readiness Score:** `{quality_score:.2f} / 100`

### 🌪️ Space Weather Monitor
* **Warning Level:** `{storm_state.get('Warning_Level', 'UNKNOWN')}`
* **Category:** {storm_state.get('Storm_Category', 'UNKNOWN')}
* **Max Kp:** {storm_state.get('Max_Kp', 0)} | **Min Dst:** {storm_state.get('Min_Dst', 0)}

### ⚠️ Anomaly Detection
"""
        
        if anomalies.get("is_anomalous"):
            md += "**WARNING: Anomalies Detected in Predictions!**\n"
            if anomalies.get("negative_topside_detected"): md += "- [x] Unphysical Negative Topside TEC\n"
            if anomalies.get("extreme_tec_detected"): md += "- [x] Extreme TEC Values (> Hard Cap)\n"
            if anomalies.get("massive_jump_detected"): md += "- [x] Massive Discontinuous Temporal Jump\n"
        else:
            md += "**ALL CLEAR: No prediction anomalies detected.**\n"
            
        md += """
---
*Generated automatically by the Hybrid Mamba-TKAN Offline Engine.*
"""
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
            
        return path
