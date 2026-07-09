from typing import Dict, Any, List

class HypothesisGenerator:
    """
    Synthesizes natural language hypotheses based on empirical data mined from the pipeline.
    """
    
    @staticmethod
    def generate(relationships: Dict[str, float], novelty: Dict[str, Any], drift: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule-based heuristic generator.
        """
        hypotheses = []
        
        # 1. Feature Importance Hypothesis
        top_feature = list(relationships.keys())[0] if relationships else "Unknown"
        top_score = list(relationships.values())[0] if relationships else 0.0
        
        if top_score > 0.8:
            hypotheses.append({
                "statement": f"'{top_feature}' strongly dominates the nonlinear predictive mapping of Topside TEC.",
                "confidence_score": float(top_score * 100),
                "supporting_evidence": "Mutual Information Analysis"
            })
            
        # 2. Novelty Hypothesis
        num_novel = novelty.get("num_novel_events", 0)
        if num_novel > 0:
            hypotheses.append({
                "statement": f"Discovered {num_novel} unprecedented ionospheric signatures deviating from known physics boundaries.",
                "confidence_score": 85.0,
                "supporting_evidence": "Isolation Forest Anomaly Density"
            })
            
        # 3. Concept Drift Hypothesis
        if drift.get("is_drifting", False):
            hypotheses.append({
                "statement": "The global ionospheric state is exhibiting long-term statistical drift, likely indicating a seasonal or solar cycle phase transition.",
                "confidence_score": min(99.0, drift.get("drift_z_score", 0.0) * 20.0),
                "supporting_evidence": f"Z-Score Shift: {drift.get('drift_z_score'):.2f}"
            })
            
        return hypotheses
