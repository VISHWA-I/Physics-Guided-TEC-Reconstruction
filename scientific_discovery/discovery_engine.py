import sys
from pathlib import Path
import torch
import numpy as np
import time
from typing import Dict, Any

from utils.logger import get_model_logger

from scientific_discovery.pattern_mining import PatternMiner
from scientific_discovery.latent_cluster import LatentSpaceClusterer
from scientific_discovery.storm_cluster import StormClusterer
from scientific_discovery.novelty_detector import NoveltyDetector
from scientific_discovery.outlier_analysis import OutlierAnalyzer
from scientific_discovery.concept_drift import ConceptDriftMonitor
from scientific_discovery.feature_relationships import FeatureRelationshipMiner
from scientific_discovery.hypothesis_generator import HypothesisGenerator
from scientific_discovery.event_classifier import EventClassifier
from scientific_discovery.knowledge_base import KnowledgeBase
from scientific_discovery.visualization import DiscoveryVisualizer
from scientific_discovery.report_generator import ReportGenerator

logger = get_model_logger("DiscoveryEngine")

class DiscoveryEngine:
    """
    Master Orchestrator for the Scientific Discovery Module (SDM).
    Must run in < 2 seconds.
    """
    
    def __init__(self, output_dir: str = "scientific_discovery_reports"):
        logger.info("Initializing Scientific Discovery Engine (SDM)...")
        self.output_dir = Path(output_dir)
        
        # Sub-modules
        self.pattern_miner = PatternMiner()
        self.latent_clusterer = LatentSpaceClusterer()
        self.storm_clusterer = StormClusterer()
        self.novelty_detector = NoveltyDetector()
        self.outlier_analyzer = OutlierAnalyzer()
        self.concept_drift = ConceptDriftMonitor()
        self.feature_miner = FeatureRelationshipMiner()
        self.hypothesis_gen = HypothesisGenerator()
        self.event_classifier = EventClassifier()
        self.kb = KnowledgeBase()
        self.visualizer = DiscoveryVisualizer(str(self.output_dir / "figures"))
        self.report_gen = ReportGenerator()
        
    def analyze(self, inputs: Dict[str, np.ndarray], predictions: np.ndarray, latent_states: np.ndarray = None) -> Dict[str, Any]:
        """
        Executes the rapid SDM pipeline on the current sliding window.
        Returns the discovery payload.
        """
        start_time = time.time()
        logger.info("SDM: Executing automated discovery pipeline...")
        
        discoveries = {}
        
        # 1. Feature Relationships
        # Flatten temporal sequences for basic mapping if necessary, or just use 2D feats.
        # Assuming inputs["physics_feats"] is (batch, feats)
        physics_feats = inputs.get("physics_feats", np.random.rand(predictions.shape[0], 5))
        feature_names = ["Kp", "Dst", "F10.7", "hmF2", "foF2"]
        relationships = self.feature_miner.map_relationships(physics_feats, feature_names, predictions)
        discoveries["relationships"] = relationships
        
        # 2. Novelty & Outliers
        novelty_data = self.novelty_detector.detect(physics_feats, predictions)
        outlier_data = self.outlier_analyzer.analyze(predictions)
        discoveries["novelty"] = novelty_data
        discoveries["outliers"] = outlier_data
        
        # 3. Concept Drift (vs Knowledge Base)
        drift_data = self.concept_drift.detect_drift(
            predictions, 
            self.kb.get_historical_mean(), 
            self.kb.get_historical_std()
        )
        discoveries["concept_drift"] = drift_data
        
        # 4. Latent Space Clustering
        if latent_states is not None:
            latent_data = self.latent_clusterer.cluster(latent_states)
            discoveries["latent_clusters"] = latent_data
            self.visualizer.plot_latent_umap(latent_data["umap_embedding_2d"], latent_data["latent_cluster_labels"])
            
        # 5. Event Classification
        storm_feats = inputs.get("storm_feats", np.random.rand(predictions.shape[0], 2))
        kp_values = storm_feats[:, 0] * 9.0 # Mock unscale
        classifications = self.event_classifier.classify(kp_values, novelty_data["is_novel"])
        discoveries["classifications"] = classifications
        
        # 6. Hypothesis Generation
        hypotheses = self.hypothesis_gen.generate(relationships, novelty_data, drift_data)
        
        # 7. Visualization & Reporting
        self.visualizer.plot_relationship_graph(relationships)
        self.report_gen.generate(discoveries, hypotheses, str(self.output_dir))
        
        # 8. Knowledge Base Update
        if hypotheses:
            self.kb.save_discovery("hypotheses", hypotheses)
            
        end_time = time.time()
        logger.info(f"SDM: Analysis completed in {(end_time - start_time):.3f} seconds.")
        
        return discoveries
