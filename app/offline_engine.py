import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from app.model_loader import ModelLoader
from app.prediction_engine import PredictionEngine
from app.delay_calculator import DelayCalculator
from app.confidence_engine import ConfidenceEngine
from app.anomaly_detector import AnomalyDetector
from app.storm_monitor import StormMonitor
from app.quality_score import QualityScore
from app.export_manager import ExportManager
from app.report_generator import OperationalReportGenerator

from utils.logger import get_model_logger
logger = get_model_logger("OfflineEngine")

class OfflineEngine:
    """
    Master operational engine. Wraps all application logic into a single cohesive pipeline
    for processing offline sliding windows autonomously.
    """
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None, output_dir: str = "offline_outputs"):
        logger.info("Initializing Operational Offline Engine...")
        
        # Core ML
        self.model, self.device = ModelLoader.load(config_path, checkpoint_path)
        self.pred_engine = PredictionEngine(self.model, self.device)
        
        # Handlers
        self.delay_calc = DelayCalculator()
        self.conf_engine = ConfidenceEngine()
        self.anomaly_detector = AnomalyDetector()
        self.storm_monitor = StormMonitor()
        self.quality_scorer = QualityScore()
        self.export_manager = ExportManager(f"{output_dir}/exports")
        self.report_gen = OperationalReportGenerator(f"{output_dir}/reports")
        
        logger.info("Offline Engine ready.")
        
    def process_window(self, window_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Processes a single pre-synchronized data window end-to-end.
        """
        # 1. Run Inference
        preds = self.pred_engine.predict(**window_inputs)
        
        # 2. Extract Delays
        delays = self.delay_calc.process_delays(preds)
        
        # 3. Assess Confidence
        confidence = self.conf_engine.evaluate(preds)
        
        # 4. Monitor Storms (Assume storm_feats is passed and Kp/Dst are idx 0,1)
        # Mocking extraction for operational robustness
        storm_feats = window_inputs.get("storm_feats")
        if storm_feats is not None:
            kp = storm_feats[:, 0].numpy()
            dst = storm_feats[:, 1].numpy()
            storm_state = self.storm_monitor.monitor(kp, dst)
        else:
            storm_state = self.storm_monitor.monitor(0.0, 0.0)
            
        # 5. Detect Anomalies
        anomalies = self.anomaly_detector.detect(
            preds["topside_tec"], preds["net_tec"], preds["electron_density"]
        )
        
        # 6. Compute Operational Readiness
        quality = self.quality_scorer.calculate(confidence, anomalies, storm_state)
        
        # 7. Package Results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result_payload = {
            "timestamp": timestamp,
            "predictions": {k: v.numpy() for k, v in preds.items()},
            "delays": delays,
            "confidence": confidence,
            "storm_state": storm_state,
            "anomalies": anomalies,
            "quality_score": quality
        }
        
        # 8. Generate Automated Exports & Reports
        # Convert tensors to lists for JSON export
        clean_preds = {k: np.squeeze(v.numpy()).tolist() if isinstance(v, torch.Tensor) else np.squeeze(v) for k, v in preds.items()}
        self.export_manager.export(clean_preds, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.report_gen.generate(
            timestamp=timestamp,
            quality_score=quality,
            storm_state=storm_state,
            anomalies=anomalies,
            filename=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        
        return result_payload
