import torch
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any

from models.model_config import ModelConfig
from app.model_loader import ModelLoader

# Import our benchmarking modules
from benchmark.baseline_loader import BaselineLoader
from benchmark.ablation import AblationStudy
from benchmark.statistical_tests import StatisticalTests
from benchmark.complexity import ComplexityProfiler
from benchmark.latency import LatencyProfiler
from benchmark.memory_profile import MemoryProfiler
from benchmark.publication_tables import PublicationTables
from benchmark.publication_figures import PublicationFigures
from benchmark.reproducibility import ReproducibilityTracker
from benchmark.experiment_archive import ExperimentArchive
from benchmark.paper_report import PaperReportGenerator
from evaluation.metrics import ScientificMetrics

from utils.logger import get_model_logger
logger = get_model_logger("BenchmarkEngine")

class BenchmarkEngine:
    """
    Master Orchestrator for Phase 6.
    Runs comparisons against all baselines, conducts ablation studies, calculates statistics,
    and generates LaTeX/EPS publication artifacts.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, data_path: str, output_dir: str = "benchmark_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        
        logger.info("Initializing Publication Benchmark Engine...")
        self.model, self.device = ModelLoader.load(config_path, checkpoint_path)
        self.baseline_loader = BaselineLoader()
        self.ablation_engine = AblationStudy(self.model)
        self.fig_gen = PublicationFigures(str(self.output_dir / "figures"))
        
    def run_suite(self, dummy_inputs: Dict[str, torch.Tensor], target_tec: np.ndarray):
        """
        Executes the entire benchmarking and publication pipeline.
        """
        logger.info("Starting Benchmarking Suite...")
        
        # 1. Evaluate our Hybrid Model
        with torch.no_grad():
            our_output = self.model(**{k: v.to(self.device) for k, v in dummy_inputs.items()})
            our_preds = our_output.topside_tec.cpu().numpy().flatten()
            
        metrics_calc = ScientificMetrics()
        our_metrics = metrics_calc.compute(target_tec, our_preds)
        
        # 2. Load and Evaluate Baselines
        logger.info("Loading baseline predictions...")
        baselines = self.baseline_loader.load_all(target_length=len(target_tec))
        
        baseline_results = {"Hybrid_Mamba_TKAN": our_metrics}
        std_devs = {"Hybrid_Mamba_TKAN": np.std(our_preds)}
        correlations = {"Hybrid_Mamba_TKAN": our_metrics.get("Pearson_r", 1.0)}
        
        best_baseline_name = None
        best_baseline_preds = None
        best_rmse = float('inf')
        
        for name, preds in baselines.items():
            res = metrics_calc.compute(target_tec, preds)
            baseline_results[name] = res
            std_devs[name] = np.std(preds)
            correlations[name] = res.get("Pearson_r", 0.0)
            
            if res.get("RMSE", float('inf')) < best_rmse:
                best_rmse = res.get("RMSE", float('inf'))
                best_baseline_name = name
                best_baseline_preds = preds
                
        # 3. Statistical Testing vs Best Baseline
        logger.info(f"Running Statistical Tests against best baseline ({best_baseline_name})...")
        if best_baseline_preds is not None:
            stats = StatisticalTests.compute_all_tests(our_preds, best_baseline_preds, target_tec)
        else:
            stats = {}
            
        # 4. Computational Profiling
        logger.info("Running Computational Profiling...")
        dummy_gpu = {k: v.to(self.device) for k, v in dummy_inputs.items()}
        comp_profile = ComplexityProfiler.profile(self.model, dummy_gpu)
        latency_profile = LatencyProfiler.profile_inference(self.model, dummy_gpu, num_runs=5)
        mem_profile = MemoryProfiler.profile(self.model, dummy_gpu)
        
        full_comp_profile = {**comp_profile, **latency_profile, **mem_profile}
        
        # 5. Reproducibility Check
        logger.info("Capturing Reproducibility State...")
        reprod = ReproducibilityTracker.capture_environment(self.checkpoint_path, self.data_path)
        
        # 6. Generate Publication Artifacts (Tables/Figures/Reports)
        logger.info("Generating LaTeX tables and EPS figures...")
        latex_str = PublicationTables.generate_latex_table(baseline_results)
        with open(self.output_dir / "table_baselines.tex", "w", encoding="utf-8") as f:
            f.write(latex_str)
            
        ref_std = np.std(target_tec)
        self.fig_gen.plot_taylor_diagram_approximation(std_devs, correlations, ref_std)
        
        error_dict = {
            "Hybrid Model": np.abs(our_preds - target_tec),
        }
        if best_baseline_preds is not None:
             error_dict[best_baseline_name] = np.abs(best_baseline_preds - target_tec)
        self.fig_gen.plot_box_whisker(error_dict)
        
        # Final Report
        PaperReportGenerator.generate(baseline_results, stats, full_comp_profile, reprod, str(self.output_dir))
        
        # 7. Zip the Archive
        logger.info("Archiving Experiment...")
        zip_path = ExperimentArchive.archive_results(str(self.output_dir))
        logger.info(f"Phase 6 Benchmark Suite Complete! Archive created at: {zip_path}")
