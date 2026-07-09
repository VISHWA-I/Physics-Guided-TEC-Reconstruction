import numpy as np
from scipy import stats
from typing import Dict, Tuple

class StatisticalTests:
    """
    Performs rigorous statistical significance testing for publication.
    """
    
    @staticmethod
    def paired_t_test(y_err_model1: np.ndarray, y_err_model2: np.ndarray) -> Tuple[float, float]:
        """
        Parametric test for significance. Returns (t-statistic, p-value).
        """
        return stats.ttest_rel(y_err_model1, y_err_model2)
        
    @staticmethod
    def wilcoxon_test(y_err_model1: np.ndarray, y_err_model2: np.ndarray) -> Tuple[float, float]:
        """
        Non-parametric test for significance. Better for non-normal error distributions.
        Returns (w-statistic, p-value).
        """
        return stats.wilcoxon(y_err_model1, y_err_model2)
        
    @staticmethod
    def cohens_d(y_err_model1: np.ndarray, y_err_model2: np.ndarray) -> float:
        """
        Calculates effect size (Cohen's d).
        """
        diff = y_err_model1 - y_err_model2
        return float(np.mean(diff) / (np.std(diff) + 1e-8))
        
    @staticmethod
    def friedman_test(error_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Non-parametric test for comparing > 2 models.
        Input: error_matrix of shape (num_samples, num_models)
        Returns: (statistic, p-value)
        """
        # transpose so unpacks as *args (each row is a model's errors)
        return stats.friedmanchisquare(*error_matrix.T)
        
    @staticmethod
    def compute_all_tests(our_preds: np.ndarray, base_preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Runs the full suite against a single baseline.
        """
        our_err = np.abs(our_preds - targets)
        base_err = np.abs(base_preds - targets)
        
        t_stat, t_p = StatisticalTests.paired_t_test(our_err, base_err)
        w_stat, w_p = StatisticalTests.wilcoxon_test(our_err, base_err)
        d = StatisticalTests.cohens_d(our_err, base_err)
        
        return {
            "T_Test_p_value": float(t_p),
            "Wilcoxon_p_value": float(w_p),
            "Cohens_D": float(d),
            "Significant_05": bool(w_p < 0.05)
        }
