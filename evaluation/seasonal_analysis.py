import numpy as np
from typing import Dict
from evaluation.metrics import ScientificMetrics

class SeasonalAnalysis:
    """
    Evaluates model performance stratified by Season.
    """
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, day_of_year: np.ndarray, is_northern_hemisphere: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Bins data by Season based on Day of Year (DOY) and Hemisphere.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        doy = np.asarray(day_of_year).flatten()
        is_nh = np.asarray(is_northern_hemisphere).flatten().astype(bool)
        
        results = {}
        metrics_calc = ScientificMetrics()
        
        # Approximate seasonal DOY boundaries (Northern Hemisphere)
        # Spring Equinox ~80 to ~172
        # Summer Solstice ~172 to ~266
        # Autumn Equinox ~266 to ~355
        # Winter Solstice ~355 to 365, 1 to 80
        
        # Helper lambda to define seasons considering hemisphere swap
        def get_season_mask(season_name: str) -> np.ndarray:
            if season_name == "Summer":
                nh_mask = is_nh & (doy >= 172) & (doy < 266)
                sh_mask = (~is_nh) & ((doy >= 355) | (doy < 80))
                return nh_mask | sh_mask
            elif season_name == "Winter":
                nh_mask = is_nh & ((doy >= 355) | (doy < 80))
                sh_mask = (~is_nh) & (doy >= 172) & (doy < 266)
                return nh_mask | sh_mask
            elif season_name == "Equinox":
                # Combine Spring and Autumn for general Equinox behavior
                nh_mask = is_nh & (((doy >= 80) & (doy < 172)) | ((doy >= 266) & (doy < 355)))
                sh_mask = (~is_nh) & (((doy >= 266) & (doy < 355)) | ((doy >= 80) & (doy < 172)))
                return nh_mask | sh_mask
            return np.zeros_like(doy, dtype=bool)

        summer_mask = get_season_mask("Summer")
        winter_mask = get_season_mask("Winter")
        equinox_mask = get_season_mask("Equinox")
        
        if np.any(summer_mask):
            results["Summer"] = metrics_calc.compute(y_true[summer_mask], y_pred[summer_mask])
        if np.any(winter_mask):
            results["Winter"] = metrics_calc.compute(y_true[winter_mask], y_pred[winter_mask])
        if np.any(equinox_mask):
            results["Equinox"] = metrics_calc.compute(y_true[equinox_mask], y_pred[equinox_mask])
            
        return results
