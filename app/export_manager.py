import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
import scipy.io as sio

# Optional dependencies for specific formats
try:
    import h5py
except ImportError:
    h5py = None

class ExportManager:
    """
    Handles robust data exportation to standard scientific formats.
    """
    
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
    def export(self, data: Dict[str, Any], filename_base: str, formats: list = ["csv", "json"]):
        """
        Exports the provided dictionary of numpy arrays/scalars to specified formats.
        """
        # Convert all 1D arrays to a DataFrame for CSV/Excel
        try:
            df = pd.DataFrame({k: np.squeeze(v) for k, v in data.items() if isinstance(v, (list, np.ndarray))})
        except ValueError:
            # If shapes mismatch (e.g. some are scalars), just stringify the dict
            df = pd.DataFrame([data])
            
        for fmt in formats:
            fmt = fmt.lower()
            path = self.export_dir / f"{filename_base}.{fmt}"
            
            if fmt == "csv":
                df.to_csv(path, index=False)
            elif fmt == "xlsx" or fmt == "excel":
                df.to_excel(path, index=False)
            elif fmt == "json":
                # JSON requires native python types
                clean_data = {}
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        clean_data[k] = v.tolist()
                    else:
                        clean_data[k] = v
                with open(path, "w") as f:
                    json.dump(clean_data, f, indent=4)
            elif fmt == "mat":
                sio.savemat(str(path), data)
            elif fmt == "hdf5" or fmt == "h5":
                if h5py:
                    with h5py.File(path, 'w') as f:
                        for k, v in data.items():
                            f.create_dataset(k, data=v)
                else:
                    print("h5py not installed. Skipping HDF5 export.")
                    
        return f"Successfully exported to {self.export_dir}"
