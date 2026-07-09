import shutil
from pathlib import Path
from datetime import datetime

class ExperimentArchive:
    """
    Zips up the entire benchmarking run into a self-contained, timestamped artifact.
    """
    
    @staticmethod
    def archive_results(source_dir: str, archive_dir: str = "archives"):
        """
        Compresses the source_dir (usually benchmark_reports) into a ZIP file.
        """
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = archive_path / f"benchmark_run_{timestamp}"
        
        shutil.make_archive(str(zip_name), 'zip', source_dir)
        return f"{zip_name}.zip"
