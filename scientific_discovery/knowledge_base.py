import json
import os
from datetime import datetime
from typing import Dict, Any, List

class KnowledgeBase:
    """
    Persistent store for scientific discoveries. 
    Allows comparison of new observations against historical norms.
    """
    
    def __init__(self, db_path: str = "scientific_discovery/knowledge_base.json"):
        self.db_path = db_path
        self.kb = self._load()
        
    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"patterns": [], "storms": [], "hypotheses": []}
        
    def save_discovery(self, category: str, data: Any):
        """
        Commits a new discovery to the local KB.
        """
        if category not in self.kb:
            self.kb[category] = []
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.kb[category].append(entry)
        
        # Save to disk safely
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.kb, f, indent=4)
            
    def get_historical_mean(self) -> float:
        """Mock method for historical baseline retrieval."""
        return 15.0 # Mock TEC
        
    def get_historical_std(self) -> float:
        """Mock method for historical variance retrieval."""
        return 5.0
