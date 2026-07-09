import numpy as np
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    from sklearn.decomposition import PCA
    
from sklearn.cluster import HDBSCAN
from typing import Dict, Any

class LatentSpaceClusterer:
    """
    Analyzes deep high-dimensional latent representations to discover hidden topological states.
    Uses UMAP for fast reduction (or PCA as fallback) and HDBSCAN for density clustering.
    """
    
    def __init__(self):
        if UMAP_AVAILABLE:
            self.reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        else:
            self.reducer = PCA(n_components=2, random_state=42)
        # Using sklearn's native HDBSCAN available in recent versions
        self.clusterer = HDBSCAN(min_cluster_size=5)
        
    def cluster(self, latent_vectors: np.ndarray) -> Dict[str, Any]:
        """
        Reduces and clusters latent states.
        """
        # Ensure 2D
        if len(latent_vectors.shape) > 2:
            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
            
        # Rapid dimensionality reduction
        embedding = self.reducer.fit_transform(latent_vectors)
        
        # Density clustering
        labels = self.clusterer.fit_predict(embedding)
        
        return {
            "umap_embedding_2d": embedding,
            "latent_cluster_labels": labels,
            "num_distinct_states": len(set(labels)) - (1 if -1 in labels else 0)
        }
