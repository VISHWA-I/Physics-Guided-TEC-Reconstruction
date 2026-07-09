import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Any

class DiscoveryVisualizer:
    """
    Generates high-quality analytical plots for the Scientific Discovery Module.
    """
    
    def __init__(self, output_dir: str = "scientific_discovery_reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('default')
        
    def plot_latent_umap(self, embedding: np.ndarray, labels: np.ndarray, filename: str = "latent_umap.png"):
        """
        Visualizes the clustered high-dimensional representations.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.7)
        ax.set_title("Latent Space Topology (UMAP)", fontweight='bold')
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(scatter, ax=ax, label="Cluster ID")
        
        fig.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, format='png')
        plt.close()
        
    def plot_relationship_graph(self, relationships: Dict[str, float], filename: str = "knowledge_graph.png"):
        """
        Generates a network graph showing driver-target coupling strength.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        G = nx.Graph()
        
        target = "Topside TEC"
        G.add_node(target, size=1000, color='red')
        
        for feature, score in relationships.items():
            if score > 0.1: # Only plot significant relations
                G.add_node(feature, size=500, color='lightblue')
                G.add_edge(target, feature, weight=score * 5)
                
        pos = nx.spring_layout(G, seed=42)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[target], node_size=1500, node_color='salmon', edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color='gray', alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
        
        ax.set_title("Ionospheric Driver Correlation Network", fontweight='bold')
        ax.axis('off')
        
        fig.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, format='png')
        plt.close()
