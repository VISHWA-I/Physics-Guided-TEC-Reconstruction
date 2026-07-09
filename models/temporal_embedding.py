import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from models.model_config import ModelConfig

class PhysicsAwareTemporalEmbedding(nn.Module):
    """
    Physics-Aware Dynamic Temporal Embedding.
    Injects sequence order, long-range representations, and physical temporal modulations.
    Supports automatic dynamic resizing.
    """
    def __init__(self, config: ModelConfig, initial_seq_len: int = 96):
        super().__init__()
        self.config = config
        self.d_model = config.embedding_dimension
        
        self.dynamic_resize = config.te_dynamic_resize
        self.max_seq_len = config.te_max_sequence_length
        self.use_learned = config.te_position_embedding in ["learned", "hybrid"]
        self.use_physics = config.te_physics_embedding
        
        self.current_seq_len = initial_seq_len if self.max_seq_len == "auto" else int(self.max_seq_len)
        
        # 1. Learned Positional Embedding
        if self.use_learned:
            self.embedding = nn.Embedding(self.current_seq_len, self.d_model)
            
        # 2. Physics & Temporal Projections
        if self.use_physics:
            # We will project physical parameters directly into the embedding dimension
            # Local Time (sin, cos) -> 2
            # Day of Year (sin, cos) -> 2
            # Solar Activity (F10.7, SSN, Kp, Dst, Ap) -> 5
            # Station Location (Lat, Lon) -> 2
            # Total Physics Dim = 11
            self.physics_dim = 11
            self.physics_projection = nn.Linear(self.physics_dim, self.d_model)
            
        # Precompute initial Sinusoidal PE
        self._register_sinusoidal(self.current_seq_len)
        
        self._print_status(initial_seq_len)

    def _register_sinusoidal(self, length: int):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(length, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def _expand_embeddings_if_needed(self, required_length: int, device: torch.device):
        if required_length <= self.current_seq_len:
            return
            
        if not self.dynamic_resize:
            raise RuntimeError(f"Sequence length exceeds positional embedding capacity.\n"
                               f"Current sequence length : {required_length}\n"
                               f"Embedding size : {self.current_seq_len}\n"
                               f"Increase max_sequence_length or enable dynamic resizing.")
                               
        # Expand Learned Embedding
        if self.use_learned:
            new_embedding = nn.Embedding(required_length, self.d_model).to(device)
            # Copy old weights
            new_embedding.weight.data[:self.current_seq_len] = self.embedding.weight.data
            self.embedding = new_embedding
            
        # Expand Sinusoidal buffer
        self._register_sinusoidal(required_length)
        self.pe = self.pe.to(device)
        
        self.current_seq_len = required_length

    def forward(self, x: torch.Tensor, geo_feats: torch.Tensor, storm_feats: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 1. Dynamic Check & Resize
        self._expand_embeddings_if_needed(seq_len, device)
        
        temporal_representation = torch.zeros_like(x)
        
        # 2. Learned Positional
        if self.use_learned:
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            temporal_representation += self.embedding(positions)
            
        # 3. Sinusoidal Positional
        temporal_representation += self.pe[:seq_len, :].unsqueeze(0)
        
        # 4. Physics Encoding
        if self.use_physics:
            # geo_feats: [Latitude, Longitude, DayOfYear, LocalTime] -> Indices 0, 1, 2, 3
            lat = geo_feats[:, 0:1]
            lon = geo_feats[:, 1:2]
            doy = geo_feats[:, 2:3]
            lt = geo_feats[:, 3:4]
            
            # storm_feats: [Kp, Dst, Ap, F10.7, SSN] -> Indices 0, 1, 2, 3, 4
            solar_geo = storm_feats[:, 0:5]
            
            # Cyclic encodings
            doy_sin = torch.sin(2 * math.pi * doy / 365.25)
            doy_cos = torch.cos(2 * math.pi * doy / 365.25)
            lt_sin = torch.sin(2 * math.pi * lt / 24.0)
            lt_cos = torch.cos(2 * math.pi * lt / 24.0)
            
            # Combine all physics: (Batch, 11)
            physics_vector = torch.cat([lt_sin, lt_cos, doy_sin, doy_cos, solar_geo, lat, lon], dim=-1)
            
            # Project to d_model: (Batch, d_model)
            physics_emb = self.physics_projection(physics_vector)
            
            # Broadcast across sequence: (Batch, 1, d_model)
            temporal_representation += physics_emb.unsqueeze(1)
            
        return x + temporal_representation

    def _print_status(self, seq: int):
        print("\n================================================")
        print("Physics-Aware Temporal Embedding")
        print("================================================")
        print("Embedding Type\nHybrid")
        print("Learned\nEnabled" if self.use_learned else "Learned\nDisabled")
        print("Sinusoidal\nEnabled")
        print("Local Time\nEnabled" if self.use_physics else "Local Time\nDisabled")
        print("Day Of Year\nEnabled" if self.use_physics else "Day Of Year\nDisabled")
        print("Solar Activity\nEnabled" if self.use_physics else "Solar Activity\nDisabled")
        print("Station Encoding\nEnabled" if self.use_physics else "Station Encoding\nDisabled")
        print(f"Dynamic Resize\n{'Enabled' if self.dynamic_resize else 'Disabled'}")
        print(f"Maximum Length\n{self.max_seq_len}")
        print(f"Current Window\n{seq}")
        print("Sampling\n15 min")
        print("Status\nREADY")
        print("================================================\n")

class FeatureTypeEmbedding(nn.Module):
    """
    Provides a distinct learnable embedding vector for each physical feature category.
    """
    def __init__(self, d_model: int, feature_names: List[str]):
        super().__init__()
        self.category_map = {"TEC": 0, "Solar": 1, "Geomagnetic": 2}
        self.embedding = nn.Embedding(len(self.category_map), d_model)
        
        feature_indices = []
        for feat in feature_names:
            if feat == "TEC":
                cat = "TEC"
            elif feat in ("F10.7", "SSN"):
                cat = "Solar"
            elif feat in ("Kp", "Ap", "Dst"):
                cat = "Geomagnetic"
            else:
                cat = "TEC" 
            feature_indices.append(self.category_map[cat])
            
        self.register_buffer('feature_indices', torch.tensor(feature_indices, dtype=torch.long))
        self.branch_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat_embs = self.embedding(self.feature_indices)
        branch_context = cat_embs.mean(dim=0)
        branch_context = self.branch_projection(branch_context)
        return x + branch_context
