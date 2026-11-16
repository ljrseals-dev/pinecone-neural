"""Neural memory layers for the Pinecone Agent Harness.

Implemented:
  - Layer 2: GNN Score Diffusion Reranking
  - Layer 3: Synaptic Associative Memory
  - Layer 4: Learned Utility Scoring (MemRL-style)
"""

from .gnn_rerank import gnn_score_diffusion
from .synaptic import SynapticLayer
from .utility import UtilityModel, hand_tuned_utility
from .pipeline import NeuralRecallPipeline

__all__ = [
    "gnn_score_diffusion",
    "SynapticLayer",
    "UtilityModel",
    "hand_tuned_utility",
    "NeuralRecallPipeline",
]
