"""Neural memory layers for the Pinecone Agent Harness.

Layers implemented:
  - Layer 2: GNN Score Diffusion Reranking (parameter-free)
  - Layer 3: Synaptic Associative Memory (Hebbian + spreading activation)

Higher layers (MemRL, Neuro-Symbolic KG, Neural Compression, DHM)
are stubbed with clean interfaces for progressive adoption.
"""

from .gnn_rerank import gnn_score_diffusion
from .synaptic import SynapticLayer
from .pipeline import NeuralRecallPipeline

__all__ = [
    "gnn_score_diffusion",
    "SynapticLayer",
    "NeuralRecallPipeline",
]
