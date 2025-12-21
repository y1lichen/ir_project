"""
STRUCT-REC: Movie Structural Retrieval System

Modules:
    - features: Feature extraction (sentiment arc, NetLSD)
    - retrieval: Search engine (DTW, topology search)
    - lightgcn: LightGCN recommendation model
"""

from .features import FeatureExtractor
from .retrieval import StructRetrieval
from .lightgcn import SimpleLightGCN, DataLoader
from .tfidf_similarity import QuerySimilarity
from .bert_similarity import BertQuerySimilarity

__all__ = [
    "FeatureExtractor",
    "StructRetrieval",
    "SimpleLightGCN",
    "DataLoader",
    "QuerySimilarity",
    "BertQuerySimilarity"
]
