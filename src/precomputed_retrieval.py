"""
Precomputed Distance Matrix Retrieval

This module provides a fast retrieval system by precomputing all pairwise
distances once, then using simple array lookups for queries.

Speedup: ~100x faster than on-the-fly DTW computation
"""

import numpy as np
import pickle
import os
from tqdm import tqdm
from fastdtw import fastdtw


class PrecomputedRetrieval:
    """
    Fast retrieval using precomputed distance matrices.

    Instead of computing DTW for every query, we precompute all pairwise
    distances once and store them in matrices. This reduces query time
    from O(N * DTW_cost) to O(N) for sorting.
    """

    def __init__(self, feature_db, cache_path='data/distance_cache.pkl'):
        """
        Initialize with precomputed distance matrices.

        Args:
            feature_db: Dict of {movie_id: {'arc': [...], 'netlsd': [...]}}
            cache_path: Path to save/load precomputed distances
        """
        self.db = feature_db
        self.cache_path = cache_path

        # Create ordered list of movie IDs for matrix indexing
        self.movie_ids = list(feature_db.keys())
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.n_movies = len(self.movie_ids)

        # Load or compute distance matrices
        if os.path.exists(cache_path):
            print(f"[INFO] Loading precomputed distances from {cache_path}...")
            self._load_cache()
        else:
            print(f"[INFO] Computing distance matrices (this may take ~30 minutes)...")
            self._compute_all_distances()
            self._save_cache()

    def _compute_all_distances(self):
        """Compute all pairwise DTW and NetLSD distances."""
        n = self.n_movies

        # Initialize matrices (use float32 to save memory)
        self.dtw_matrix = np.zeros((n, n), dtype=np.float32)
        self.netlsd_matrix = np.zeros((n, n), dtype=np.float32)

        # Compute DTW distances (the slow part)
        print("[INFO] Computing DTW distance matrix...")
        for i in tqdm(range(n), desc="DTW Matrix"):
            arc_i = self.db[self.movie_ids[i]]['arc']

            for j in range(i + 1, n):  # Only compute upper triangle
                arc_j = self.db[self.movie_ids[j]]['arc']

                # Handle missing arcs
                if arc_i is None or arc_j is None or len(arc_i) == 0 or len(arc_j) == 0:
                    dist = float('inf')
                else:
                    dist, _ = fastdtw(arc_i, arc_j, radius=1, dist=lambda x, y: abs(x - y))

                self.dtw_matrix[i, j] = dist
                self.dtw_matrix[j, i] = dist  # Symmetric

        # Compute NetLSD distances (faster)
        print("[INFO] Computing NetLSD distance matrix...")
        for i in tqdm(range(n), desc="NetLSD Matrix"):
            vec_i = self.db[self.movie_ids[i]]['netlsd']

            for j in range(i + 1, n):
                vec_j = self.db[self.movie_ids[j]]['netlsd']

                if vec_i is None or vec_j is None:
                    dist = float('inf')
                else:
                    dist = np.linalg.norm(vec_i - vec_j)

                self.netlsd_matrix[i, j] = dist
                self.netlsd_matrix[j, i] = dist

    def _save_cache(self):
        """Save computed matrices to disk."""
        print(f"[INFO] Saving distance cache to {self.cache_path}...")
        cache_data = {
            'movie_ids': self.movie_ids,
            'dtw_matrix': self.dtw_matrix,
            'netlsd_matrix': self.netlsd_matrix
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        # Report file size
        size_mb = os.path.getsize(self.cache_path) / (1024 * 1024)
        print(f"[INFO] Cache saved ({size_mb:.1f} MB)")

    def _load_cache(self):
        """Load precomputed matrices from disk."""
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        cached_ids = cache_data['movie_ids']

        # Verify cache matches current feature_db
        if cached_ids != self.movie_ids:
            print("[WARN] Cache movie IDs don't match current feature_db!")
            print("[INFO] Recomputing distance matrices...")
            self._compute_all_distances()
            self._save_cache()
        else:
            self.dtw_matrix = cache_data['dtw_matrix']
            self.netlsd_matrix = cache_data['netlsd_matrix']
            print(f"[INFO] Loaded {self.n_movies}x{self.n_movies} distance matrices")

    def search_by_narrative(self, query_id, top_k=5):
        """
        Fast narrative search using precomputed DTW distances.

        Time complexity: O(N log K) instead of O(N * DTW_cost)
        """
        if query_id not in self.id_to_idx:
            return []

        query_idx = self.id_to_idx[query_id]
        distances = self.dtw_matrix[query_idx]

        # Get sorted indices (excluding self)
        sorted_indices = np.argsort(distances)

        results = []
        for idx in sorted_indices:
            if idx == query_idx:
                continue
            mid = self.movie_ids[idx]
            results.append((mid, float(distances[idx])))
            if len(results) >= top_k:
                break

        return results

    def search_by_topology(self, query_id, top_k=5):
        """
        Fast topology search using precomputed NetLSD distances.
        """
        if query_id not in self.id_to_idx:
            return []

        query_idx = self.id_to_idx[query_id]
        distances = self.netlsd_matrix[query_idx]

        sorted_indices = np.argsort(distances)

        results = []
        for idx in sorted_indices:
            if idx == query_idx:
                continue
            mid = self.movie_ids[idx]
            results.append((mid, float(distances[idx])))
            if len(results) >= top_k:
                break

        return results

    def hybrid_search(self, query_id, alpha=0.5, top_k=5):
        """
        Fast hybrid search using precomputed distances.

        Uses Min-Max normalization to combine narrative and topology distances.
        """
        if query_id not in self.id_to_idx:
            return []

        query_idx = self.id_to_idx[query_id]

        # Get distance vectors
        dtw_dists = self.dtw_matrix[query_idx].copy()
        netlsd_dists = self.netlsd_matrix[query_idx].copy()

        # Exclude self
        dtw_dists[query_idx] = float('inf')
        netlsd_dists[query_idx] = float('inf')

        # Min-Max normalization (excluding inf values)
        valid_dtw = dtw_dists[dtw_dists != float('inf')]
        valid_netlsd = netlsd_dists[netlsd_dists != float('inf')]

        if len(valid_dtw) == 0 or len(valid_netlsd) == 0:
            return []

        # Normalize
        dtw_min, dtw_max = valid_dtw.min(), valid_dtw.max()
        netlsd_min, netlsd_max = valid_netlsd.min(), valid_netlsd.max()

        def normalize(vals, vmin, vmax):
            if vmax - vmin == 0:
                return np.zeros_like(vals)
            return (vals - vmin) / (vmax - vmin)

        dtw_norm = normalize(dtw_dists, dtw_min, dtw_max)
        netlsd_norm = normalize(netlsd_dists, netlsd_min, netlsd_max)

        # Combine
        combined = alpha * dtw_norm + (1 - alpha) * netlsd_norm
        combined[query_idx] = float('inf')  # Exclude self

        # Sort and return
        sorted_indices = np.argsort(combined)

        results = []
        for idx in sorted_indices:
            if idx == query_idx:
                continue
            mid = self.movie_ids[idx]
            results.append((mid, float(combined[idx])))
            if len(results) >= top_k:
                break

        return results

    def get_all_rankings(self, query_id, method='hybrid'):
        """
        Get full ranking of all movies for a query.

        Useful for evaluation metrics that need complete rankings.
        """
        return self.hybrid_search(query_id, top_k=self.n_movies) if method == 'hybrid' else \
               self.search_by_narrative(query_id, top_k=self.n_movies) if method == 'narrative' else \
               self.search_by_topology(query_id, top_k=self.n_movies)
