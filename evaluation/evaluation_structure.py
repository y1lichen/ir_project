"""
Structural Evaluation for Movie Recommendation System

This module provides comprehensive evaluation for the movie recommendation system
based on structural consistency and beyond-accuracy metrics with academic grounding.

Academic References:
1. Reagan et al. (2016) - Emotional arc classification (6 types)
2. Kaminskas & Bridge (2017) - Beyond-accuracy metrics (ILD, Novelty, Coverage)
3. Boyd et al. (2020) - Narrative rhythm correlation
4. Tsitsulin et al. (2018) - NetLSD graph similarity
5. Ziegler et al. (2005) - Intra-list diversity
6. Ge et al. (2010) - Coverage and serendipity
7. Zhou et al. (2010) - Novelty score

This evaluator validates the system design by measuring:
1. Structural consistency: Arc type consistency, narrative rhythm correlation
2. Beyond-accuracy metrics: ILD, novelty, coverage
3. Dimension analysis: Independence between narrative and topology dimensions
"""

import numpy as np
import json
import os
import pickle
import random
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from fastdtw import fastdtw


# =============================================================================
# Precomputed Retrieval System
# =============================================================================

class PrecomputedRetrieval:
    """
    Fast retrieval using precomputed distance matrices.

    Instead of computing DTW for every query, we precompute all pairwise
    distances once and store them in matrices. This reduces query time
    from O(N * DTW_cost) to O(N) for sorting.

    Speedup: ~100x faster than on-the-fly DTW computation
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


# =============================================================================
# Structural Evaluator
# =============================================================================

class StructuralEvaluator:
    """
    Evaluator that validates the movie recommendation system based on
    structural consistency and beyond-accuracy metrics with academic grounding.
    """

    def __init__(self, feature_db, metadata_list, parsed_dir='data/parsed'):
        """
        Initialize the structural evaluator.

        Args:
            feature_db: Dict of {movie_id: {'arc': [...], 'netlsd': [...]}}
            metadata_list: List of movie metadata from movies.json
            parsed_dir: Directory containing parsed movie JSON files
        """
        print("[INFO] Initializing StructuralEvaluator...")

        self.feature_db = feature_db
        self.parsed_dir = parsed_dir

        # Build metadata dict
        self.metadata = {}
        for item in metadata_list:
            mid = item['id']
            self.metadata[mid] = {
                'title': item.get('title', mid),
                'genres': set(item.get('genres', [])),
                'writers': item.get('writers', []),
                'avg_rating': item.get('avg_rating'),
                'reviews': item.get('reviews', [])
            }

        # Precompute arc features and Reagan classification
        print("[INFO] Classifying arcs using Reagan et al. (2016) 6-type taxonomy...")
        self.arc_types = self._classify_all_arcs_reagan()

        # Precompute graph features
        print("[INFO] Precomputing graph features (NetLSD)...")
        self.graph_features = self._precompute_graph_features()

        # Build review TF-IDF matrix for baseline comparison
        print("[INFO] Building review TF-IDF matrix...")
        self.review_tfidf, self.review_movie_ids = self._build_review_tfidf()

        # Compute item popularity for novelty calculation
        print("[INFO] Computing item popularity distribution...")
        self.item_popularity = self._compute_item_popularity()

        # Build script TF-IDF matrix for content-based baseline
        print("[INFO] Building script TF-IDF matrix for baseline...")
        self.script_tfidf, self.script_movie_ids = self._build_script_tfidf()
        if self.script_tfidf is not None:
            self.script_id_to_idx = {mid: i for i, mid in enumerate(self.script_movie_ids)}
            print(f"[INFO] Script TF-IDF matrix: {self.script_tfidf.shape}")
        else:
            self.script_id_to_idx = {}
            print("[WARN] Script TF-IDF matrix not available")

        print("[INFO] StructuralEvaluator initialized.")

    # ==========================================================================
    # Arc Classification - Reagan et al. (2016)
    # ==========================================================================

    def _classify_arc_reagan(self, arc):
        """
        Classify arc into Reagan et al.'s 6 emotional arc types.

        Reference:
            Reagan, A. J., et al. (2016). "The emotional arcs of stories are
            dominated by six basic shapes." EPJ Data Science, 5(1), 31.

        Method:
            1. Divide arc into 3 segments (beginning, middle, end)
            2. Calculate slope of each segment
            3. Match to 6 patterns based on slope signs

        Arc Types:
            - rags_to_riches: Steady rise
            - riches_to_rags: Steady fall
            - man_in_hole: Fall then rise
            - icarus: Rise then fall
            - cinderella: Rise-fall-rise
            - oedipus: Fall-rise-fall

        Returns:
            str: One of the 6 arc types or 'unknown'
        """
        if arc is None or len(arc) < 30:
            return 'unknown'

        arc = np.array(arc)
        n = len(arc)

        # Divide into 3 equal segments
        seg1 = arc[:n//3]
        seg2 = arc[n//3:2*n//3]
        seg3 = arc[2*n//3:]

        def get_trend(segment):
            """Calculate slope of a segment."""
            if len(segment) < 2:
                return 0
            return (segment[-1] - segment[0]) / len(segment)

        t1 = get_trend(seg1)  # Beginning
        t2 = get_trend(seg2)  # Middle
        t3 = get_trend(seg3)  # End

        # Threshold for "significant" change
        threshold = 0.005

        rising = lambda t: t > threshold
        falling = lambda t: t < -threshold

        # Pattern matching to Reagan's 6 types
        # 1. Rags to Riches: overall rising
        if rising(t1) and rising(t3):
            if not falling(t2):
                return 'rags_to_riches'

        # 2. Riches to Rags: overall falling
        if falling(t1) and falling(t3):
            if not rising(t2):
                return 'riches_to_rags'

        # 5. Cinderella: rise-fall-rise
        if rising(t1) and falling(t2) and rising(t3):
            return 'cinderella'

        # 6. Oedipus: fall-rise-fall
        if falling(t1) and rising(t2) and falling(t3):
            return 'oedipus'

        # 3. Man in a Hole: fall then rise
        if falling(t1) and rising(t3):
            return 'man_in_hole'

        # 4. Icarus: rise then fall
        if rising(t1) and falling(t3):
            return 'icarus'

        # Default: based on overall trajectory
        overall = get_trend(arc)
        if rising(overall):
            return 'rags_to_riches'
        elif falling(overall):
            return 'riches_to_rags'
        else:
            return 'man_in_hole'  # Most common pattern

    def _classify_all_arcs_reagan(self):
        """Classify all movies using Reagan's 6-type taxonomy."""
        arc_types = {}
        for mid, feats in self.feature_db.items():
            arc = feats.get('arc')
            if arc is not None:
                arc_types[mid] = self._classify_arc_reagan(arc)
        return arc_types

    # ==========================================================================
    # Graph Feature Extraction
    # ==========================================================================

    def _load_parsed_data(self, movie_id):
        """Load parsed data for a movie."""
        filepath = os.path.join(self.parsed_dir, f"{movie_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _compute_graph_features(self, parsed_data):
        """
        Extract graph features from character interactions.

        Reference:
            Tsitsulin, A., et al. (2018). "NetLSD: Hearing the shape of a graph."
            KDD 2018.
        """
        if parsed_data is None:
            return None

        interactions = parsed_data.get('interactions', [])
        if not interactions:
            return None

        # Build graph
        G = nx.Graph()
        for edge in interactions:
            u, v = edge.get('a'), edge.get('b')
            w = edge.get('count', 1)
            if u and v:
                G.add_edge(u, v, weight=np.log(1 + w))

        if G.number_of_nodes() < 2:
            return None

        # Extract features
        degrees = [d for _, d in G.degree()]

        features = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
        }

        # Clustering coefficient
        try:
            features['clustering_coeff'] = nx.average_clustering(G)
        except:
            features['clustering_coeff'] = 0.0

        return features

    def _precompute_graph_features(self):
        """Precompute graph features for all movies."""
        graph_features = {}
        for mid in tqdm(self.feature_db.keys(), desc="Loading graph features"):
            parsed_data = self._load_parsed_data(mid)
            features = self._compute_graph_features(parsed_data)
            if features is not None:
                graph_features[mid] = features
        return graph_features

    # ==========================================================================
    # Review TF-IDF and Popularity
    # ==========================================================================

    def _build_review_tfidf(self):
        """Build TF-IDF matrix from review texts for semantic similarity."""
        movie_ids = []
        review_texts = []

        for mid, meta in self.metadata.items():
            reviews = meta.get('reviews', [])
            if reviews:
                text = ' '.join([r.get('text', '') for r in reviews])
                if len(text.strip()) > 50:
                    movie_ids.append(mid)
                    review_texts.append(text)

        if not review_texts:
            return None, []

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(review_texts)

        return tfidf_matrix, movie_ids

    def _build_script_tfidf(self, cache_path='data/script_tfidf_cache.pkl'):
        """
        Build TF-IDF matrix from movie scripts with caching.

        Reference:
            Salton, G., & Buckley, C. (1988). "Term-weighting approaches in
            automatic text retrieval." Information Processing & Management.

        Returns:
            Tuple of (tfidf_matrix, movie_ids) or (None, []) if failed
        """
        # Try loading from cache
        if os.path.exists(cache_path):
            print(f"[INFO] Loading script TF-IDF from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            return cached['matrix'], cached['movie_ids']

        # Build from scripts
        scripts_dir = os.path.join(os.path.dirname(self.parsed_dir), 'scripts')
        movie_ids = []
        script_texts = []

        print(f"[INFO] Building script TF-IDF matrix from {scripts_dir}...")
        for mid in tqdm(self.feature_db.keys(), desc="Loading scripts"):
            script_path = os.path.join(scripts_dir, f"{mid}.txt")
            if os.path.exists(script_path):
                try:
                    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    if len(text.strip()) > 100:
                        movie_ids.append(mid)
                        script_texts.append(text)
                except Exception as e:
                    continue

        if not script_texts:
            print("[WARN] No script texts found for TF-IDF")
            return None, []

        print(f"[INFO] Vectorizing {len(script_texts)} scripts...")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(script_texts)

        # Save cache
        print(f"[INFO] Saving script TF-IDF cache to: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({'matrix': tfidf_matrix, 'movie_ids': movie_ids}, f)

        return tfidf_matrix, movie_ids

    def _compute_item_popularity(self):
        """
        Compute item popularity based on number of reviews.

        Reference:
            Zhou, T., et al. (2010). "Solving the apparent diversity-accuracy
            dilemma of recommender systems."

        More reviews = more popular = lower novelty when recommended.
        Uses Laplace smoothing to avoid zero probabilities.
        """
        review_counts = {}
        total_reviews = 0

        for mid, meta in self.metadata.items():
            count = len(meta.get('reviews', []))
            review_counts[mid] = count
            total_reviews += count

        # Fallback to uniform if no reviews exist
        if total_reviews == 0:
            return {mid: 1.0 / len(self.feature_db) for mid in self.feature_db}

        # Laplace smoothing: (count + 1) / (total + vocab_size)
        n_items = len(self.feature_db)
        return {
            mid: (review_counts.get(mid, 0) + 1) / (total_reviews + n_items)
            for mid in self.feature_db.keys()
        }

    # ==========================================================================
    # Baseline Methods
    # ==========================================================================

    def _baseline_random(self, qid, k):
        """Random baseline - lower bound for evaluation."""
        all_ids = [m for m in self.feature_db.keys() if m != qid]
        sampled = random.sample(all_ids, min(k, len(all_ids)))
        return [(m, 0) for m in sampled]

    def _baseline_popularity(self, qid, k):
        """Popularity baseline - non-personalized recommendation."""
        rated_movies = [(mid, meta.get('avg_rating', 0) or 0)
                       for mid, meta in self.metadata.items()
                       if mid != qid and mid in self.feature_db]
        rated_movies.sort(key=lambda x: x[1], reverse=True)
        return rated_movies[:k]

    def _baseline_tfidf_script(self, qid, k):
        """
        TF-IDF script content-based baseline.

        Reference:
            Salton, G., & Buckley, C. (1988). "Term-weighting approaches in
            automatic text retrieval." Information Processing & Management.
        """
        if not hasattr(self, 'script_tfidf') or self.script_tfidf is None:
            return self._baseline_random(qid, k)
        if qid not in self.script_id_to_idx:
            return self._baseline_random(qid, k)

        query_idx = self.script_id_to_idx[qid]
        similarities = cosine_similarity(
            self.script_tfidf[query_idx],
            self.script_tfidf
        ).flatten()

        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in sorted_indices:
            mid = self.script_movie_ids[idx]
            if mid != qid:
                results.append((mid, similarities[idx]))
            if len(results) >= k:
                break
        return results

    def _baseline_tfidf_review(self, qid, k):
        """
        TF-IDF review-based baseline.

        Uses existing review TF-IDF matrix for semantic similarity.
        """
        if self.review_tfidf is None:
            return self._baseline_random(qid, k)

        review_id_to_idx = {mid: i for i, mid in enumerate(self.review_movie_ids)}
        if qid not in review_id_to_idx:
            return self._baseline_random(qid, k)

        query_idx = review_id_to_idx[qid]
        similarities = cosine_similarity(
            self.review_tfidf[query_idx],
            self.review_tfidf
        ).flatten()

        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in sorted_indices:
            mid = self.review_movie_ids[idx]
            if mid != qid:
                results.append((mid, similarities[idx]))
            if len(results) >= k:
                break
        return results

    def _baseline_knn_genre(self, qid, k):
        """
        Genre-based KNN baseline (Jaccard similarity).

        Reference:
            Traditional content-based filtering approach.
            Ricci, F., et al. (2015). "Recommender Systems Handbook."
        """
        query_genres = self.metadata.get(qid, {}).get('genres', set())
        if not query_genres:
            return self._baseline_random(qid, k)

        def jaccard(a, b):
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        similarities = []
        for mid in self.feature_db.keys():
            if mid == qid:
                continue
            target_genres = self.metadata.get(mid, {}).get('genres', set())
            sim = jaccard(query_genres, target_genres)
            similarities.append((mid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _baseline_knn_rating(self, qid, k):
        """
        Rating-based KNN baseline.

        Recommends movies with similar average ratings.
        """
        query_rating = self.metadata.get(qid, {}).get('avg_rating')
        if query_rating is None:
            return self._baseline_random(qid, k)

        distances = []
        for mid in self.feature_db.keys():
            if mid == qid:
                continue
            target_rating = self.metadata.get(mid, {}).get('avg_rating')
            if target_rating is not None:
                dist = abs(query_rating - target_rating)
                distances.append((mid, dist))

        distances.sort(key=lambda x: x[1])
        # Convert distance to similarity score
        return [(m, 1.0 - d/10.0) for m, d in distances[:k]]

    # ==========================================================================
    # Helper: Get Recommendations
    # ==========================================================================

    def _get_recommendations(self, retriever, qid, method, k):
        """Get recommendations using specified method."""
        # Structural methods (require retriever)
        if method == 'narrative':
            res = retriever.search_by_narrative(qid, top_k=k+1)
        elif method == 'topology':
            res = retriever.search_by_topology(qid, top_k=k+1)
        elif method == 'hybrid':
            res = retriever.hybrid_search(qid, top_k=k+1)
        # Baseline methods (6 types)
        elif method == 'random':
            res = self._baseline_random(qid, k)
        elif method == 'popularity':
            res = self._baseline_popularity(qid, k)
        elif method == 'tfidf_script':
            res = self._baseline_tfidf_script(qid, k)
        elif method == 'tfidf_review':
            res = self._baseline_tfidf_review(qid, k)
        elif method == 'knn_genre':
            res = self._baseline_knn_genre(qid, k)
        elif method == 'knn_rating':
            res = self._baseline_knn_rating(qid, k)
        else:
            raise ValueError(f"Unknown method: {method}")

        return [r[0] for r in res if r[0] != qid][:k]

    # ==========================================================================
    # Structural Consistency Metrics
    # ==========================================================================

    def arc_type_consistency(self, retriever, method='narrative', k=5):
        """
        Emotional Arc Type Consistency (ATC@K)

        Reference:
            Reagan, A. J., et al. (2016). "The emotional arcs of stories are
            dominated by six basic shapes." EPJ Data Science.

        Measures the proportion of recommended movies that have the same
        emotional arc type as the query movie, using Reagan's 6-type classification.

        Returns:
            float: Proportion of recommendations matching query arc type [0, 1]
        """
        hits = 0
        total = 0

        movie_ids = [mid for mid in self.feature_db.keys() if mid in self.arc_types]

        for qid in tqdm(movie_ids, desc=f"ATC@{k} ({method})"):
            query_type = self.arc_types.get(qid)
            if query_type == 'unknown':
                continue

            rec_ids = self._get_recommendations(retriever, qid, method, k)

            for rid in rec_ids:
                rec_type = self.arc_types.get(rid, 'unknown')
                if rec_type == query_type:
                    hits += 1

            total += len(rec_ids)

        return hits / total if total > 0 else 0.0

    def narrative_rhythm_correlation(self, retriever, method='narrative', k=5):
        """
        Narrative Rhythm Correlation (NRC@K)

        Reference:
            Boyd, R. L., et al. (2020). "The narrative arc: Revealing core
            narrative structures through text analysis." Science Advances.

        Measures Pearson correlation between emotional arc derivatives
        (rate of emotional change) of query and recommended movies.

        Returns:
            float: Mean Pearson correlation coefficient [-1, 1]
        """
        correlations = []

        movie_ids = list(self.feature_db.keys())

        for qid in tqdm(movie_ids, desc=f"NRC@{k} ({method})"):
            query_arc = self.feature_db[qid].get('arc')
            if query_arc is None or len(query_arc) < 10:
                continue

            query_derivative = np.diff(query_arc)

            rec_ids = self._get_recommendations(retriever, qid, method, k)

            for rid in rec_ids:
                rec_arc = self.feature_db.get(rid, {}).get('arc')
                if rec_arc is None or len(rec_arc) < 10:
                    continue

                rec_derivative = np.diff(rec_arc)

                min_len = min(len(query_derivative), len(rec_derivative))
                if min_len < 10:
                    continue

                corr, _ = pearsonr(query_derivative[:min_len], rec_derivative[:min_len])
                if not np.isnan(corr):
                    correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    def graph_feature_similarity(self, retriever, method='topology', k=5):
        """
        Graph Feature Similarity (GFS@K)

        Reference:
            Tsitsulin, A., et al. (2018). "NetLSD: Hearing the shape of a graph."
            KDD 2018.

        Measures continuous similarity of character network features.

        Returns:
            float: Mean normalized similarity score [0, 1]
        """
        similarities = []

        movie_ids = [mid for mid in self.feature_db.keys() if mid in self.graph_features]

        for qid in tqdm(movie_ids, desc=f"GFS@{k} ({method})"):
            query_feats = self.graph_features.get(qid)
            if query_feats is None:
                continue

            rec_ids = self._get_recommendations(retriever, qid, method, k)

            for rid in rec_ids:
                rec_feats = self.graph_features.get(rid)
                if rec_feats is None:
                    continue

                # Compare key features (normalized)
                node_sim = 1 - abs(query_feats['node_count'] - rec_feats['node_count']) / \
                           max(query_feats['node_count'], rec_feats['node_count'], 1)
                density_sim = 1 - abs(query_feats['density'] - rec_feats['density'])
                degree_sim = 1 - abs(query_feats['avg_degree'] - rec_feats['avg_degree']) / \
                            max(query_feats['avg_degree'], rec_feats['avg_degree'], 1)

                avg_sim = (node_sim + density_sim + degree_sim) / 3
                similarities.append(avg_sim)

        return np.mean(similarities) if similarities else 0.0

    # ==========================================================================
    # Beyond-Accuracy Metrics (Kaminskas & Bridge, 2017)
    # ==========================================================================

    def intra_list_diversity(self, retriever, method='hybrid', k=5, distance_type='narrative'):
        """
        Intra-List Diversity (ILD@K)

        Reference:
            Ziegler, C. N., et al. (2005). "Improving recommendation lists
            through topic diversification." WWW 2005.

            Kaminskas, M., & Bridge, D. (2017). "Diversity, serendipity, novelty,
            and coverage." ACM TIIS.

        Formula:
            ILD = (2 / k(k-1)) * Σ d(i,j) for all pairs in recommendation list

        Higher ILD indicates more diverse recommendations.

        Args:
            distance_type: 'narrative' (DTW distance) or 'topology' (NetLSD L2)

        Returns:
            float: Mean ILD across all queries
        """
        ild_scores = []

        movie_ids = list(self.feature_db.keys())

        for qid in tqdm(movie_ids, desc=f"ILD@{k} ({method}, {distance_type})"):
            rec_ids = self._get_recommendations(retriever, qid, method, k)

            if len(rec_ids) < 2:
                continue

            # Compute pairwise distances
            total_dist = 0
            pair_count = 0

            for i in range(len(rec_ids)):
                for j in range(i + 1, len(rec_ids)):
                    mid_i, mid_j = rec_ids[i], rec_ids[j]

                    if distance_type == 'narrative':
                        arc_i = self.feature_db.get(mid_i, {}).get('arc')
                        arc_j = self.feature_db.get(mid_j, {}).get('arc')
                        if arc_i is not None and arc_j is not None:
                            dist, _ = fastdtw(arc_i, arc_j, radius=1)
                            total_dist += dist
                            pair_count += 1
                    else:  # topology
                        vec_i = self.feature_db.get(mid_i, {}).get('netlsd')
                        vec_j = self.feature_db.get(mid_j, {}).get('netlsd')
                        if vec_i is not None and vec_j is not None:
                            dist = np.linalg.norm(np.array(vec_i) - np.array(vec_j))
                            total_dist += dist
                            pair_count += 1

            if pair_count > 0:
                ild = (2 * total_dist) / (len(rec_ids) * (len(rec_ids) - 1))
                ild_scores.append(ild)

        return np.mean(ild_scores) if ild_scores else 0.0

    def novelty_score(self, retriever, method='hybrid', k=5):
        """
        Item Novelty Score (Novelty@K)

        Reference:
            Zhou, T., et al. (2010). "Solving the apparent diversity-accuracy
            dilemma of recommender systems."

            Kaminskas, M., & Bridge, D. (2017). "Diversity, serendipity, novelty,
            and coverage." ACM TIIS.

        Formula:
            Novelty = mean(-log2(popularity)) for recommended items

        Higher novelty indicates recommendations of less popular items.

        Returns:
            float: Mean novelty score (higher = more novel)
        """
        novelty_scores = []

        movie_ids = list(self.feature_db.keys())

        for qid in tqdm(movie_ids, desc=f"Novelty@{k} ({method})"):
            rec_ids = self._get_recommendations(retriever, qid, method, k)

            for rid in rec_ids:
                pop = self.item_popularity.get(rid, 0.001)
                novelty = -np.log2(max(pop, 1e-10))
                novelty_scores.append(novelty)

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def catalog_coverage(self, retriever, method='hybrid', k=5):
        """
        Catalog Coverage (Coverage@K)

        Reference:
            Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010). "Beyond accuracy:
            Evaluating recommender systems by coverage and serendipity." RecSys 2010.

        Formula:
            Coverage = |Unique Recommended Items| / |All Items|

        Higher coverage indicates the system recommends a broader range of items.

        Returns:
            float: Coverage ratio [0, 1]
        """
        all_recommended = set()

        movie_ids = list(self.feature_db.keys())

        for qid in tqdm(movie_ids, desc=f"Coverage@{k} ({method})"):
            rec_ids = self._get_recommendations(retriever, qid, method, k)
            all_recommended.update(rec_ids)

        total_items = len(self.feature_db)
        return len(all_recommended) / total_items if total_items > 0 else 0.0

    # ==========================================================================
    # Dimension Analysis
    # ==========================================================================

    def dimension_independence(self, retriever, sample_size=200):
        """
        Dimension Independence Score (DIS)

        Measures Spearman correlation between narrative and topology rankings.
        Low correlation indicates the dimensions capture different aspects.

        Returns:
            float: Mean Spearman correlation [-1, 1]
                   Low value = dimensions are independent (hybrid is valuable)
        """
        correlations = []

        movie_ids = list(self.feature_db.keys())
        sample_ids = random.sample(movie_ids, min(sample_size, len(movie_ids)))

        for qid in tqdm(sample_ids, desc="DIS"):
            narr_res = retriever.search_by_narrative(qid, top_k=len(self.feature_db))
            topo_res = retriever.search_by_topology(qid, top_k=len(self.feature_db))

            narr_ranks = {r[0]: i for i, r in enumerate(narr_res)}
            topo_ranks = {r[0]: i for i, r in enumerate(topo_res)}

            common_ids = set(narr_ranks.keys()) & set(topo_ranks.keys())
            if len(common_ids) < 10:
                continue

            common_ids = list(common_ids)
            narr_r = [narr_ranks[mid] for mid in common_ids]
            topo_r = [topo_ranks[mid] for mid in common_ids]

            corr, _ = spearmanr(narr_r, topo_r)
            if not np.isnan(corr):
                correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    # ==========================================================================
    # Baseline Comparison
    # ==========================================================================

    def compute_baseline_metrics(self, retriever, k=5):
        """
        Compute metrics for all baseline methods.

        Baseline Methods:
        1. random: Random sampling (lower bound)
        2. popularity: Rating-based (non-personalized)
        3. tfidf_script: Script content similarity (Salton & Buckley, 1988)
        4. tfidf_review: Review text similarity
        5. knn_genre: Genre Jaccard similarity (traditional CB)
        6. knn_rating: Rating proximity (quality-based)

        Academic References:
        - TF-IDF: Salton & Buckley (1988)
        - KNN: Ricci et al. (2015) Recommender Systems Handbook
        - Evaluation: McNee et al. (2006) Beyond Accuracy

        Returns:
            dict: Baseline metrics for comparison
        """
        baselines = {}

        baseline_methods = [
            'random',
            'popularity',
            'tfidf_script',
            'tfidf_review',
            'knn_genre',
            'knn_rating'
        ]

        for baseline_method in baseline_methods:
            print(f"\n--- Computing {baseline_method.upper()} baseline ---")
            baselines[baseline_method] = {
                'ATC': self.arc_type_consistency(retriever, method=baseline_method, k=k),
                'NRC': self.narrative_rhythm_correlation(retriever, method=baseline_method, k=k),
                'GFS': self.graph_feature_similarity(retriever, method=baseline_method, k=k),
                'ILD_narr': self.intra_list_diversity(retriever, method=baseline_method, k=k, distance_type='narrative'),
                'ILD_topo': self.intra_list_diversity(retriever, method=baseline_method, k=k, distance_type='topology'),
            }

        return baselines

    # ==========================================================================
    # Visualization Methods
    # ==========================================================================

    def generate_visualizations(self, results, baselines=None, save_dir='figures'):
        """
        Generate visualization charts for evaluation results.

        Args:
            results: Dict with 'narrative', 'topology', 'hybrid' metrics
            baselines: Optional dict with baseline method results
            save_dir: Directory to save figures

        Returns:
            List of saved figure paths
        """
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        methods = ['narrative', 'topology', 'hybrid']
        colors = {'narrative': '#3498db', 'topology': '#e74c3c', 'hybrid': '#2ecc71'}

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        width = 0.25

        # 1. Beyond-Accuracy Metrics (Bar Chart)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ILD subplot
        ax1 = axes[0]
        ild_metrics = ['ILD_narr', 'ILD_topo']
        x = np.arange(len(ild_metrics))
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in ild_metrics]
            bars = ax1.bar(x + i * width, values, width, label=method.capitalize(),
                          color=colors[method], alpha=0.85)
        ax1.set_xlabel('Diversity Type')
        ax1.set_ylabel('ILD Score (higher = more diverse)')
        ax1.set_title('Intra-List Diversity\n(Ziegler et al., 2005)')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['Narrative\nDiversity', 'Topology\nDiversity'])
        ax1.legend()

        # Coverage & Novelty subplot
        ax2 = axes[1]
        other_metrics = ['Coverage']
        x = np.arange(len(other_metrics) + 1)

        # Normalize Novelty for display (divide by 10 for scale)
        for i, method in enumerate(methods):
            cov = results[method].get('Coverage', 0)
            nov = results[method].get('Novelty', 0) / 10  # Scale down for visualization
            values = [cov, nov]
            bars = ax2.bar(x + i * width, values, width, label=method.capitalize(),
                          color=colors[method], alpha=0.85)

        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Coverage & Novelty\n(Ge et al., 2010; Zhou et al., 2010)')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(['Catalog\nCoverage', 'Novelty\n(scaled /10)'])
        ax2.legend()

        plt.tight_layout()
        path = os.path.join(save_dir, 'beyond_accuracy.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path)
        print(f"[VIZ] Saved: {path}")

        # 2. Method Comparison (All methods including baselines)
        if baselines:
            fig, ax = plt.subplots(figsize=(18, 7))
            compare_metrics = ['ATC', 'NRC', 'GFS']
            x = np.arange(len(compare_metrics))
            bar_width = 0.09  # Adjusted for 9 methods

            # All baseline methods + our structural methods
            all_methods = ['random', 'popularity', 'tfidf_script', 'tfidf_review',
                          'knn_genre', 'knn_rating', 'narrative', 'topology', 'hybrid']
            method_colors = {
                'random': '#95a5a6',
                'popularity': '#f39c12',
                'tfidf_script': '#9b59b6',
                'tfidf_review': '#8e44ad',
                'knn_genre': '#1abc9c',
                'knn_rating': '#16a085',
                'narrative': '#3498db',
                'topology': '#e74c3c',
                'hybrid': '#2ecc71'
            }
            method_labels = {
                'random': 'Random',
                'popularity': 'Popularity',
                'tfidf_script': 'TF-IDF Script',
                'tfidf_review': 'TF-IDF Review',
                'knn_genre': 'KNN Genre',
                'knn_rating': 'KNN Rating',
                'narrative': 'Narrative',
                'topology': 'Topology',
                'hybrid': 'Hybrid'
            }

            for i, method in enumerate(all_methods):
                if method in baselines:
                    values = [baselines[method].get(m, 0) for m in compare_metrics]
                elif method in results:
                    values = [results[method].get(m, 0) for m in compare_metrics]
                else:
                    continue
                bars = ax.bar(x + i * bar_width, values, bar_width,
                             label=method_labels.get(method, method),
                             color=method_colors.get(method, '#7f8c8d'), alpha=0.85)

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Method Comparison\n(Structural Methods vs Traditional IR/ML Baselines)')
            ax.set_xticks(x + 4 * bar_width)  # Center for 9 methods
            ax.set_xticklabels(['Arc Type\nConsistency', 'Narrative\nRhythm', 'Graph\nSimilarity'])
            ax.legend(loc='upper right', ncol=3, fontsize=9)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            path = os.path.join(save_dir, 'method_comparison.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(path)
            print(f"[VIZ] Saved: {path}")

        print(f"\n[VIZ] Generated {len(saved_files)} visualization files in '{save_dir}/'")
        return saved_files

    # ==========================================================================
    # Full Evaluation Report
    # ==========================================================================

    def generate_full_report(self, retriever, k=5, include_baselines=True, generate_plots=True):
        """
        Generate comprehensive evaluation report with academic grounding.

        References all metrics to their academic sources.
        """
        print("\n" + "=" * 70)
        print("STRUCTURAL RECOMMENDATION EVALUATION REPORT")
        print("Based on academic evaluation frameworks")
        print("=" * 70)

        results = {'narrative': {}, 'topology': {}, 'hybrid': {}}

        for method in ['narrative', 'topology', 'hybrid']:
            print(f"\n{'='*70}")
            print(f"Evaluating {method.upper()} method")
            print("=" * 70)

            # Structural Consistency Metrics
            print("\n--- Structural Consistency Metrics ---")
            atc = self.arc_type_consistency(retriever, method=method, k=k)
            nrc = self.narrative_rhythm_correlation(retriever, method=method, k=k)
            gfs = self.graph_feature_similarity(retriever, method=method, k=k)

            # Beyond-Accuracy Metrics
            print("\n--- Beyond-Accuracy Metrics (Kaminskas & Bridge, 2017) ---")
            ild_narr = self.intra_list_diversity(retriever, method=method, k=k, distance_type='narrative')
            ild_topo = self.intra_list_diversity(retriever, method=method, k=k, distance_type='topology')
            novelty = self.novelty_score(retriever, method=method, k=k)
            coverage = self.catalog_coverage(retriever, method=method, k=k)

            results[method] = {
                'ATC': atc,
                'NRC': nrc,
                'GFS': gfs,
                'ILD_narr': ild_narr,
                'ILD_topo': ild_topo,
                'Novelty': novelty,
                'Coverage': coverage,
            }

        # Dimension Independence
        print("\n" + "=" * 70)
        print("Dimension Analysis")
        print("=" * 70)
        dis = self.dimension_independence(retriever)

        # Baselines
        baselines = {}
        if include_baselines:
            print("\n" + "=" * 70)
            print("Baseline Comparison")
            print("=" * 70)
            baselines = self.compute_baseline_metrics(retriever, k=k)

        # Print Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print("\n--- Structural Consistency (Reagan et al., 2016; Boyd et al., 2020) ---")
        print(f"{'Metric':<20} | {'Narrative':>12} | {'Topology':>12} | {'Hybrid':>12}")
        print("-" * 62)
        for metric in ['ATC', 'NRC', 'GFS']:
            print(f"{metric:<20} | {results['narrative'][metric]:>12.4f} | {results['topology'][metric]:>12.4f} | {results['hybrid'][metric]:>12.4f}")

        print("\n--- Beyond-Accuracy (Kaminskas & Bridge, 2017; Ziegler et al., 2005) ---")
        print(f"{'Metric':<20} | {'Narrative':>12} | {'Topology':>12} | {'Hybrid':>12}")
        print("-" * 62)
        for metric in ['ILD_narr', 'ILD_topo', 'Novelty', 'Coverage']:
            print(f"{metric:<20} | {results['narrative'][metric]:>12.4f} | {results['topology'][metric]:>12.4f} | {results['hybrid'][metric]:>12.4f}")

        print(f"\n--- Dimension Independence Score (Spearman rho) ---")
        print(f"DIS: {dis:.4f}")
        print("  (Low value = dimensions are independent, hybrid is valuable)")

        if include_baselines:
            print("\n--- Baseline Comparison (All Methods) ---")
            header = f"{'Metric':<10} | {'Random':>8} | {'Popular':>8} | {'TF-Script':>9} | {'TF-Review':>9} | {'KNN-Gen':>8} | {'KNN-Rat':>8} | {'Narr':>8} | {'Hybrid':>8}"
            print(header)
            print("-" * len(header))
            for metric in ['ATC', 'NRC', 'GFS']:
                row = f"{metric:<10}"
                for bl in ['random', 'popularity', 'tfidf_script', 'tfidf_review', 'knn_genre', 'knn_rating']:
                    val = baselines.get(bl, {}).get(metric, 0)
                    row += f" | {val:>8.4f}"
                narr_val = results['narrative'].get(metric, 0)
                hyb_val = results['hybrid'].get(metric, 0)
                row += f" | {narr_val:>8.4f} | {hyb_val:>8.4f}"
                print(row)

        print("\n" + "=" * 70)
        print("METRIC LEGEND")
        print("=" * 70)
        print("""
ATC  : Arc Type Consistency (Reagan et al., 2016) - Higher is better
NRC  : Narrative Rhythm Correlation (Boyd et al., 2020) - Higher is better
GFS  : Graph Feature Similarity (Tsitsulin et al., 2018) - Higher is better
ILD  : Intra-List Diversity (Ziegler et al., 2005) - Higher = more diverse
Novelty : Item Novelty Score (Zhou et al., 2010) - Higher = more novel
Coverage: Catalog Coverage (Ge et al., 2010) - Higher = broader recommendations
DIS  : Dimension Independence Score - Low = dimensions are independent
""")

        # Generate visualizations
        if generate_plots:
            print("\n" + "=" * 70)
            print("Generating Visualizations")
            print("=" * 70)
            saved_plots = self.generate_visualizations(results, baselines)
        else:
            saved_plots = []

        return {
            'results': results,
            'dimension_independence': dis,
            'baselines': baselines,
            'visualization_files': saved_plots
        }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Main entry point for running structural evaluation.

    Usage:
        python evaluation/evaluation_structure.py
    """
    import sys

    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Change to project root for relative paths
    os.chdir(project_root)

    from main import load_or_process_features

    # Load features
    print("[INFO] Loading features...")
    feature_db = load_or_process_features()

    # Initialize precomputed retriever (fast!)
    print("[INFO] Initializing precomputed retriever...")
    retriever = PrecomputedRetrieval(feature_db, cache_path='data/distance_cache.pkl')

    # Load metadata
    print("[INFO] Loading metadata...")
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    # Initialize structural evaluator
    evaluator = StructuralEvaluator(
        feature_db=feature_db,
        metadata_list=metadata_list,
        parsed_dir='data/parsed'
    )

    # Run full evaluation with baselines
    print("\n[INFO] Running full evaluation with academic-grounded metrics...")
    results = evaluator.generate_full_report(
        retriever,
        k=5,
        include_baselines=True
    )

    # Save results
    print("\n[INFO] Saving results to structural_eval_results.json...")
    with open('structural_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Evaluation complete!")
    print("\n" + "=" * 70)
    print("CITATION GUIDE")
    print("=" * 70)
    print("""
When citing these evaluation metrics in your paper, please use:

Structural Consistency Metrics:
- ATC: Reagan, A. J., et al. (2016). "The emotional arcs of stories are
       dominated by six basic shapes." EPJ Data Science, 5(1), 31.
- NRC: Boyd, R. L., et al. (2020). "The narrative arc: Revealing core
       narrative structures through text analysis." Science Advances, 6(32).
- GFS: Tsitsulin, A., et al. (2018). "NetLSD: Hearing the shape of a graph."
       KDD 2018.

Beyond-Accuracy Metrics:
- ILD: Ziegler, C. N., et al. (2005). "Improving recommendation lists through
       topic diversification." WWW 2005.
- Framework: Kaminskas, M., & Bridge, D. (2017). "Diversity, serendipity,
             novelty, and coverage." ACM TIIS, 7(1), 1-42.
- Novelty: Zhou, T., et al. (2010). "Solving the apparent diversity-accuracy
           dilemma of recommender systems."
- Coverage: Ge, M., et al. (2010). "Beyond accuracy: Evaluating recommender
            systems by coverage and serendipity." RecSys 2010.

Why Genre-based PR Curve is not appropriate:
- McNee, S. M., et al. (2006). "Being accurate is not enough: How accuracy
  metrics have hurt recommender systems." CHI Extended Abstracts.
""")


if __name__ == '__main__':
    main()
