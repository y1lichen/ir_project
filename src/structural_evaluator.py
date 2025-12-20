"""
Structural Evaluator for Movie Recommendation System

Academic-grounded evaluation framework based on:
1. Reagan et al. (2016) - Emotional arc classification (6 types)
2. Kaminskas & Bridge (2017) - Beyond-accuracy metrics (ILD, Novelty, Coverage)
3. Boyd et al. (2020) - Narrative rhythm correlation
4. Tsitsulin et al. (2018) - NetLSD graph similarity
5. Ziegler et al. (2005) - Intra-list diversity

This evaluator validates the system design by measuring:
1. Structural consistency: Arc type consistency, narrative rhythm correlation
2. Beyond-accuracy metrics: ILD, novelty, coverage
3. Proxy ground truth: Same-writer recall, rating proximity, semantic similarity
4. Dimension analysis: Independence between narrative and topology dimensions
"""

import numpy as np
import json
import os
import random
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from fastdtw import fastdtw


class StructuralEvaluator:
    """
    Evaluator that validates the movie recommendation system based on
    structural consistency and beyond-accuracy metrics with academic grounding.
    """

    # Reagan et al. (2016) - Six emotional arc types
    ARC_TYPES = [
        'rags_to_riches',   # Steady rise
        'riches_to_rags',   # Steady fall
        'man_in_hole',      # Fall then rise
        'icarus',           # Rise then fall
        'cinderella',       # Rise-fall-rise
        'oedipus'           # Fall-rise-fall
    ]

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

        # Build writer index for proxy evaluation
        print("[INFO] Building writer index...")
        self.writer_index = self._build_writer_index()

        # Build review TF-IDF matrix for semantic similarity
        print("[INFO] Building review TF-IDF matrix...")
        self.review_tfidf, self.review_movie_ids = self._build_review_tfidf()

        # Compute item popularity for novelty calculation
        print("[INFO] Computing item popularity distribution...")
        self.item_popularity = self._compute_item_popularity()

        # Build user-movie index for LOO evaluation
        print("[INFO] Building user-movie index for LOO evaluation...")
        self.user_movies = self._build_user_movie_index()
        print(f"[INFO] Found {len(self.user_movies)} users with 2+ reviews")

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
            - rags_to_riches: Steady rise (悲轉喜)
            - riches_to_rags: Steady fall (喜轉悲)
            - man_in_hole: Fall then rise (困境脫困)
            - icarus: Rise then fall (伊卡洛斯)
            - cinderella: Rise-fall-rise (灰姑娘)
            - oedipus: Fall-rise-fall (俄狄浦斯)

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
    # Writer Index, Review TF-IDF, and Popularity
    # ==========================================================================

    def _build_writer_index(self):
        """Build index of movies by writer for proxy evaluation."""
        writer_to_movies = {}
        for mid, meta in self.metadata.items():
            for writer in meta.get('writers', []):
                if writer not in writer_to_movies:
                    writer_to_movies[writer] = set()
                writer_to_movies[writer].add(mid)

        # Only keep writers with 2+ movies
        return {w: movies for w, movies in writer_to_movies.items() if len(movies) >= 2}

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

    def _compute_item_popularity(self):
        """
        Compute item popularity distribution for novelty calculation.

        Reference:
            Zhou, T., et al. (2010). "Solving the apparent diversity-accuracy
            dilemma of recommender systems."
        """
        # Use uniform popularity since we don't have interaction data
        # In real scenario, this would be based on view/rating counts
        n_items = len(self.feature_db)
        return {mid: 1.0 / n_items for mid in self.feature_db.keys()}

    def _build_user_movie_index(self):
        """
        Build user -> movie list mapping for Leave-One-Out (LOO) evaluation.

        Reference:
            Bauer, C., & Zangerle, E. (2020). Offline evaluation options for
            recommender systems. Information Retrieval Journal.

        Returns:
            dict: {user_id: [{'movie_id': mid, 'rating': rating}, ...]}
                  Only includes users with 2+ movies
        """
        from collections import defaultdict
        user_movies = defaultdict(list)

        for mid, meta in self.metadata.items():
            # Only include movies that are in feature_db
            if mid not in self.feature_db:
                continue

            for review in meta.get('reviews', []):
                user = review.get('user')
                rating = review.get('rating')
                if user and rating:
                    user_movies[user].append({
                        'movie_id': mid,
                        'rating': rating
                    })

        # Filter users with 2+ movies (required for LOO)
        return {u: m for u, m in user_movies.items() if len(m) >= 2}

    # ==========================================================================
    # Helper: Get Recommendations
    # ==========================================================================

    def _get_recommendations(self, retriever, qid, method, k):
        """Get recommendations using specified method."""
        if method == 'narrative':
            res = retriever.search_by_narrative(qid, top_k=k+1)
        elif method == 'topology':
            res = retriever.search_by_topology(qid, top_k=k+1)
        elif method == 'hybrid':
            res = retriever.hybrid_search(qid, top_k=k+1)
        elif method == 'random':
            # Random baseline
            all_ids = list(self.feature_db.keys())
            all_ids = [m for m in all_ids if m != qid]
            sampled = random.sample(all_ids, min(k, len(all_ids)))
            res = [(m, 0) for m in sampled]
        elif method == 'popularity':
            # Popularity baseline (using rating as proxy)
            rated_movies = [(mid, meta.get('avg_rating', 0) or 0)
                           for mid, meta in self.metadata.items()
                           if mid != qid and mid in self.feature_db]
            rated_movies.sort(key=lambda x: x[1], reverse=True)
            res = rated_movies[:k]
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
            Novelty = mean(-log₂(popularity)) for recommended items

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
    # Proxy Ground Truth Metrics
    # ==========================================================================

    def auteur_consistency(self, retriever, method='hybrid', k=5):
        """
        Auteur Consistency Score (ACS@K)

        Measures if recommendations include movies by the same writer/director.
        This serves as a proxy for ground truth since auteurs often have
        consistent narrative and structural styles.

        Returns:
            float: Hit rate of same-writer recommendations [0, 1]
        """
        hits = 0
        total = 0

        for qid in tqdm(self.feature_db.keys(), desc=f"ACS@{k} ({method})"):
            query_writers = self.metadata.get(qid, {}).get('writers', [])
            if not query_writers:
                continue

            # Find same-writer movies
            same_writer_movies = set()
            for writer in query_writers:
                same_writer_movies.update(self.writer_index.get(writer, set()))
            same_writer_movies.discard(qid)

            if not same_writer_movies:
                continue

            rec_ids = set(self._get_recommendations(retriever, qid, method, k))

            if rec_ids & same_writer_movies:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    def quality_proximity(self, retriever, method='hybrid', k=5):
        """
        Quality Proximity Score (QPS@K)

        Measures average rating difference between query and recommendations.
        Lower values indicate recommendations have similar quality ratings.

        Returns:
            float: Mean absolute rating difference (lower is better)
        """
        rating_diffs = []

        for qid in tqdm(self.feature_db.keys(), desc=f"QPS@{k} ({method})"):
            query_rating = self.metadata.get(qid, {}).get('avg_rating')
            if query_rating is None:
                continue

            rec_ids = self._get_recommendations(retriever, qid, method, k)

            for rid in rec_ids:
                rec_rating = self.metadata.get(rid, {}).get('avg_rating')
                if rec_rating is not None:
                    rating_diffs.append(abs(query_rating - rec_rating))

        return np.mean(rating_diffs) if rating_diffs else float('inf')

    def semantic_consistency(self, retriever, method='hybrid', k=5):
        """
        Semantic Consistency Score (SCS@K)

        Measures TF-IDF cosine similarity of review texts between
        query and recommended movies.

        Returns:
            float: Mean cosine similarity [0, 1]
        """
        if self.review_tfidf is None:
            return 0.0

        similarities = []
        review_id_to_idx = {mid: i for i, mid in enumerate(self.review_movie_ids)}

        for qid in tqdm(self.feature_db.keys(), desc=f"SCS@{k} ({method})"):
            if qid not in review_id_to_idx:
                continue

            query_idx = review_id_to_idx[qid]

            rec_ids = self._get_recommendations(retriever, qid, method, k)

            for rid in rec_ids:
                if rid in review_id_to_idx:
                    rec_idx = review_id_to_idx[rid]
                    sim = cosine_similarity(
                        self.review_tfidf[query_idx],
                        self.review_tfidf[rec_idx]
                    )[0, 0]
                    similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

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
    # User-Based Evaluation (LOO Protocol)
    # ==========================================================================

    def user_leave_one_out_hit(self, retriever, method='hybrid', k=10):
        """
        User-Based Leave-One-Out Hit Rate (UB-LOO Hit@K)

        Reference:
            Bauer, C., & Zangerle, E. (2020). Offline evaluation options for
            recommender systems. Information Retrieval Journal.

            Zangerle, E., et al. (2022). Evaluating Recommender Systems: Survey
            and Framework. ACM Computing Surveys.

        For each user who rated at least two movies, we randomly hold out one
        movie as the ground truth and use another rated movie as the query.

        Args:
            retriever: The retrieval system to evaluate
            method: 'narrative', 'topology', or 'hybrid'
            k: Number of recommendations to retrieve

        Returns:
            float: Hit rate [0, 1]
        """
        hits = 0
        total = 0

        for user, movie_list in tqdm(self.user_movies.items(),
                                     desc=f"UB-LOO Hit@{k} ({method})"):
            if len(movie_list) < 2:
                continue

            # Randomly select held-out and query
            indices = list(range(len(movie_list)))
            random.shuffle(indices)
            held_out_idx = indices[0]
            query_idx = indices[1]

            held_out_movie = movie_list[held_out_idx]['movie_id']
            query_movie = movie_list[query_idx]['movie_id']

            # Get recommendations
            rec_ids = self._get_recommendations(retriever, query_movie, method, k)

            if held_out_movie in rec_ids:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    def user_leave_one_out_ndcg(self, retriever, method='hybrid', k=10):
        """
        User-Based Leave-One-Out NDCG@K

        Reference:
            Standard IR evaluation metric. NDCG considers the ranking position
            of the held-out item in the recommendation list.

        Formula:
            NDCG = DCG / IDCG
            DCG = 1 / log2(rank + 1) if hit, else 0
            IDCG = 1 (since we have exactly 1 relevant item)

        Args:
            retriever: The retrieval system to evaluate
            method: 'narrative', 'topology', or 'hybrid'
            k: Number of recommendations to retrieve

        Returns:
            float: NDCG score [0, 1]
        """
        ndcg_scores = []

        for user, movie_list in tqdm(self.user_movies.items(),
                                     desc=f"UB-LOO NDCG@{k} ({method})"):
            if len(movie_list) < 2:
                continue

            # Randomly select held-out and query
            indices = list(range(len(movie_list)))
            random.shuffle(indices)
            held_out_idx = indices[0]
            query_idx = indices[1]

            held_out_movie = movie_list[held_out_idx]['movie_id']
            query_movie = movie_list[query_idx]['movie_id']

            # Get recommendations
            rec_ids = self._get_recommendations(retriever, query_movie, method, k)

            # Calculate NDCG
            if held_out_movie in rec_ids:
                rank = rec_ids.index(held_out_movie) + 1  # 1-indexed
                dcg = 1.0 / np.log2(rank + 1)
                ndcg = dcg  # IDCG = 1 for single relevant item
            else:
                ndcg = 0.0

            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def rating_weighted_hit_rate(self, retriever, method='hybrid', k=10):
        """
        Rating-Weighted Hit Rate (RW-Hit@K)

        Reference:
            Extends the standard Hit@K metric by weighting hits by the user's
            rating of the held-out item. Higher-rated items count more.

        Formula:
            RW-Hit = Σ(I(hit) × rating) / Σ(rating)

        Args:
            retriever: The retrieval system to evaluate
            method: 'narrative', 'topology', or 'hybrid'
            k: Number of recommendations to retrieve

        Returns:
            float: Weighted hit rate [0, 1]
        """
        weighted_hits = 0.0
        total_weight = 0.0

        for user, movie_list in tqdm(self.user_movies.items(),
                                     desc=f"RW-Hit@{k} ({method})"):
            if len(movie_list) < 2:
                continue

            # Randomly select held-out and query
            indices = list(range(len(movie_list)))
            random.shuffle(indices)
            held_out_idx = indices[0]
            query_idx = indices[1]

            held_out_movie = movie_list[held_out_idx]['movie_id']
            held_out_rating = movie_list[held_out_idx]['rating']
            query_movie = movie_list[query_idx]['movie_id']

            # Normalize rating to [0, 1]
            normalized_rating = held_out_rating / 10.0

            # Get recommendations
            rec_ids = self._get_recommendations(retriever, query_movie, method, k)

            if held_out_movie in rec_ids:
                weighted_hits += normalized_rating

            total_weight += normalized_rating

        return weighted_hits / total_weight if total_weight > 0 else 0.0

    def compute_user_based_metrics(self, retriever, method='hybrid', k=10):
        """
        Compute all user-based evaluation metrics.

        Returns:
            dict: {'Hit@K': float, 'NDCG@K': float, 'RW_Hit@K': float}
        """
        return {
            f'Hit@{k}': self.user_leave_one_out_hit(retriever, method=method, k=k),
            f'NDCG@{k}': self.user_leave_one_out_ndcg(retriever, method=method, k=k),
            f'RW_Hit@{k}': self.rating_weighted_hit_rate(retriever, method=method, k=k),
        }

    # ==========================================================================
    # Baseline Methods
    # ==========================================================================

    def compute_baseline_metrics(self, retriever, k=5):
        """
        Compute metrics for random and popularity baselines.

        Returns:
            dict: Baseline metrics for comparison
        """
        baselines = {}

        for baseline_method in ['random', 'popularity']:
            print(f"\n--- Computing {baseline_method.upper()} baseline ---")
            baselines[baseline_method] = {
                'ATC': self.arc_type_consistency(retriever, method=baseline_method, k=k),
                'NRC': self.narrative_rhythm_correlation(retriever, method=baseline_method, k=k),
                'GFS': self.graph_feature_similarity(retriever, method=baseline_method, k=k),
                'ILD_narr': self.intra_list_diversity(retriever, method=baseline_method, k=k, distance_type='narrative'),
                'ILD_topo': self.intra_list_diversity(retriever, method=baseline_method, k=k, distance_type='topology'),
                'ACS': self.auteur_consistency(retriever, method=baseline_method, k=k),
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
            baselines: Optional dict with 'random', 'popularity' baselines
            save_dir: Directory to save figures

        Returns:
            List of saved figure paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        methods = ['narrative', 'topology', 'hybrid']
        colors = {'narrative': '#3498db', 'topology': '#e74c3c', 'hybrid': '#2ecc71'}

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        # 1. Structural Consistency Metrics (Bar Chart)
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['ATC', 'NRC', 'GFS']
        x = np.arange(len(metrics))
        width = 0.25

        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, values, width, label=method.capitalize(),
                         color=colors[method], alpha=0.85)
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Structural Consistency Metrics\n(Reagan et al., 2016; Boyd et al., 2020)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Arc Type\nConsistency', 'Narrative Rhythm\nCorrelation', 'Graph Feature\nSimilarity'])
        ax.legend(loc='upper right')
        ax.set_ylim(0, max([results[m].get('GFS', 0) for m in methods]) * 1.2)

        plt.tight_layout()
        path = os.path.join(save_dir, 'structural_consistency.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path)
        print(f"[VIZ] Saved: {path}")

        # 2. Beyond-Accuracy Metrics (Bar Chart)
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
        other_metrics = ['Coverage', 'Novelty']
        x = np.arange(len(other_metrics))

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

        # 3. User-Based Evaluation (Bar Chart)
        fig, ax = plt.subplots(figsize=(10, 6))
        ub_metrics = ['Hit@10', 'NDCG@10', 'RW_Hit@10']
        x = np.arange(len(ub_metrics))

        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) * 100 for m in ub_metrics]  # Convert to percentage
            bars = ax.bar(x + i * width, values, width, label=method.capitalize(),
                         color=colors[method], alpha=0.85)
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score (%)')
        ax.set_title('User-Based Evaluation (LOO Protocol)\n(Bauer & Zangerle, 2020)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Hit@10', 'NDCG@10', 'Rating-Weighted\nHit@10'])
        ax.legend(loc='upper right')

        plt.tight_layout()
        path = os.path.join(save_dir, 'user_based_evaluation.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path)
        print(f"[VIZ] Saved: {path}")

        # 4. Proxy Ground Truth Metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        proxy_metrics = ['ACS', 'SCS']  # Exclude QPS since lower is better
        x = np.arange(len(proxy_metrics))

        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in proxy_metrics]
            bars = ax.bar(x + i * width, values, width, label=method.capitalize(),
                         color=colors[method], alpha=0.85)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score (higher is better)')
        ax.set_title('Proxy Ground Truth Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Auteur\nConsistency', 'Semantic\nConsistency'])
        ax.legend()

        plt.tight_layout()
        path = os.path.join(save_dir, 'proxy_ground_truth.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path)
        print(f"[VIZ] Saved: {path}")

        # 5. Baseline Comparison (if available)
        if baselines:
            fig, ax = plt.subplots(figsize=(12, 6))
            compare_metrics = ['ATC', 'NRC', 'GFS', 'ACS']
            x = np.arange(len(compare_metrics))
            bar_width = 0.2

            all_methods = ['random', 'popularity', 'narrative', 'hybrid']
            method_colors = {
                'random': '#95a5a6',
                'popularity': '#f39c12',
                'narrative': '#3498db',
                'hybrid': '#2ecc71'
            }

            for i, method in enumerate(all_methods):
                if method in ['random', 'popularity']:
                    values = [baselines[method].get(m, 0) for m in compare_metrics]
                else:
                    values = [results[method].get(m, 0) for m in compare_metrics]
                bars = ax.bar(x + i * bar_width, values, bar_width,
                             label=method.capitalize(), color=method_colors[method], alpha=0.85)

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Baseline Comparison\n(Our methods vs Random/Popularity)')
            ax.set_xticks(x + 1.5 * bar_width)
            ax.set_xticklabels(['Arc Type\nConsistency', 'Narrative\nRhythm', 'Graph\nSimilarity', 'Auteur\nConsistency'])
            ax.legend(loc='upper right')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            path = os.path.join(save_dir, 'baseline_comparison.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(path)
            print(f"[VIZ] Saved: {path}")

        # 6. Radar Chart (Overall Comparison)
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Select metrics for radar (normalize to 0-1 scale)
        radar_metrics = ['ATC', 'NRC', 'GFS', 'Coverage', 'ACS', 'SCS']
        radar_labels = ['Arc Type\nConsistency', 'Narrative\nRhythm', 'Graph\nSimilarity',
                       'Coverage', 'Auteur\nConsistency', 'Semantic\nConsistency']

        # Compute angles
        num_vars = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        for method in methods:
            values = []
            for m in radar_metrics:
                val = results[method].get(m, 0)
                # Normalize NRC from [-1,1] to [0,1]
                if m == 'NRC':
                    val = (val + 1) / 2
                values.append(val)
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=method.capitalize(),
                   color=colors[method], alpha=0.8)
            ax.fill(angles, values, alpha=0.15, color=colors[method])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Comparison\n(Normalized Scores)', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        path = os.path.join(save_dir, 'radar_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path)
        print(f"[VIZ] Saved: {path}")

        # 7. Summary Dashboard (Combined View)
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        # Top-left: Structural Consistency
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['ATC', 'NRC', 'GFS']
        x = np.arange(len(metrics))
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            ax1.bar(x + i * width, values, width, label=method.capitalize(),
                   color=colors[method], alpha=0.85)
        ax1.set_title('Structural Consistency', fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['ATC', 'NRC', 'GFS'])
        ax1.legend(fontsize=8)

        # Top-right: Beyond-Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Coverage']
        x = np.arange(len(metrics))
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            ax2.bar(x + i * width, values, width, label=method.capitalize(),
                   color=colors[method], alpha=0.85)
        ax2.set_title('Catalog Coverage', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(['Coverage'])
        ax2.legend(fontsize=8)

        # Middle-left: User-Based
        ax3 = fig.add_subplot(gs[1, 0])
        metrics = ['Hit@10', 'NDCG@10', 'RW_Hit@10']
        x = np.arange(len(metrics))
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) * 100 for m in metrics]
            ax3.bar(x + i * width, values, width, label=method.capitalize(),
                   color=colors[method], alpha=0.85)
        ax3.set_title('User-Based Evaluation (%)', fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(['Hit@10', 'NDCG@10', 'RW_Hit@10'])
        ax3.legend(fontsize=8)

        # Middle-right: Proxy Metrics
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = ['ACS', 'SCS']
        x = np.arange(len(metrics))
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            ax4.bar(x + i * width, values, width, label=method.capitalize(),
                   color=colors[method], alpha=0.85)
        ax4.set_title('Proxy Ground Truth', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(['Auteur', 'Semantic'])
        ax4.legend(fontsize=8)

        # Bottom: ILD comparison
        ax5 = fig.add_subplot(gs[2, :])
        metrics = ['ILD_narr', 'ILD_topo']
        x = np.arange(len(metrics))
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            ax5.bar(x + i * width, values, width, label=method.capitalize(),
                   color=colors[method], alpha=0.85)
        ax5.set_title('Intra-List Diversity (ILD)', fontweight='bold')
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(['Narrative ILD', 'Topology ILD'])
        ax5.legend(fontsize=8)

        plt.suptitle('Evaluation Dashboard', fontsize=16, fontweight='bold', y=1.02)
        path = os.path.join(save_dir, 'dashboard.png')
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

            # Proxy Ground Truth Metrics
            print("\n--- Proxy Ground Truth Metrics ---")
            acs = self.auteur_consistency(retriever, method=method, k=k)
            qps = self.quality_proximity(retriever, method=method, k=k)
            scs = self.semantic_consistency(retriever, method=method, k=k)

            results[method] = {
                'ATC': atc,
                'NRC': nrc,
                'GFS': gfs,
                'ILD_narr': ild_narr,
                'ILD_topo': ild_topo,
                'Novelty': novelty,
                'Coverage': coverage,
                'ACS': acs,
                'QPS': qps,
                'SCS': scs
            }

        # Dimension Independence
        print("\n" + "=" * 70)
        print("Dimension Analysis")
        print("=" * 70)
        dis = self.dimension_independence(retriever)

        # User-Based Evaluation (LOO Protocol)
        print("\n" + "=" * 70)
        print("User-Based Evaluation (LOO Protocol)")
        print(f"Using {len(self.user_movies)} users with 2+ reviews")
        print("=" * 70)
        ub_results = {}
        for method in ['narrative', 'topology', 'hybrid']:
            print(f"\n--- {method.upper()} method ---")
            ub_results[method] = {
                'Hit@10': self.user_leave_one_out_hit(retriever, method=method, k=10),
                'NDCG@10': self.user_leave_one_out_ndcg(retriever, method=method, k=10),
                'RW_Hit@10': self.rating_weighted_hit_rate(retriever, method=method, k=10),
            }

        # Add UB results to main results
        for method in ['narrative', 'topology', 'hybrid']:
            results[method].update(ub_results[method])

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

        print("\n--- Proxy Ground Truth ---")
        print(f"{'Metric':<20} | {'Narrative':>12} | {'Topology':>12} | {'Hybrid':>12}")
        print("-" * 62)
        for metric in ['ACS', 'QPS', 'SCS']:
            print(f"{metric:<20} | {results['narrative'][metric]:>12.4f} | {results['topology'][metric]:>12.4f} | {results['hybrid'][metric]:>12.4f}")

        print(f"\n--- Dimension Independence Score (Spearman ρ) ---")
        print(f"DIS: {dis:.4f}")
        print("  (Low value = dimensions are independent, hybrid is valuable)")

        print("\n--- User-Based Evaluation (Bauer & Zangerle, 2020) ---")
        print(f"{'Metric':<20} | {'Narrative':>12} | {'Topology':>12} | {'Hybrid':>12}")
        print("-" * 62)
        for metric in ['Hit@10', 'NDCG@10', 'RW_Hit@10']:
            print(f"{metric:<20} | {results['narrative'][metric]:>12.4f} | {results['topology'][metric]:>12.4f} | {results['hybrid'][metric]:>12.4f}")

        if include_baselines:
            print("\n--- Baseline Comparison ---")
            print(f"{'Metric':<12} | {'Random':>10} | {'Popularity':>10} | {'Narrative':>10} | {'Hybrid':>10}")
            print("-" * 62)
            for metric in ['ATC', 'NRC', 'GFS', 'ILD_narr', 'ACS']:
                rand_val = baselines['random'].get(metric, 0)
                pop_val = baselines['popularity'].get(metric, 0)
                narr_val = results['narrative'].get(metric, 0)
                hyb_val = results['hybrid'].get(metric, 0)
                print(f"{metric:<12} | {rand_val:>10.4f} | {pop_val:>10.4f} | {narr_val:>10.4f} | {hyb_val:>10.4f}")

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
ACS  : Auteur Consistency Score - Higher is better (proxy ground truth)
QPS  : Quality Proximity Score - Lower is better
SCS  : Semantic Consistency Score - Higher is better
DIS  : Dimension Independence Score - Low = dimensions are independent

--- User-Based Metrics (Bauer & Zangerle, 2020) ---
Hit@10    : User-Based Leave-One-Out Hit Rate - Higher is better
NDCG@10   : Normalized Discounted Cumulative Gain - Higher is better
RW_Hit@10 : Rating-Weighted Hit Rate - Higher is better
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
            'user_based_evaluation': ub_results,
            'visualization_files': saved_plots
        }


def visualize_from_json(json_path, save_dir='figures'):
    """
    Generate visualizations from a saved JSON results file.

    Usage:
        from src.structural_evaluator import visualize_from_json
        visualize_from_json('structural_eval_results.json')

    Args:
        json_path: Path to the saved evaluation results JSON
        save_dir: Directory to save visualization files

    Returns:
        List of saved figure paths
    """
    import json
    import os

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', {})
    baselines = data.get('baselines', {})

    # Create a minimal evaluator just for visualization
    class MinimalVisualizer:
        def __init__(self):
            pass

    viz = MinimalVisualizer()

    # Copy the visualization method
    import types
    viz.generate_visualizations = types.MethodType(
        StructuralEvaluator.generate_visualizations, viz
    )

    return viz.generate_visualizations(results, baselines, save_dir)
