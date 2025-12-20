"""
User Overlap Evaluation for Movie Recommendation System

This module evaluates movie recommendations based on user overlap (Jaccard similarity).
The hypothesis is: if two movies are liked by the same users, they are similar.

Metrics:
- User Jaccard Similarity: |Users_A ∩ Users_B| / |Users_A ∪ Users_B|

Methods Evaluated:
- Random Baseline
- Narrative (DTW)
- Topology (NetLSD)
"""

import json
import os
import pickle
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_rel

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import StructRetrieval


# =========================
# 設定
# =========================
RATING_THRESHOLD = 7  # 只考慮評分 >= 7 的用戶（表示喜歡）


def bootstrap_mean_diff(a, b, n_boot=10000, seed=42):
    """
    Bootstrap 估計 mean(a - b) 的 95% CI
    """
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx] - b[idx]))
    return np.percentile(diffs, [2.5, 50, 97.5])


def get_movie_users(movies_file='data/movies.json', rating_threshold=RATING_THRESHOLD):
    """
    建立 {movie_id: set(user_names)} 的映射

    Args:
        movies_file: 電影資料 JSON 檔案路徑
        rating_threshold: 評分閾值，只計入 >= 此值的用戶

    Returns:
        dict: {movie_id: set(user_names)}
    """
    with open(movies_file, 'r', encoding='utf-8') as f:
        movies = json.load(f)

    movie_users = {}
    for m in movies:
        # 只記錄喜歡該電影的用戶 (rating >= threshold)
        users = set(
            r['user'] for r in m.get('reviews', [])
            if r.get('rating') is not None and r['rating'] >= rating_threshold
        )
        if users:
            movie_users[m['id']] = users

    return movie_users


def calculate_jaccard(users_a, users_b):
    """
    計算兩個用戶集合的 Jaccard 相似度

    Args:
        users_a: 用戶集合 A
        users_b: 用戶集合 B

    Returns:
        float: Jaccard similarity [0, 1]
    """
    if not users_a or not users_b:
        return 0.0
    intersection = len(users_a & users_b)
    union = len(users_a | users_b)
    return intersection / union if union > 0 else 0.0


def evaluate_retrieval(features_cache='data/features_cache.pkl', movies_file='data/movies.json'):
    """
    使用用戶重疊度評估推薦系統

    評估指標: User Jaccard Similarity
    - 假設相似電影會被相同用戶喜歡
    - 比較 Narrative, Topology 和 Random Baseline
    """

    # =========================
    # 1. 載入特徵與用戶數據
    # =========================
    print("[INFO] 載入特徵資料...")
    with open(features_cache, 'rb') as f:
        feature_db = pickle.load(f)

    print("[INFO] 建立用戶-電影映射...")
    movie_users = get_movie_users(movies_file)
    retriever = StructRetrieval(feature_db)

    # 篩選出既有特徵又有用戶數據的電影作為測試集
    test_movies = [mid for mid in feature_db.keys() if mid in movie_users]

    print(f"[INFO] 可評估的電影數: {len(test_movies)}")

    # =========================
    # 2. 評估迴圈
    # =========================
    metrics = {
        'Random': [],
        'Narrative (DTW)': [],
        'Topology (NetLSD)': []
    }

    valid_queries = 0

    for query_id in tqdm(test_movies, desc="計算用戶重疊度"):
        # 取得 Ground Truth 用戶群
        users_query = movie_users[query_id]
        if len(users_query) < 2:
            continue  # 用戶太少，統計不準

        valid_queries += 1

        # A. Narrative (DTW) 推薦
        narrative_recs = retriever.search_by_narrative(query_id, top_k=5)
        n_scores = []
        for rec_id, _ in narrative_recs:
            if rec_id in movie_users:
                n_scores.append(calculate_jaccard(users_query, movie_users[rec_id]))
        metrics['Narrative (DTW)'].append(np.mean(n_scores) if n_scores else 0.0)

        # B. Topology (NetLSD) 推薦
        topo_recs = retriever.search_by_topology(query_id, top_k=5)
        t_scores = []
        for rec_id, _ in topo_recs:
            if rec_id in movie_users:
                t_scores.append(calculate_jaccard(users_query, movie_users[rec_id]))
        metrics['Topology (NetLSD)'].append(np.mean(t_scores) if t_scores else 0.0)

        # C. Random Baseline (排除 query_id 本身)
        candidates = [mid for mid in test_movies if mid != query_id]
        random_recs = np.random.choice(candidates, min(5, len(candidates)), replace=False)
        r_scores = []
        for rec_id in random_recs:
            if rec_id in movie_users:
                r_scores.append(calculate_jaccard(users_query, movie_users[rec_id]))
        metrics['Random'].append(np.mean(r_scores) if r_scores else 0.0)

    # =========================
    # 3. 平均結果
    # =========================
    print("\n" + "=" * 60)
    print("用戶重疊度評估 (User Jaccard Similarity)")
    print(f"評分閾值: >= {RATING_THRESHOLD} (表示喜歡)")
    print(f"測試樣本數: {valid_queries}")
    print("=" * 60)

    results = {k: np.mean(v) for k, v in metrics.items()}
    base_score = results['Random']

    for method, score in results.items():
        lift = (score - base_score) / base_score * 100 if base_score > 0 else 0
        print(f"{method:<20} : {score:.4f} (Lift: {lift:+.2f}%)")

    # =========================
    # 4. 統計檢定（Paired t-test）
    # =========================
    print("\n" + "=" * 60)
    print("統計檢定（Paired t-test）")
    print("=" * 60)

    random_scores = np.array(metrics['Random'])
    narrative_scores = np.array(metrics['Narrative (DTW)'])
    topology_scores = np.array(metrics['Topology (NetLSD)'])

    t_nr, p_nr = ttest_rel(narrative_scores, random_scores)
    t_tr, p_tr = ttest_rel(topology_scores, random_scores)
    t_nt, p_nt = ttest_rel(narrative_scores, topology_scores)

    print(f"Narrative vs Random  : t = {t_nr:.3f}, p = {p_nr:.4e}")
    print(f"Topology  vs Random  : t = {t_tr:.3f}, p = {p_tr:.4e}")
    print(f"Narrative vs Topology: t = {t_nt:.3f}, p = {p_nt:.4e}")

    # =========================
    # 5. Bootstrap 信賴區間
    # =========================
    print("\n" + "=" * 60)
    print("Bootstrap 95% 信賴區間（Mean Difference）")
    print("=" * 60)

    ci_nr = bootstrap_mean_diff(narrative_scores, random_scores)
    ci_tr = bootstrap_mean_diff(topology_scores, random_scores)

    print(f"Narrative - Random  : 95% CI = [{ci_nr[0]:.4f}, {ci_nr[2]:.4f}]")
    print(f"Topology  - Random  : 95% CI = [{ci_tr[0]:.4f}, {ci_tr[2]:.4f}]")

    # =========================
    # 6. 結論
    # =========================
    print("\n" + "-" * 60)

    if p_nr < 0.05 and results['Narrative (DTW)'] > results['Random']:
        print("✅ 敘事結構檢索顯著優於隨機猜測 (p < 0.05)")
    elif results['Narrative (DTW)'] > results['Random']:
        print("⚠️  敘事結構檢索優於隨機猜測，但未達統計顯著")
    else:
        print("❌ 敘事結構檢索未優於隨機猜測")

    if p_tr < 0.05 and results['Topology (NetLSD)'] > results['Random']:
        print("✅ 拓撲結構檢索顯著優於隨機猜測 (p < 0.05)")
    elif results['Topology (NetLSD)'] > results['Random']:
        print("⚠️  拓撲結構檢索優於隨機猜測，但未達統計顯著")
    else:
        print("❌ 拓撲結構檢索未優於隨機猜測")

    return {
        'results': results,
        'metrics': metrics,
        'valid_queries': valid_queries
    }


if __name__ == "__main__":
    evaluate_retrieval()
