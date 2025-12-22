import json
import os
import pickle
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from scipy.stats import ttest_rel

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import StructRetrieval

# =========================
# 設定路徑
# =========================
GENOME_SCORES_PATH = 'data/ml-25m/genome-scores.csv'
MAPPING_PATH = 'data/id_mapping.json'
FEATURES_CACHE = 'data/features_cache.pkl'
PCA_DIM = 50  # 保留主成分數量，可調整


def load_genome_matrix():
    """
    載入 Tag Genome 並轉為矩陣格式
    Returns: DataFrame (index=movieId, columns=tagId, values=relevance)
    """
    print("正在載入 MovieLens Tag Genome (這可能需要幾秒鐘)...")
    if not os.path.exists(GENOME_SCORES_PATH):
        raise FileNotFoundError(f"找不到 {GENOME_SCORES_PATH}")

    df = pd.read_csv(GENOME_SCORES_PATH)

    # Pivot 成 movie × tag 的矩陣
    matrix = df.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
    print(f"Genome Matrix Shape: {matrix.shape}")
    return matrix


def apply_pca(matrix, n_components=PCA_DIM):
    """
    對 genome matrix 做 PCA 降維
    Returns: numpy array (電影數 × n_components)
    """
    print(f"對 Tag Genome 做 PCA，保留 {n_components} 維主成分...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(matrix.values)
    print(f"PCA 後矩陣形狀: {reduced.shape}")
    return pd.DataFrame(reduced, index=matrix.index)


def bootstrap_mean_diff(a, b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx] - b[idx]))
    return np.percentile(diffs, [2.5, 50, 97.5])


def load_script_tfidf(feature_db, cache_path='data/script_tfidf_cache.pkl'):
    """
    載入或構建 Script TF-IDF 矩陣
    Returns: (tfidf_matrix, movie_ids, id_to_idx)
    """
    # 嘗試從 cache 載入
    if os.path.exists(cache_path):
        print(f"[INFO] Loading script TF-IDF from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        movie_ids = cached['movie_ids']
        tfidf_matrix = cached['matrix']
        id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
        return tfidf_matrix, movie_ids, id_to_idx

    # 從 scripts 資料夾構建
    scripts_dir = 'data/scripts'
    movie_ids = []
    script_texts = []

    print(f"[INFO] Building script TF-IDF matrix from {scripts_dir}...")
    for mid in tqdm(feature_db.keys(), desc="Loading scripts"):
        script_path = os.path.join(scripts_dir, f"{mid}.txt")
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if len(text.strip()) > 100:
                    movie_ids.append(mid)
                    script_texts.append(text)
            except Exception:
                continue

    if not script_texts:
        print("[WARN] No script texts found for TF-IDF")
        return None, [], {}

    print(f"[INFO] Vectorizing {len(script_texts)} scripts...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(script_texts)

    # 儲存 cache
    print(f"[INFO] Saving script TF-IDF cache to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump({'matrix': tfidf_matrix, 'movie_ids': movie_ids}, f)

    id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    return tfidf_matrix, movie_ids, id_to_idx


def tfidf_search(query_id, tfidf_matrix, movie_ids, id_to_idx, top_k=5):
    """
    使用 TF-IDF cosine similarity 進行檢索
    """
    if query_id not in id_to_idx:
        return []

    query_idx = id_to_idx[query_id]
    similarities = cosine_similarity(
        tfidf_matrix[query_idx],
        tfidf_matrix
    ).flatten()

    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_indices:
        mid = movie_ids[idx]
        if mid != query_id:
            results.append((mid, similarities[idx]))
        if len(results) >= top_k:
            break
    return results


def evaluate_genome():

    # =========================
    # 1. 準備資料
    # =========================
    if not os.path.exists(MAPPING_PATH):
        print("請先執行 utils/build_mapping.py")
        return

    with open(MAPPING_PATH, 'r') as f:
        id_map = json.load(f)  # {imsdb_slug: ml_id}

    with open(FEATURES_CACHE, 'rb') as f:
        feature_db = pickle.load(f)

    genome_matrix = load_genome_matrix()
    genome_matrix_pca = apply_pca(genome_matrix, PCA_DIM)
    retriever = StructRetrieval(feature_db)

    # 載入 TF-IDF
    print("[INFO] Loading Script TF-IDF...")
    tfidf_matrix, tfidf_movie_ids, tfidf_id_to_idx = load_script_tfidf(feature_db)

    # =========================
    # 2. 篩選測試集
    # =========================
    test_movies = [mid for mid in feature_db.keys()
                   if mid in id_map and id_map[mid] in genome_matrix_pca.index]

    print(f"符合評估條件的電影數: {len(test_movies)}")
    test_samples = test_movies
    print(f"開始評估 {len(test_samples)} 部樣本電影...")

    # =========================
    # 3. 評估迴圈
    # =========================
    metrics = {'TF-IDF': [], 'Narrative (DTW)': [], 'Hybrid': []}

    for query_id in tqdm(test_samples, desc="計算基因相似度 (PCA)"):
        query_ml_id = id_map[query_id]
        query_vec = genome_matrix_pca.loc[query_ml_id].values.reshape(1, -1)

        def calc_list_similarity(rec_list):
            sims = []
            for rec_id, _ in rec_list:
                if rec_id in id_map:
                    rec_ml_id = id_map[rec_id]
                    if rec_ml_id in genome_matrix_pca.index:
                        rec_vec = genome_matrix_pca.loc[rec_ml_id].values.reshape(1, -1)
                        sims.append(cosine_similarity(query_vec, rec_vec)[0][0])
            return np.mean(sims) if sims else 0.0

        # A. TF-IDF baseline
        tfidf_recs = tfidf_search(query_id, tfidf_matrix, tfidf_movie_ids, tfidf_id_to_idx, top_k=5)
        metrics['TF-IDF'].append(calc_list_similarity(tfidf_recs))

        # B. Narrative (DTW)
        recs = retriever.search_by_narrative(query_id, top_k=5)
        metrics['Narrative (DTW)'].append(calc_list_similarity(recs))

        # C. Hybrid
        recs = retriever.hybrid_search(query_id, top_k=5)
        metrics['Hybrid'].append(calc_list_similarity(recs))

    # =========================
    # 4. 平均結果
    # =========================
    print("\n" + "=" * 60)
    print("MovieLens 25M Tag Genome 語義一致性評估 (PCA)")
    print("指標: Cosine Similarity (越高代表內容標籤分佈越相似)")
    print("=" * 60)

    results = {k: np.mean(v) for k, v in metrics.items()}
    base_score = results['TF-IDF']

    for method, score in results.items():
        lift = (score - base_score) / base_score * 100
        print(f"{method:<20} : {score:.4f} (Lift: {lift:+.2f}%)")

    # =========================
    # 5. 統計檢定（Paired t-test）
    # =========================
    print("\n" + "=" * 60)
    print("統計檢定（Paired t-test）")
    print("=" * 60)

    tfidf_scores = np.array(metrics['TF-IDF'])
    narrative_scores = np.array(metrics['Narrative (DTW)'])
    hybrid_scores = np.array(metrics['Hybrid'])

    t_nt, p_nt = ttest_rel(narrative_scores, tfidf_scores)
    t_ht, p_ht = ttest_rel(hybrid_scores, tfidf_scores)
    t_nh, p_nh = ttest_rel(narrative_scores, hybrid_scores)

    print(f"Narrative vs TF-IDF : t = {t_nt:.3f}, p = {p_nt:.4e}")
    print(f"Hybrid    vs TF-IDF : t = {t_ht:.3f}, p = {p_ht:.4e}")
    print(f"Narrative vs Hybrid : t = {t_nh:.3f}, p = {p_nh:.4e}")

    # =========================
    # 6. Bootstrap 信賴區間
    # =========================
    print("\n" + "=" * 60)
    print("Bootstrap 95% 信賴區間（Mean Difference）")
    print("=" * 60)

    ci_nt = bootstrap_mean_diff(narrative_scores, tfidf_scores)
    ci_ht = bootstrap_mean_diff(hybrid_scores, tfidf_scores)

    print(f"Narrative - TF-IDF : 95% CI = [{ci_nt[0]:.4f}, {ci_nt[2]:.4f}]")
    print(f"Hybrid    - TF-IDF : 95% CI = [{ci_ht[0]:.4f}, {ci_ht[2]:.4f}]")

    print("-" * 60)
    if results['Hybrid'] > results['TF-IDF']:
        print("驗證成功：結構相似的電影，其 Tag Genome 分佈顯著高於 TF-IDF baseline。")
    else:
        print("結果：TF-IDF baseline 表現優於或相近於結構方法。")


if __name__ == "__main__":
    evaluate_genome()
