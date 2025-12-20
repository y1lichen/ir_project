import json
import pickle
import numpy as np
from src.retrieval import StructRetrieval

def get_movie_users(movies_file='data/movies.json'):
    """建立 {movie_id: set(user_names)} 的映射"""
    with open(movies_file, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    movie_users = {}
    for m in movies:
        # 只記錄喜歡該電影的用戶 (rating >= 7)
        users = set(r['user'] for r in m.get('reviews', []) if r['rating'] >= 0)
        if users:
            movie_users[m['id']] = users
    return movie_users

def calculate_jaccard(users_a, users_b):
    if not users_a or not users_b: return 0.0
    intersection = len(users_a & users_b)
    union = len(users_a | users_b)
    return intersection / union

def evaluate_retrieval(features_cache='data/features_cache.pkl', movies_file='data/movies.json'):
    # 1. 載入特徵與用戶數據
    with open(features_cache, 'rb') as f:
        feature_db = pickle.load(f)
    
    movie_users = get_movie_users(movies_file)
    retriever = StructRetrieval(feature_db)
    
    # 篩選出既有特徵又有用戶數據的電影作為測試集
    test_movies = [mid for mid in feature_db.keys() if mid in movie_users]
    
    print(f"可評估的電影數: {len(test_movies)}")
    
    total_narrative_score = 0
    total_topology_score = 0
    total_random_score = 0
    valid_queries = 0
    
    for query_id in test_movies:
        # 取得 Ground Truth 用戶群
        users_query = movie_users[query_id]
        if len(users_query) < 2: continue # 用戶太少，統計不準
        
        valid_queries += 1
        
        # A. 你的系統推薦 (敘事結構)
        narrative_recs = retriever.search_by_narrative(query_id, top_k=5)
        n_score = 0
        for rec_id, _ in narrative_recs:
            if rec_id in movie_users:
                n_score += calculate_jaccard(users_query, movie_users[rec_id])
        total_narrative_score += (n_score / 5) # 平均分
        
        # B. 你的系統推薦 (拓撲結構)
        topo_recs = retriever.search_by_topology(query_id, top_k=5)
        t_score = 0
        for rec_id, _ in topo_recs:
            if rec_id in movie_users:
                t_score += calculate_jaccard(users_query, movie_users[rec_id])
        total_topology_score += (t_score / 5)

        # C. 隨機推薦 (Baseline)
        random_recs = np.random.choice(test_movies, 5, replace=False)
        r_score = 0
        for rec_id in random_recs:
            if rec_id in movie_users:
                r_score += calculate_jaccard(users_query, movie_users[rec_id])
        total_random_score += (r_score / 5)

    print("-" * 30)
    print(f"平均用戶重疊度 (User Jaccard Similarity) - 測試樣本: {valid_queries}")
    print(f"Random Baseline : {total_random_score / valid_queries:.4f}")
    print(f"Narrative (DTW): {total_narrative_score / valid_queries:.4f}")
    print(f"Topology (NetLSD): {total_topology_score / valid_queries:.4f}")
    print("-" * 30)
    
    if (total_narrative_score > total_random_score):
        print("✅ 敘事結構檢索優於隨機猜測")
    if (total_topology_score > total_random_score):
        print("✅ 拓撲結構檢索優於隨機猜測")

if __name__ == "__main__":
    evaluate_retrieval()