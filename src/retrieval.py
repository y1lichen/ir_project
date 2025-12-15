import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class StructRetrieval:
    def __init__(self, feature_db):
        """
        feature_db: dict, {movie_id: {'arc': np.array, 'netlsd': np.array}}
        """
        self.db = feature_db

    def search_by_narrative(self, query_id, top_k=5):
        """
        使用 DTW 搜尋情感弧最相似的電影
        """
        if query_id not in self.db: return []
        
        query_arc = self.db[query_id]['arc']
        distances = []
        
        for mid, feats in self.db.items():
            if mid == query_id: continue
            candidate_arc = feats['arc']
            dist, _ = fastdtw(query_arc, candidate_arc, radius=1, dist=lambda x, y: abs(x - y)) 
            # FastDTW 計算時間序列距離 (由半徑 radius 控制精度/速度)
            distances.append((mid, dist))
        
        # 距離越小越相似
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def search_by_topology(self, query_id, top_k=5):
        """
        使用歐幾里得距離搜尋社會結構最相似的電影 (NetLSD)
        """
        if query_id not in self.db: return []
        
        query_vec = self.db[query_id]['netlsd']
        distances = []
        
        for mid, feats in self.db.items():
            if mid == query_id: continue
            candidate_vec = feats['netlsd']
            
            # 對於 NetLSD 向量，直接計算 L2 距離
            dist = np.linalg.norm(query_vec - candidate_vec)
            distances.append((mid, dist))
            
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def hybrid_search(self, query_id, alpha=0.5):
        """
        混合搜尋：使用 Min-Max Normalization 解決尺度不平衡問題
        """
        # 1. 獲取所有電影的單項距離
        narrative_res = self.search_by_narrative(query_id, top_k=len(self.db))
        topo_res = self.search_by_topology(query_id, top_k=len(self.db))
        
        # 轉成字典以便查詢 {mid: score}
        dict_narr = dict(narrative_res)
        dict_topo = dict(topo_res)
        
        # 2. 提取分數陣列用於計算統計量
        scores_narr = np.array(list(dict_narr.values()))
        scores_topo = np.array(list(dict_topo.values()))
        
        # 3. 定義標準化函數 (Min-Max Scaling)
        # 將分數映射到 [0, 1]，0 代表最相似 (距離最小)，1 代表最不相似
        def normalize(score, min_val, max_val):
            if max_val - min_val == 0: return 0
            return (score - min_val) / (max_val - min_val)

        min_n, max_n = scores_narr.min(), scores_narr.max()
        min_t, max_t = scores_topo.min(), scores_topo.max()

        final_scores = []
        
        # 4. 僅對兩者共有的電影計算混合分數
        common_movies = set(dict_narr.keys()) & set(dict_topo.keys())
        
        for mid in common_movies:
            # 正規化
            n_norm = normalize(dict_narr[mid], min_n, max_n)
            t_norm = normalize(dict_topo[mid], min_t, max_t)
            
            # 加權融合
            combined_score = alpha * n_norm + (1 - alpha) * t_norm
            final_scores.append((mid, combined_score))
        
        final_scores.sort(key=lambda x: x[1])
        return final_scores[:5]
        """
        混合搜尋：同時考慮敘事節奏與社會結構
        """
        narrative_res = dict(self.search_by_narrative(query_id, top_k=1223)) # 全搜
        topo_res = dict(self.search_by_topology(query_id, top_k=1223))
        
        # 簡單的 Rank Aggregation (或分數加權)
        final_scores = []
        for mid in narrative_res:
            if mid in topo_res:
                # 這裡需要對分數做標準化(Normalization)才能加權，這裡僅為示意
                score = alpha * narrative_res[mid] + (1-alpha) * topo_res[mid]
                final_scores.append((mid, score))
        
        final_scores.sort(key=lambda x: x[1])
        return final_scores[:5]