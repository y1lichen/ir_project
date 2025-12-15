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