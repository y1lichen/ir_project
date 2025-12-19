import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, feature_db, metadata_list, threshold_mode="flexible"):
        """
        feature_db: { movie_id: {'arc': [...], 'netlsd': [...] } }
        metadata_list: look up movies.json to check metadata_list structure
        """
        self.threshold_mode = threshold_mode # 'strict' (>=2) 或 'flexible' (>1)
        self.feature_db = feature_db

        # 1. turn movie metadata to dict & turn genres(list) to set
        self.metadata = {}
        for item in metadata_list:
            mid = item['id']
            self.metadata[mid] = {
                'genres': set(item.get('genres', [])),
                'title': item.get('title', mid)
            }

        # 2. 計算每一部電影的「所有相關電影數量」
        print("[INFO] 預計算相關電影數量 (Ground Truth)...")
        self.relevant_counts_cache = {}
        all_ids = list(self.metadata.keys())
        
        for qid in tqdm(all_ids, desc="Pre-calculating"):
            query_genres = self.metadata[qid]['genres']
            if not query_genres:
                self.relevant_counts_cache[qid] = 0
                continue

            count = 0
            for target_id in all_ids:
                if qid == target_id: continue
                if self._is_relevant(query_genres, self.metadata[target_id]['genres']):
                    count += 1
            self.relevant_counts_cache[qid] = count

    def _is_relevant(self, query_genres, target_genres):
        if self.threshold_mode == "strict":
            threshold = 2 if len(query_genres) >= 2 else 1 # 至少要有兩個 Genre 重疊才視為相關
        else:
            threshold = 1 # 只要有重疊一個即視為相關
        return len(query_genres & target_genres) >= threshold
            

    def get_metrics(self, query_id, recommended_ids, k=5):
        query_data = self.metadata.get(query_id)
        if not query_data or not query_data['genres']: 
            return 0, 0, 0, 0

        query_genres = query_data['genres']
        
        # 1. Calculate TP
        tp = 0
        for rid in recommended_ids[:k]:
            if rid == query_id: continue
            if self._is_relevant(query_genres, self.metadata[rid]['genres']):
                tp += 1

        # 2. Precision
        precision = tp / k
        
        # 3. Recall
        total_relevant_count = self.relevant_counts_cache.get(query_id, 0)
        recall = tp / total_relevant_count if total_relevant_count > 0 else 0
        
        # 4. Hit (只要 tp > 0 就算命中了)
        hit = 1 if tp > 0 else 0
        
        # 5. F1 Score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1, hit


    def precision_at_k(self, retriever, method='hybrid', k=5):
        all_p, all_r, all_f, all_h = [], [], [], []
        movie_ids = list(self.feature_db.keys())

        # 使用 tqdm 顯示進度條
        start_time = time.time()
        for qid in tqdm(movie_ids, desc=f"Evaluating {method} @K={k}"):
            try:
                if method == 'narrative':
                    res = retriever.search_by_narrative(qid, top_k=k)
                elif method == 'topology':
                    res = retriever.search_by_topology(qid, top_k=k)
                else:
                    res = retriever.hybrid_search(qid, top_k=k)

                rec_ids = [r[0] for r in res]
                p, r, f, h = self.get_metrics(qid, rec_ids, k=k)
                
                all_p.append(p)
                all_r.append(r)
                all_f.append(f)
                all_h.append(h)
                
            except Exception as e:
                print(f"\n[WARN] 處理 {qid} 時發生錯誤: {e}")
                continue
        
        elapsed = time.time() - start_time
        print(f"[INFO] 處理完成，耗時 {elapsed:.2f} 秒。成功處理 {len(movie_ids)} 部電影。")

        print(f"\n--- {method.upper()} Evaluation (K={k}) ---")
        print(f"Mean Precision: {np.mean(all_p):.4f}")
        print(f"Mean Recall:    {np.mean(all_r):.4f}")
        print(f"Mean F1-Score:  {np.mean(all_f):.4f}\n")
        print(f"Hit Rate:       {np.mean(all_h):.4f}")        


    def get_pr_points(self, query_id, all_sorted_ids):
        """
        針對單一查詢，計算從 K=1 到 N 的所有 P-R 點
        """
        query_genres = self.metadata[query_id]['genres']
        total_relevant = self.relevant_counts_cache.get(query_id, 0)
        if total_relevant == 0: return None
        
        tp = 0
        points = []
        for k, rid in enumerate(all_sorted_ids, 1):
            if rid == query_id: continue
            if self._is_relevant(query_genres, self.metadata[rid]['genres']):
                tp += 1
                precision = tp / k
                recall = tp / total_relevant
                points.append((recall, precision))
            
            if tp == total_relevant: break # 找齊了就可以停了
            
        return points

    def interpolate_11_points(self, all_pr_points):
        """
        計算 11-point interpolated precision (平均所有 query 的結果)
        """
        recall_levels = np.linspace(0, 1, 11) # recall_levels = [0.0, 0.1, ..., 1.0]
        queries_interpolated = [] # 存所有 Query 在這 11 個點上的 Precision

        for pr_points in all_pr_points:
            if not pr_points: continue
            
            # 針對單個 Query 做內插法
            q_interp = []
            for r_level in recall_levels:
                # 找出所有 recall >= r_level 的 precision 最大值
                possible_p = [p for r, p in pr_points if r >= r_level]
                q_interp.append(max(possible_p) if possible_p else 0)
            queries_interpolated.append(q_interp)

        return recall_levels, np.mean(queries_interpolated, axis=0)
    
    def calculate_auc(self, recall_levels, precision_levels):
        auc_score = np.trapz(precision_levels, recall_levels)
        return auc_score
    

    def plot_11point_curve(self, retriever):
        movie_ids = list(self.feature_db.keys())
        methods = ['narrative', 'topology', 'hybrid']
        plt.figure(figsize=(10, 7))

        for method in methods:
            all_query_pr = []
            for qid in tqdm(movie_ids, desc=f"PR-Curve for {method}"):
                if method == 'narrative':
                    res = retriever.search_by_narrative(qid, top_k=len(self.feature_db))
                elif method == 'topology':
                    res = retriever.search_by_topology(qid, top_k=len(self.feature_db))
                else:
                    res = retriever.hybrid_search(qid, top_k=len(self.feature_db))
                
                sorted_ids = [r[0] for r in res]
                pr_points = self.get_pr_points(qid, sorted_ids)
                if pr_points:
                    all_query_pr.append(pr_points)

            # 1. 計算 11 點內插
            recalls, precisions = self.interpolate_11_points(all_query_pr)
            
            # 2. print recalls & precisions table
            print(f"\n[ {method.upper()} - Precision at Recall Levels ]")
            print(f"{'Recall':<10} | {'Precision':<10}")
            print("-" * 25)
            for r, p in zip(recalls, precisions):
                print(f"{r:<10.1f} | {p:<10.4f}")

            # 3. calculate AUC
            auc_score = self.calculate_auc(recalls, precisions)
            print(f"==> {method.upper()} AUC Score: {auc_score:.4f}\n")

            # 4. plot
            plt.plot(recalls, precisions, marker='o', linewidth=2, label=f'{method.upper()} (AUC: {auc_score:.4f})')

        plt.xlabel('Recall')
        plt.ylabel('Interpolated Precision')
        plt.title('11-point Interpolated Precision-Recall Curve')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()