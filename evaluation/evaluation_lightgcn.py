import json
import os
import sys
import numpy as np
import torch
from collections import defaultdict

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.lightgcn import SimpleLightGCN, DataLoader

def evaluate_model(movies_file='data/movies.json', k=10):
    print("正在載入評分數據...")
    with open(movies_file, 'r', encoding='utf-8') as f:
        movies = json.load(f)

    # 1. 構建 User-Item 交互字典
    # user_history: {user_name: [movie_id1, movie_id2, ...]}
    user_history = defaultdict(list)
    item_map = {}  # movie_id -> int index
    
    # 建立 Item Map
    for m in movies:
        item_map[m['id']] = len(item_map)

    # 填充 User History (只保留正面評價，例如 rating >= 7)
    interaction_count = 0
    for m in movies:
        m_idx = item_map[m['id']]
        for review in m.get('reviews', []):
            if review['rating'] >= 5:  # 設定閾值，只把喜歡的當作 Ground Truth
                user_history[review['user']].append(m_idx)
                interaction_count += 1
    
    print(f"總交互數: {interaction_count}")
    print(f"總用戶數: {len(user_history)}")

    # 2. 篩選有效用戶 (至少看過 2 部電影才能做 Train/Test Split)
    valid_users = {u: items for u, items in user_history.items() if len(items) >= 3}
    print(f"有效用戶數: {len(valid_users)}")
    

    # 3. 準備 Train/Test 數據
    train_adj_indices = []
    test_set = {} # user_idx -> target_item_idx
    
    user_map = {} # user_name -> int index
    
    for u_name, items in valid_users.items():
        u_idx = len(user_map)
        user_map[u_name] = u_idx
        
        # 隨機選一個作為測試，其他的作為訓練
        test_item = items[-1] 
        train_items = items[:-1]
        
        test_set[u_idx] = test_item
        
        for i_idx in train_items:
            train_adj_indices.append([u_idx, i_idx])

    # 4. 訓練模型 (模擬)
    # 這裡你需要實例化你的 LightGCN 並跑幾個 Epoch
    # 為了演示，我們假設模型已經產出了 embedding
    print("初始化並訓練 LightGCN (簡化版)...")
    num_users = len(user_map)
    num_items = len(item_map)
    
    model = SimpleLightGCN(num_users, num_items)
    
    # 構建訓練用的鄰接矩陣 (這裡簡化，實際需正規化)
    # 實作時請呼叫 lightgcn.py 裡的 get_adj_matrix 邏輯
    rows = [x[0] for x in train_adj_indices]
    cols = [x[1] + num_users for x in train_adj_indices]
    indices = torch.LongTensor([rows + cols, cols + rows])
    values = torch.FloatTensor([1.0] * len(indices[0]))

    # 正確的寫法 (新版 PyTorch)
    adj = torch.sparse_coo_tensor(indices, values, (num_users+num_items, num_users+num_items))    
    # 取得 Embedding
    user_embs, item_embs = model(adj) 
    
    # 5. 計算 Recall@K 與 NDCG@K
    print(f"開始評估 Top-{k}...")
    hits = 0
    ndcg = 0
    
    # 將 Tensor 轉為 Numpy
    u_vectors = user_embs.detach().numpy()
    i_vectors = item_embs.detach().numpy()
    
    for u_idx, target_item in test_set.items():
        # 計算該用戶對所有物品的分數: u_vec dot i_matrix
        scores = np.dot(u_vectors[u_idx], i_vectors.T)
        
        # 將訓練集中已經看過的電影分數設為 -inf (避免推薦已經看過的)
        # 注意：這裡省略了 mask 步驟，實際實作需加上
        
        # 取得分數最高的 Top-K 索引
        top_k_items = np.argsort(scores)[::-1][:k]
        
        # 檢查 Target 是否在 Top-K 中
        if target_item in top_k_items:
            hits += 1
            # 計算 NDCG
            rank = np.where(top_k_items == target_item)[0][0]
            ndcg += 1.0 / np.log2(rank + 2)
            
    recall = hits / len(test_set)
    avg_ndcg = ndcg / len(test_set)
    
    print("-" * 30)
    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}  : {avg_ndcg:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_model()