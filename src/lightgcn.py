import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(SimpleLightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # 初始嵌入 (第 0 層)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 初始化權重
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        """
        adj_matrix: 歸一化後的鄰接矩陣 (Sparse Tensor)
        """
        # 拼接用戶和物品嵌入
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [all_emb]
        
        # LightGCN 核心：多層圖卷積 (無非線性激活，只有線性傳播)
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        
        # 最終嵌入為各層嵌入的平均
        light_out = torch.stack(embs, dim=1)
        light_out = torch.mean(light_out, dim=1)
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

class DataLoader:
    def __init__(self, movies_json_path):
        self.user_map = {} # username -> int id
        self.item_map = {} # movie_id -> int id
        self.interactions = []
        self.load_data(movies_json_path)

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            movies = json.load(f) # 假設這是一個 list
        
        for movie in movies:
            m_id = movie['id']
            if m_id not in self.item_map:
                self.item_map[m_id] = len(self.item_map)
            
            # 解析 reviews
            for review in movie.get('reviews', []):
                u_name = review['user']
                # 過濾低分評論 (例如只保留 rating >= 7 作為正向交互)
                if review['rating'] >= 7:
                    if u_name not in self.user_map:
                        self.user_map[u_name] = len(self.user_map)
                    
                    self.interactions.append([
                        self.user_map[u_name], 
                        self.item_map[m_id]
                    ])

    def get_adj_matrix(self):
        # 構建用於 PyTorch 的稀疏鄰接矩陣
        # 這裡省略了複雜的 D^-0.5 A D^-0.5 歸一化過程，實際使用時需加上
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        
        # 建立雙向邊
        rows = [x[0] for x in self.interactions] + [x[1] + num_users for x in self.interactions]
        cols = [x[1] + num_users for x in self.interactions] + [x[0] for x in self.interactions]
        
        indices = torch.LongTensor([rows, cols])
        values = torch.FloatTensor([1.0] * len(rows)) # 簡化：全設為1
        
        shape = (num_users + num_items, num_users + num_items)
        return torch.sparse_FloatTensor(indices, values, shape)

# 用法示範
# loader = DataLoader('data/movies.json')
# model = SimpleLightGCN(len(loader.user_map), len(loader.item_map))
# adj = loader.get_adj_matrix()
# user_embs, item_embs = model(adj)