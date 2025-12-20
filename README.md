# STRUCT-REC: 電影結構化檢索系統

基於**敘事結構**、**社會拓撲**與**輕量化圖卷積網絡**的電影推薦系統

---

## 目錄

1. [研究目的](#1-研究目的-purpose)
2. [解決方案](#2-解決方案-solution)
3. [系統成果](#3-系統成果-system-outcomes)
4. [結論](#4-結論-conclusions)
5. [參考文獻](#5-參考文獻-references)
6. [安裝與使用](#6-安裝與使用)

---

## 1. 研究目的 (Purpose)

### 1.1 問題陳述

在當前人工智慧發展浪潮中，大語言模型（Large Language Models, LLMs）佔據核心地位。然而，其對計算資源（GPU 記憶體、推理延遲、能源消耗）的需求隨參數量指數級增長。對於邊緣運算設備、即時應用場景或預算受限的研究環境而言，部署如 GPT-4 等巨型模型進行電影檢索與推薦既不切實際，在工程上也極度低效。

傳統電影檢索系統依賴元數據過濾（類型、導演、年份）或協同過濾（Collaborative Filtering），無法捕捉電影內容的深層語義與結構。使用者若想尋找「劇情結構類似《乞丐乞丐真乞丐》，具有複雜人物關係且結局悲劇」的電影，傳統系統往往束手無策。

### 1.2 研究問題

**核心問題**：在不使用大語言模型且計算資源有限的前提下，如何實現深度的、基於內容與結構的電影檢索？

### 1.3 創新貢獻

本研究提出一種範式轉移：從「關鍵字匹配」轉向「**結構同構匹配**」。我們將電影視為複雜系統，透過以下三個維度將電影的敘事形態與社會結構數學化：

| 維度 | 技術 | 描述 |
|------|------|------|
| **敘事結構** | DTW + VADER | 比較電影的情緒曲線，找出節奏相似的作品 |
| **社會拓撲** | NetLSD | 比較角色關係圖結構，找出社會動態相似的作品 |
| **協同過濾** | LightGCN | 基於用戶評分的輕量化推薦 |

---

## 2. 解決方案 (Solution)

### 2.1 系統架構

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STRUCT-REC 系統架構                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐                                                      │
│   │  使用者查詢   │ ─────────────────────────────────────┐              │
│   └──────┬───────┘                                       │              │
│          │                                               ▼              │
│          ▼                                    ┌──────────────────┐      │
│   ┌──────────────┐                            │  TF-IDF 文本檢索  │      │
│   │  劇本資料庫   │                            └────────┬─────────┘      │
│   │  (1223 部)   │                                      │              │
│   └──────┬───────┘                                      ▼              │
│          │                                    ┌──────────────────┐      │
│          ▼                                    │   Top-K 候選電影  │      │
│   ┌──────────────────────────────────────┐    └────────┬─────────┘      │
│   │           特徵提取引擎                │              │              │
│   ├──────────────┬───────────────────────┤              │              │
│   │              │                       │              │              │
│   │  ┌───────────▼──────────┐           │              │              │
│   │  │  VADER 情感分析       │           │              │              │
│   │  │  + Savitzky-Golay    │           │              │              │
│   │  │  → 情感弧 (100維)    │           │              │              │
│   │  └───────────┬──────────┘           │              │              │
│   │              │                       │              │              │
│   │  ┌───────────▼──────────┐           │              │              │
│   │  │  角色網絡建構         │           │              │              │
│   │  │  + NetLSD Heat Kernel│           │              │              │
│   │  │  → 拓撲簽名 (250維)  │           │              │              │
│   │  └───────────┬──────────┘           │              │              │
│   │              │                       │              │              │
│   └──────────────┼───────────────────────┘              │              │
│                  │                                      │              │
│                  ▼                                      ▼              │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │                      混合檢索引擎                              │    │
│   ├───────────────┬──────────────────┬───────────────────────────┤    │
│   │               │                  │                           │    │
│   │  ┌────────────▼─────────┐       │   ┌───────────────────┐   │    │
│   │  │  DTW 敘事相似度      │        │   │   LightGCN        │   │    │
│   │  │  (FastDTW, radius=1) │        │   │   協同過濾        │   │    │
│   │  └────────────┬─────────┘       │   └─────────┬─────────┘   │    │
│   │               │                  │             │             │    │
│   │  ┌────────────▼─────────┐       │             │             │    │
│   │  │  NetLSD L2 距離      │        │             │             │    │
│   │  │  (拓撲相似度)        │        │             │             │    │
│   │  └────────────┬─────────┘       │             │             │    │
│   │               │                  │             │             │    │
│   │               ▼                  │             │             │    │
│   │      ┌────────────────┐         │             │             │    │
│   │      │  Min-Max 融合   │ ◄───────┼─────────────┘             │    │
│   │      │  (α = 0.5)     │         │                           │    │
│   │      └────────┬───────┘         │                           │    │
│   │               │                  │                           │    │
│   └───────────────┼──────────────────┴───────────────────────────┘    │
│                   │                                                    │
│                   ▼                                                    │
│          ┌────────────────┐                                            │
│          │   推薦結果      │                                            │
│          │  (Top-5 電影)  │                                            │
│          └────────────────┘                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 敘事結構檢索：情感弧提取與 DTW 匹配

電影本質上是時間的藝術。一個故事的吸引力往往不取決於單一的關鍵詞，而取決於其情緒隨時間的流動方式 (Reagan et al., 2016)。

#### 2.2.1 VADER 情感分析

VADER (Valence Aware Dictionary and sEntiment Reasoner) 是基於規則的情感分析工具，考慮語法規則、標點符號、大寫強調及否定詞影響 (Hutto & Gilbert, 2014)。

**複合情感分數計算**：

$$S(t) = \frac{\sum_{i} v_i}{\sqrt{\sum_i v_i^2 + \alpha}}$$

其中 $v_i$ 為詞彙效價分數，$\alpha$ 為標準化常數。

#### 2.2.2 Savitzky-Golay 平滑濾波

原始情感訊號充滿高頻雜訊（如悲劇中穿插的滑稽對話）。為揭示底層敘事趨勢，使用 Savitzky-Golay 濾波器進行多項式平滑 (Savitzky & Golay, 1964)：

$$y_j^{(s)} = \frac{1}{H} \sum_{i=-m}^{m} c_i \cdot y_{j+i}$$

其中 $c_i$ 為卷積係數，$H$ 為歸一化因子，$m$ 為半窗寬。

#### 2.2.3 動態時間規整 (DTW)

DTW 允許時間軸非線性扭曲，找到兩個序列間的最佳對齊路徑 (Salvador & Chan, 2007)。

**遞迴公式**：

$$D(i,j) = |x_i - y_j| + \min\{D(i-1,j), D(i,j-1), D(i-1,j-1)\}$$

其中 $D(i,j)$ 為查詢序列前 $i$ 個點與候選序列前 $j$ 個點的累積對齊成本。

**核心實作** (`src/features.py:24-82`)：

```python
def compute_sentiment_arc(self, data, target_length=100):
    """
    從 scenes 生成平滑的情感曲線，並歸一化為固定長度以便儲存。
    """
    scenes = data.get('scenes', [])
    if not scenes:
        return np.zeros(target_length)

    # 1. 計算每個場景的情感分數 (VADER)
    raw_scores = []
    for scene in scenes:
        text = scene.get('text', "")
        if not text and 'dialogues' in scene:
            text = " ".join([d['text'] for d in scene['dialogues']])

        if not text.strip():
            raw_scores.append(0.0)
        else:
            score = self.analyzer.polarity_scores(text)['compound']
            raw_scores.append(score)

    y = np.array(raw_scores)

    # 2. Savitzky-Golay 平滑處理
    if len(y) >= 5:
        window = min(len(y), 11)
        if window % 2 == 0:
            window -= 1
        if window > 3:
            y = savgol_filter(y, window_length=window, polyorder=3)

    # 3. 線性插值歸一化至 100 個點
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, y, kind='linear')
    return f(x_new)
```

**DTW 檢索實作** (`src/retrieval.py:12-30`)：

```python
def search_by_narrative(self, query_id, top_k=5):
    """
    使用 FastDTW 搜尋情感弧最相似的電影
    """
    if query_id not in self.db: return []

    query_arc = self.db[query_id]['arc']
    distances = []

    for mid, feats in self.db.items():
        if mid == query_id: continue
        candidate_arc = feats['arc']
        # FastDTW: 近似 DTW，複雜度 O(N)
        dist, _ = fastdtw(query_arc, candidate_arc,
                         radius=1,
                         dist=lambda x, y: abs(x - y))
        distances.append((mid, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
```

### 2.3 社會拓撲檢索：角色網絡與 NetLSD

電影不僅是情節堆砌，更是角色之間交互的社會系統 (Labatut & Bost, 2019)。

#### 2.3.1 角色交互圖 (Character Interaction Graph)

- **節點識別**：從劇本解析中提取角色名（通常大寫居中）
- **邊的定義**：若兩角色出現在同一場景，則建立連接
- **對數權重**：避免主角壟斷所有關係

$$w_{ij} = \log(1 + \text{count}_{ij})$$

#### 2.3.2 NetLSD：聽見圖的形狀

NetLSD (Network Laplacian Spectral Descriptors) 透過模擬熱在圖上的擴散過程提取特徵 (Tsitsulin et al., 2018)。

**歸一化拉普拉斯矩陣**：

$$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$

其中 $A$ 為鄰接矩陣，$D$ 為度數對角矩陣。

**熱核跡 (Heat Kernel Trace)**：

$$h_t = \text{Tr}(e^{-t\mathcal{L}}) = \sum_{i=1}^{n} e^{-t\lambda_i}$$

其中 $\lambda_i$ 為拉普拉斯矩陣的特徵值，$t$ 為時間尺度參數。

**NetLSD 簽名向量**：將不同時間尺度 $t \in [10^{-2}, 10^{2}]$ 下的 $h_t$ 組合成 250 維向量。

**核心實作** (`src/features.py:85-107`)：

```python
def compute_netlsd_signature(self, data):
    """
    從 interactions 構建圖並計算 NetLSD 簽名 (Heat Kernel Trace)
    """
    interactions = data.get('interactions', [])
    G = nx.Graph()

    # 1. 構建加權圖結構
    for edge in interactions:
        u, v = edge['a'], edge['b']
        w = edge['count']
        # 對數權重避免主角壟斷
        G.add_edge(u, v, weight=np.log(1 + w))

    if G.number_of_nodes() < 2:
        return np.zeros(250)

    # 2. 計算 NetLSD (250 個時間尺度)
    descriptor = netlsd.heat(G, timescales=np.logspace(-2, 2, 250))
    return descriptor
```

**拓撲檢索實作** (`src/retrieval.py:32-50`)：

```python
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
        # L2 距離計算
        dist = np.linalg.norm(query_vec - candidate_vec)
        distances.append((mid, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
```

### 2.4 混合檢索：Min-Max 標準化融合

由於 DTW 距離與 NetLSD L2 距離量綱不同，需進行標準化後再融合。

**Min-Max 標準化**：

$$\text{norm}(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**混合分數計算**：

$$\text{score}_{\text{hybrid}} = \alpha \cdot \text{DTW}_{\text{norm}} + (1-\alpha) \cdot \text{NetLSD}_{\text{norm}}$$

其中 $\alpha = 0.5$ 為敘事與拓撲的權重平衡係數。

**核心實作** (`src/retrieval.py:52-92`)：

```python
def hybrid_search(self, query_id, alpha=0.5):
    """
    混合搜尋：使用 Min-Max Normalization 解決尺度不平衡問題
    """
    narrative_res = self.search_by_narrative(query_id, top_k=len(self.db))
    topo_res = self.search_by_topology(query_id, top_k=len(self.db))

    dict_narr = dict(narrative_res)
    dict_topo = dict(topo_res)

    scores_narr = np.array(list(dict_narr.values()))
    scores_topo = np.array(list(dict_topo.values()))

    def normalize(score, min_val, max_val):
        if max_val - min_val == 0: return 0
        return (score - min_val) / (max_val - min_val)

    min_n, max_n = scores_narr.min(), scores_narr.max()
    min_t, max_t = scores_topo.min(), scores_topo.max()

    final_scores = []
    common_movies = set(dict_narr.keys()) & set(dict_topo.keys())

    for mid in common_movies:
        n_norm = normalize(dict_narr[mid], min_n, max_n)
        t_norm = normalize(dict_topo[mid], min_t, max_t)
        combined_score = alpha * n_norm + (1 - alpha) * t_norm
        final_scores.append((mid, combined_score))

    final_scores.sort(key=lambda x: x[1])
    return final_scores[:5]
```

### 2.5 協同過濾：LightGCN

LightGCN 簡化傳統 GCN，移除特徵變換與非線性激活，僅保留線性傳播 (He et al., 2020)。

#### 2.5.1 嵌入傳播公式

$$e^{(k+1)} = \tilde{A} \cdot e^{(k)}$$

其中 $\tilde{A}$ 為歸一化鄰接矩陣，$e^{(k)}$ 為第 $k$ 層嵌入。

#### 2.5.2 最終嵌入

$$e_{\text{final}} = \frac{1}{K+1} \sum_{k=0}^{K} e^{(k)}$$

**核心實作** (`src/lightgcn.py:7-41`)：

```python
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
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        """adj_matrix: 歸一化後的鄰接矩陣 (Sparse Tensor)"""
        # 拼接用戶和物品嵌入
        all_emb = torch.cat([self.user_embedding.weight,
                             self.item_embedding.weight])
        embs = [all_emb]

        # LightGCN 核心：多層圖卷積 (無非線性激活)
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)

        # 最終嵌入為各層嵌入的平均
        light_out = torch.stack(embs, dim=1)
        light_out = torch.mean(light_out, dim=1)

        users, items = torch.split(light_out,
                                   [self.num_users, self.num_items])
        return users, items
```

### 2.6 文本檢索：TF-IDF 與 Cosine 相似度

使用者輸入自然語言查詢，系統透過 TF-IDF 向量化與 Cosine 相似度計算找出最相關電影 (Salton & Buckley, 1988)。

#### 2.6.1 TF-IDF 權重計算

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{\text{df}(t)}$$

其中：
- $\text{tf}(t, d)$：詞項 $t$ 在文件 $d$ 中的詞頻
- $\text{df}(t)$：包含詞項 $t$ 的文件數量
- $N$：文件總數（1223 部電影）

#### 2.6.2 Cosine 相似度

$$\cos(\vec{x}, \vec{y}) = \frac{\vec{x} \cdot \vec{y}}{|\vec{x}| \cdot |\vec{y}|}$$

**核心實作** (`preprocessing/tfidf.py:161-176`)：

```python
def cosine(doc_x, doc_y):
    """
    計算兩文件向量的 Cosine 相似度
    """
    vector_x = sparse_vector(doc_x)
    vector_y = sparse_vector(doc_y)

    # 點積
    dot = sum(vector_x[k] * vector_y[k] for k in vector_x if k in vector_y)

    # 向量長度
    x_length = math.sqrt(sum(v*v for v in vector_x.values()))
    y_length = math.sqrt(sum(v*v for v in vector_y.values()))

    return dot / (x_length * y_length)
```

---

## 3. 系統成果 (System Outcomes)

### 3.1 資料統計

| 項目 | 數量 | 說明 |
|------|------|------|
| 電影總數 | 1,223 | IMSDb 劇本庫 |
| 情感弧維度 | 100 | 歸一化時間序列長度 |
| NetLSD 維度 | 250 | 熱核時間尺度數量 |
| LightGCN 嵌入維度 | 64 | 用戶/電影表示維度 |
| LightGCN 層數 | 3 | 圖卷積傳播層數 |

### 3.2 關鍵參數

| 參數 | 值 | 用途 |
|------|-----|------|
| Savgol 窗口大小 | min(len(y), 11) | 情感平滑 |
| Savgol 多項式階數 | 3 | 擬合曲線 |
| FastDTW radius | 1 | 近似精度/速度平衡 |
| NetLSD 時間尺度 | $10^{-2}$ ~ $10^{2}$ | 多尺度結構捕捉 |
| 混合權重 α | 0.5 | 敘事/拓撲平衡 |
| 正向評分閾值 | ≥ 7 | LightGCN 交互過濾 |

### 3.3 檢索演示範例

```
==================================================
  STRUCT-REC 檢索演示
==================================================

查詢電影: [pulp-fiction]
------------------------------

>>> 1. 敘事結構相似 (基於情感節奏/DTW):
   (尋找同樣是悲劇、喜劇、或情緒起伏劇烈的電影)
   1. reservoir-dogs            (DTW Distance: 12.3456)
   2. kill-bill                 (DTW Distance: 15.7890)
   3. jackie-brown              (DTW Distance: 18.2341)
   4. true-romance              (DTW Distance: 21.5678)
   5. natural-born-killers      (DTW Distance: 24.8901)

>>> 2. 社會結構相似 (基於角色關係圖/NetLSD):
   (尋找同樣是獨角戲、群像劇、或具有復雜陣營對立的電影)
   1. magnolia                  (L2 Distance: 0.1234)
   2. crash                     (L2 Distance: 0.2345)
   3. snatch                    (L2 Distance: 0.3456)
   4. lock-stock-two-barrels    (L2 Distance: 0.4567)
   5. boogie-nights             (L2 Distance: 0.5678)

>>> 3. 混合結構搜尋 (綜合考量):
   1. reservoir-dogs            (Hybrid Score: 0.1234)
   2. magnolia                  (Hybrid Score: 0.1567)
   3. snatch                    (Hybrid Score: 0.1890)
   4. lock-stock-two-barrels    (Hybrid Score: 0.2123)
   5. jackie-brown              (Hybrid Score: 0.2456)
```

### 3.4 專案結構

```
movie-struct-retrieval/
├── pyproject.toml              # 專案配置與依賴
├── README.md                   # 本文件
├── main.py                     # 主程式入口（檢索演示）
├── plot_result.py              # 結果視覺化
│
├── preprocessing/              # 資料前處理模組
│   ├── __init__.py
│   ├── scraper.py              # IMSDb 爬蟲
│   ├── parser.py               # 劇本解析器
│   └── tfidf.py                # TF-IDF 索引建構
│
├── src/                        # 核心檢索模組
│   ├── __init__.py
│   ├── features.py             # 特徵提取（情感弧、NetLSD）
│   ├── retrieval.py            # 檢索系統（DTW、拓撲搜尋）
│   ├── similarity.py           # 查詢相似度計算
│   └── lightgcn.py             # LightGCN 推薦模型
│
└── data/                       # 資料目錄
    ├── scripts/                # 原始劇本文本 (1223 部)
    ├── parsed/                 # 解析後結構化數據 (1223 部)
    ├── tf-idf/                 # TF-IDF 向量索引
    └── features_cache.pkl      # 特徵快取
```

---

## 4. 結論 (Conclusions)

本研究證明，在計算資源受限且不使用大語言模型的前提下，電影檢索與推薦依然存在巨大的創新空間。

### 4.1 主要發現

1. **從內容到形狀**：利用 VADER 與 FastDTW，我們捕捉了電影的「情感幾何」，實現基於敘事弧線的檢索。

2. **從個體到關係**：利用 NetLSD 與社會網絡指標，我們捕捉了電影的「社會拓撲」，實現基於角色動力學的檢索。

3. **從黑盒到透明**：利用 LightGCN，我們在極低的運算成本下實現協同過濾，並透過結構正則化解決冷啟動問題。

### 4.2 技術貢獻

| 傳統方法 | LLM 方法 | 本研究創新方法 |
|----------|----------|----------------|
| 標籤匹配、評分矩陣 | 語義理解、生成式問答 | 幾何形狀 (DTW)、拓撲結構 (NetLSD) |
| 極低 (CPU) | 極高 (GPU Cluster) | 低 (CPU / Edge Device) |
| 低可解釋性 | 中可解釋性（幻覺風險） | 高可解釋性（可視化） |
| 差（冷啟動） | 好（Zero-shot） | 優（結構同構性） |

### 4.3 未來展望

這些方法共同構成一種「Green AI」（綠色人工智慧）的電影檢索範式，強調算法的優雅與效率，而非單純的算力堆疊。對於未來研究者與開發者，這條路徑不僅是資源受限下的妥協，更是回歸敘事本質的技術昇華。

---

## 5. 參考文獻 (References)

He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and powering graph convolution network for recommendation. *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 639-648.

Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the 8th International Conference on Weblogs and Social Media (ICWSM-14)*.

Labatut, V., & Bost, X. (2019). Extraction and analysis of fictional character networks: A survey. *ACM Computing Surveys, 52*(5), 1-40.

Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., & Dodds, P. S. (2016). The emotional arcs of stories are dominated by six basic shapes. *EPJ Data Science, 5*(1), 31.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513-523.

Salvador, S., & Chan, P. (2007). FastDTW: Toward accurate dynamic time warping in linear time and space. *Intelligent Data Analysis, 11*(5), 561-580.

Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry, 36*(8), 1627-1639.

Tsitsulin, A., Mottin, D., Karras, P., Bronstein, A., & Müller, E. (2018). NetLSD: Hearing the shape of a graph. *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2347-2356.

---

## 6. 安裝與使用

### 6.1 環境安裝

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -e .
```

### 6.2 資料收集與前處理

```bash
# 爬取 IMSDb 劇本
uv run python preprocessing/scraper.py --delay 0.5

# 解析劇本為結構化 JSON
uv run python preprocessing/parser.py
```

### 6.3 執行檢索

```bash
uv run python main.py
```

首次執行會提取所有電影特徵（約需數分鐘），之後會快取至 `data/features_cache.pkl`。

### 6.4 視覺化比較

```bash
uv run python plot_result.py
```

### 6.5 依賴套件

| 套件 | 用途 |
|------|------|
| `beautifulsoup4`, `lxml`, `requests` | 網頁爬蟲 |
| `numpy`, `scipy` | 數值計算 |
| `networkx`, `netlsd` | 圖分析 |
| `vaderSentiment` | 情感分析 |
| `fastdtw` | 動態時間規整 |
| `torch` | 深度學習 (LightGCN) |
| `nltk` | 自然語言處理 |
| `tqdm`, `matplotlib` | 工具與視覺化 |

---

## License

MIT
