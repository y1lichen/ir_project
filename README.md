# STRUCT-REC: 電影結構化檢索系統

基於**敘事結構**、**社會拓撲**與**協同過濾**的電影推薦系統。

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    STRUCT-REC 系統                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  DTW/情感弧  │   │ NetLSD/拓撲 │   │  LightGCN   │       │
│  │  敘事結構    │   │  社會結構   │   │  協同過濾    │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         └────────────┬────┴────────────────┘               │
│                      ▼                                      │
│              ┌──────────────┐                               │
│              │  混合檢索器   │                               │
│              └──────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 三種檢索方法

| 方法 | 技術 | 描述 |
|------|------|------|
| **敘事結構** | DTW + VADER | 比較電影的情緒曲線，找出節奏相似的作品 |
| **社會拓撲** | NetLSD | 比較角色關係圖結構，找出社會動態相似的作品 |
| **協同過濾** | LightGCN | 基於用戶評分的推薦 |

## 專案結構

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
│   └── parser.py               # 劇本解析器
│
├── src/                        # 核心檢索模組
│   ├── __init__.py
│   ├── features.py             # 特徵提取（情感弧、NetLSD）
│   ├── retrieval.py            # 檢索系統（DTW、拓撲搜尋）
│   └── lightgcn.py             # LightGCN 推薦模型
│
└── data/                       # 資料目錄
    ├── README.md               # 數據說明
    ├── movies.json             # 電影元數據 + 用戶評論
    ├── scripts/                # 原始劇本文本 (1223 部)
    ├── parsed/                 # 解析後結構化數據 (1223 部)
    └── features_cache.pkl      # 特徵快取
```

## 安裝

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -e .
```

## 使用方式

### 1. 資料收集與前處理

```bash
# 爬取 IMSDb 劇本
uv run python preprocessing/scraper.py --delay 0.5

# 解析劇本為結構化 JSON
uv run python preprocessing/parser.py
```

### 2. 執行檢索

```bash
uv run python main.py
```

首次執行會提取所有電影特徵（約需數分鐘），之後會快取至 `data/features_cache.pkl`。

### 3. 視覺化比較

```bash
uv run python plot_result.py
```

### 輸出範例

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
   ...

>>> 2. 社會結構相似 (基於角色關係圖/NetLSD):
   (尋找同樣是獨角戲、群像劇、或具有復雜陣營對立的電影)
   1. magnolia                  (L2 Distance: 0.1234)
   2. crash                     (L2 Distance: 0.2345)
   ...

>>> 3. 混合結構搜尋 (綜合考量):
   1. reservoir-dogs            (Hybrid Score: 0.1234)
   ...
```

## 技術細節

### 情感弧提取 (src/features.py)

1. 使用 **VADER** 計算每個場景的情感分數
2. 使用 **Savitzky-Golay** 濾波平滑曲線
3. 插值歸一化為 100 個點，便於跨電影比較

### NetLSD 簽名 (src/features.py)

1. 從角色互動建立加權圖（對數權重避免主角壟斷）
2. 計算 **Heat Kernel Trace** 描述子（250 維）
3. 捕捉從局部到全局的多尺度結構特徵

### DTW 檢索 (src/retrieval.py)

- 使用 **FastDTW** 計算時間序列距離
- 找出情緒節奏最相似的電影

### LightGCN (src/lightgcn.py)

- 多層圖卷積（無非線性激活）
- 用戶-電影二部圖上的嵌入傳播
- 評分 >= 7 視為正向互動

## 資料統計

| 項目 | 數量 |
|------|------|
| 電影總數 | 1223 |
| 劇本文本 | 1223 |
| 解析數據 | 1223 |

## 依賴

- `beautifulsoup4`, `lxml`, `requests` - 爬蟲
- `numpy`, `scipy` - 數值計算
- `networkx`, `netlsd` - 圖分析
- `vaderSentiment` - 情感分析
- `fastdtw` - 動態時間規整
- `torch` - 深度學習（LightGCN）
- `tqdm`, `matplotlib` - 工具

## License

MIT
