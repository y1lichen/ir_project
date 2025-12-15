# Data 目錄說明

本目錄包含從 IMSDb 爬取的電影劇本數據，共 **1223** 部電影。

## 檔案結構

```
data/
├── movies.json          # 電影元數據 + 用戶評論
├── scripts/             # 原始劇本文本 (1223 個 .txt)
└── parsed/              # 解析後結構化數據 (1223 個 .json)
```

---

## 1. movies.json

電影元數據與用戶評論，用於 **LightGCN 協同過濾推薦**。

### 結構
```json
{
  "id": "pulp-fiction",
  "title": "Pulp Fiction",
  "detail_url": "/Movie Scripts/Pulp-Fiction Script.html",
  "writers": ["Quentin Tarantino"],
  "genres": ["Crime", "Drama"],
  "avg_rating": 9.37,
  "script_url": "/scripts/Pulp-Fiction.html",
  "has_script": true,
  "reviews": [
    {"user": "Ralph", "rating": 10, "text": "Best script ever..."},
    {"user": "Jane", "rating": 8, "text": "Great dialogue..."}
  ]
}
```

### 欄位說明
| 欄位 | 說明 |
|------|------|
| `id` | 電影唯一識別碼（與檔名對應） |
| `title` | 電影標題 |
| `writers` | 編劇列表 |
| `genres` | 類型標籤 |
| `avg_rating` | 平均評分 (1-10) |
| `reviews` | 用戶評論列表（含 user, rating, text） |

### 用途
- 建立 **用戶-電影互動矩陣** → LightGCN 推薦系統

---

## 2. scripts/*.txt

原始劇本純文本，保留標準劇本格式。

### 範例 (scripts/pulp-fiction.txt)
```
INT. COFFEE SHOP - MORNING

PUMPKIN and HONEY BUNNY sit in a booth...

                    PUMPKIN
          Forget it. It's too risky.

                    HONEY BUNNY
          You always say that...
```

### 用途
- 原始資料保存
- 重新解析時的來源

---

## 3. parsed/*.json

解析後的結構化劇本數據，用於 **DTW 情緒曲線** 和 **NetLSD 角色關係圖**。

### 結構
```json
{
  "id": "pulp-fiction",
  "total_scenes": 85,
  "total_characters": 42,
  "total_dialogues": 560,
  "characters": ["JULES", "VINCENT", "MIA", ...],
  "character_stats": {
    "JULES": {
      "dialogue_count": 120,
      "scene_count": 25,
      "scenes": [1, 2, 5, ...],
      "first_appearance": 1,
      "total_words": 2500
    }
  },
  "interactions": [
    {"a": "JULES", "b": "VINCENT", "count": 18, "scenes": [1, 2, 5, ...]}
  ],
  "scenes": [
    {
      "number": 1,
      "header": "INT. COFFEE SHOP - MORNING",
      "location_type": "INT",
      "location": "COFFEE SHOP",
      "time": "MORNING",
      "text": "場景完整文本...",
      "characters": ["PUMPKIN", "HONEY BUNNY"],
      "dialogues": [
        {"character": "PUMPKIN", "text": "Forget it..."}
      ]
    }
  ]
}
```

### 欄位說明

#### 頂層欄位
| 欄位 | 說明 |
|------|------|
| `total_scenes` | 場景總數 |
| `total_characters` | 角色總數 |
| `total_dialogues` | 對話總數 |
| `characters` | 所有角色名列表 |
| `character_stats` | 各角色統計資訊 |
| `interactions` | 角色共同出現關係 |
| `scenes` | 場景列表 |

#### character_stats 欄位
| 欄位 | 說明 |
|------|------|
| `dialogue_count` | 對話次數 |
| `scene_count` | 出現場景數 |
| `scenes` | 出現的場景編號列表 |
| `first_appearance` | 首次出現的場景編號 |
| `total_words` | 台詞總字數 |

#### interactions 欄位
| 欄位 | 說明 |
|------|------|
| `a`, `b` | 角色對（按字母排序） |
| `count` | 共同出現場景次數 |
| `scenes` | 共同出現的場景編號列表 |

### 用途
- `scenes[].text` / `scenes[].dialogues` → 情緒分析 → **DTW 曲線比對**
- `interactions` → 角色關係圖 → **NetLSD 結構比對**

---