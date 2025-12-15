import json
import os
import numpy as np
import networkx as nx
import netlsd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

class FeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def load_parsed_data(self, file_path):
        if os.path.getsize(file_path) == 0:
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # --- 敘事幾何學：情感弧 ---
    def compute_sentiment_arc(self, data, target_length=100):
        """
        從 scenes 生成平滑的情感曲線，並歸一化為固定長度以便儲存。
        """
        scenes = data.get('scenes', [])
        # 如果完全沒有場景，回傳全 0
        if not scenes:
            return np.zeros(target_length)

        # 1. 計算每個場景的情感分數
        raw_scores = []
        for scene in scenes:
            text = scene.get('text', "")
            if not text and 'dialogues' in scene:
                text = " ".join([d['text'] for d in scene['dialogues']])
            
            # 若場景全空，給 0 分
            if not text.strip():
                raw_scores.append(0.0)
            else:
                score = self.analyzer.polarity_scores(text)['compound']
                raw_scores.append(score)
        
        y = np.array(raw_scores)
        
        # 2. 安全的平滑處理 (Safe Smoothing)
        # Savitzky-Golay 限制: window_length > polyorder
        # 這裡 polyorder=3，所以 window 至少要 5 (因為要是奇數)
        if len(y) >= 5:
            try:
                # 決定視窗大小：最大 11，若資料長度小於 11 則取資料長度
                window = min(len(y), 11)
                
                # 強制轉為奇數 (如果是偶數就減 1)
                if window % 2 == 0:
                    window -= 1
                
                # 再次檢查是否大於 polyorder (3)
                if window > 3:
                    y = savgol_filter(y, window_length=window, polyorder=3)
            except Exception:
                # 如果 SciPy 還是因為邊界問題報錯，就默默吞掉錯誤，使用原始數據 y
                pass

        # 3. 處理極端情況：如果數據點只有 1 個，無法插值
        if len(y) < 2:
            # 直接填滿整個長度
            val = y[0] if len(y) == 1 else 0.0
            return np.full(target_length, val)

        # 4. 長度歸一化 (Interpolation)
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, target_length)
        
        try:
            f = interp1d(x_old, y, kind='linear') # 改用 linear 比較穩，cubic 在點少時會震盪
            return f(x_new)
        except Exception:
            return np.zeros(target_length)
        
    # --- 社會拓撲學：NetLSD ---
    def compute_netlsd_signature(self, data):
        """
        從 interactions 構建圖並計算 NetLSD 簽名 (Heat Kernel Trace)
        """
        interactions = data.get('interactions', [])
        G = nx.Graph()
        
        # 1. 構建圖結構
        for edge in interactions:
            u, v = edge['a'], edge['b']
            w = edge['count']
            # 使用對數權重避免主角壟斷
            G.add_edge(u, v, weight=np.log(1 + w))
        
        # 處理孤立節點或空圖
        if G.number_of_nodes() < 2:
            return np.zeros(250) # NetLSD 預設返回 250 維

        # 2. 計算 NetLSD
        # timescales 決定了我們要觀察圖結構的「尺度」（從局部到全局）
        # 這裡模擬熱擴散過程
        descriptor = netlsd.heat(G, timescales=np.logspace(-2, 2, 250))
        return descriptor

# --- 批次處理測試 ---
if __name__ == "__main__":
    extractor = FeatureExtractor()
    sample_path = "data/parsed/pulp-fiction.json"  # 假設檔案存在
    
    if os.path.exists(sample_path):
        data = extractor.load_parsed_data(sample_path)
        
        arc = extractor.compute_sentiment_arc(data)
        topo = extractor.compute_netlsd_signature(data)
        
        print(f"Movie: {data['id']}")
        print(f"Sentiment Arc Shape: {arc.shape}") # 應該是 (100,)
        print(f"Topology Sig Shape: {topo.shape}") # 應該是 (250,)