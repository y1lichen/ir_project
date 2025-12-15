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
        if not scenes:
            return np.zeros(target_length)

        # 1. 計算每個場景的情感分數
        raw_scores = []
        for scene in scenes:
            # 優先使用場景完整文本，若無則聚合對話
            text = scene.get('text', "")
            if not text and 'dialogues' in scene:
                text = " ".join([d['text'] for d in scene['dialogues']])
            
            score = self.analyzer.polarity_scores(text)['compound']
            raw_scores.append(score)
        
        # 2. 處理過短的電影
        if len(raw_scores) < 4: 
            # 場景太少無法做 Savitzky-Golay，直接線性插值
            y = np.array(raw_scores)
        else:
            # 3. Savitzky-Golay 平滑濾波 (去除高頻雜訊，保留敘事趨勢)
            # window_length 必須小於數據長度且為奇數
            window = min(len(raw_scores) if len(raw_scores) % 2 == 1 else len(raw_scores)-1, 11)
            y = savgol_filter(raw_scores, window_length=window, polyorder=3)

        # 4. 長度歸一化 (Interpolation) -> 讓所有電影都在 0% - 100% 的進度條上比較
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, y, kind='cubic')
        return f(x_new)

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