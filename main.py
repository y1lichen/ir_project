import os
import json
import pickle
import time
import numpy as np
from tqdm import tqdm

from src.features import FeatureExtractor
from src.retrieval import StructRetrieval
# from src.lightgcn import SimpleLightGCN, DataLoader # 若需訓練推薦模型時開啟
from src.tfidf_similarity import QuerySimilarity
from src.bert_similarity import BertQuerySimilarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PARSED_DIR = os.path.join(DATA_DIR, 'parsed')
CACHE_FILE = os.path.join(DATA_DIR, 'features_cache.pkl')

def load_or_process_features(force_update=False):
    """
    檢查是否有快取特徵，若無則執行批次提取。
    """
    if os.path.exists(CACHE_FILE) and not force_update:
        print(f"[INFO] 發現特徵快取檔案: {CACHE_FILE}")
        print("[INFO] 正在載入...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("[INFO] 未發現快取或強制更新，開始特徵提取流程...")
    print(f"[INFO] 掃描目錄: {PARSED_DIR}")
    
    extractor = FeatureExtractor()
    feature_db = {}
    
    # 獲取所有 json 檔案
    files = [f for f in os.listdir(PARSED_DIR) if f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError(f"錯誤: 在 {PARSED_DIR} 找不到任何 .json 檔案。請檢查資料路徑。")

    print(f"[INFO] 找到 {len(files)} 部電影，開始處理 (這可能需要幾分鐘)...")
    
    # 使用 tqdm 顯示進度條
    start_time = time.time()
    for filename in tqdm(files, desc="Extracting Features"):
        movie_id = filename.replace('.json', '')
        file_path = os.path.join(PARSED_DIR, filename)
        
        try:
            # 1. 載入資料
            data = extractor.load_parsed_data(file_path)
            
            # 2. 提取情感弧 (Narrative Geometry)
            arc = extractor.compute_sentiment_arc(data)
            
            # 3. 提取拓撲簽名 (Social Topology)
            netlsd_sig = extractor.compute_netlsd_signature(data)
            
            # 4. 存入字典
            feature_db[movie_id] = {
                'title': data.get('id', movie_id), # 若有 title 欄位更好
                'arc': arc,
                'netlsd': netlsd_sig
            }
            
        except Exception as e:
            print(f"\n[WARN] 處理 {filename} 時發生錯誤: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"[INFO] 處理完成，耗時 {elapsed:.2f} 秒。成功處理 {len(feature_db)} 部電影。")
    
    # 儲存快取
    print(f"[INFO] 正在儲存特徵至 {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(feature_db, f)
        
    return feature_db

def run_demo_search(feature_db, query_id):
    """
    執行並展示檢索結果
    """
    print("\n" + "="*50)
    print(f"  STRUCT-REC 檢索演示")
    print("="*50)
    
    if query_id not in feature_db:
        print(f"[ERROR] 資料庫中找不到 ID: {query_id}")
        # 隨機選一個存在的 ID
        query_id = list(feature_db.keys())[0]
        print(f"[INFO] 自動切換至隨機電影: {query_id}")

    retriever = StructRetrieval(feature_db)
    
    print(f"\n查詢電影: [{query_id}]")
    print("-" * 30)

    # 1. 敘事結構搜尋 (Sentiment Arc / DTW)
    print("\n>>> 1. 敘事結構相似 (基於情感節奏/DTW):")
    print("   (尋找同樣是悲劇、喜劇、或情緒起伏劇烈的電影)")
    narrative_results = retriever.search_by_narrative(query_id, top_k=5)
    for rank, (mid, score) in enumerate(narrative_results, 1):
        print(f"   {rank}. {mid:<25} (DTW Distance: {score:.4f})")

    # 2. 社會拓撲搜尋 (Social Network / NetLSD)
    print("\n>>> 2. 社會結構相似 (基於角色關係圖/NetLSD):")
    print("   (尋找同樣是獨角戲、群像劇、或具有復雜陣營對立的電影)")
    topo_results = retriever.search_by_topology(query_id, top_k=5)
    for rank, (mid, score) in enumerate(topo_results, 1):
        print(f"   {rank}. {mid:<25} (L2 Distance: {score:.4f})")

    # 3. 混合搜尋 (Hybrid)
    print("\n>>> 3. 混合結構搜尋 (綜合考量):")
    hybrid_results = retriever.hybrid_search(query_id)
    for rank, (mid, score) in enumerate(hybrid_results, 1):
        print(f"   {rank}. {mid:<25} (Hybrid Score: {score:.4f})")

def get_recommendations(feature_db, query_id, top_k=5):
    """
    回傳該電影的三種結構相似推薦結果
    """
    retriever = StructRetrieval(feature_db)

    narrative = retriever.search_by_narrative(query_id, top_k=top_k)
    topology = retriever.search_by_topology(query_id, top_k=top_k)
    hybrid = retriever.hybrid_search(query_id)

    return {
        "narrative": narrative,
        "topology": topology,
        "hybrid": hybrid
    }


def display_result(topk_similarity, feature_db):
    print("\n" + "="*50)
    print("根據您的查詢內容，我們找到以下相關電影：")
    print("="*50)

    for i, mid in enumerate(topk_similarity, 1):
        print(f"{i}. {mid}")

    print("\n進一步為您推薦每部電影的相似作品 ;)")

    for target_movie in topk_similarity:
        results = get_recommendations(feature_db, target_movie)

        print("\n" + "-"*50)
        print(f"以電影 [{target_movie}] 為核心的相似推薦")
        print("-"*50)

        print("\n【敘事結構相似（情感節奏 / DTW）】")
        for rank, (mid, score) in enumerate(results["narrative"], 1):
            print(f"  {rank}. {mid:<25} (DTW: {score:.4f})")

        print("\n【社會結構相似（角色網絡 / NetLSD）】")
        for rank, (mid, score) in enumerate(results["topology"], 1):
            print(f"  {rank}. {mid:<25} (L2: {score:.4f})")

        print("\n【混合結構推薦（Hybrid）】")
        for rank, (mid, score) in enumerate(results["hybrid"], 1):
            print(f"  {rank}. {mid:<25} (Score: {score:.4f})")


def main():
    # 步驟 1: 準備特徵庫
    # 如果是第一次執行，這裡會比較久
    feature_db = load_or_process_features()

    # 步驟 2: 使用者輸入關鍵字，查找電影
    user_query = input("請輸入查詢文字：")

    # use tfidf-based
    # tfidf_similarity = QuerySimilarity(DATA_DIR)
    # tfidf_topk = tfidf_similarity.retrieve_top_k(user_query)

    # use bert model based
    bert_similarity = BertQuerySimilarity(DATA_DIR)
    bert_topK = bert_similarity.retrieve_top_k(user_query, k=5)
    
    # 步驟 3: 針對找到的電影進行推薦 & display final result
    # display_result(tfidf_topk, feature_db) # tfidf-based
    display_result(bert_topK, feature_db)

    # 步驟 4 (可選): 這裡可以加入 LightGCN 的訓練或推論代碼
    # print("\n[INFO] LightGCN 推薦模型初始化中... (省略)")

if __name__ == "__main__":
    main()