import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_comparison(movie_a_id, movie_b_id, cache_path='data/features_cache.pkl'):
    with open(cache_path, 'rb') as f:
        db = pickle.load(f)
    
    if movie_a_id not in db or movie_b_id not in db:
        print("Movie ID not found.")
        return

    arc_a = db[movie_a_id]['arc']
    arc_b = db[movie_b_id]['arc']
    
    # 畫圖
    plt.figure(figsize=(10, 5))
    plt.plot(arc_a, label=f"{movie_a_id} (Query)", linewidth=2)
    plt.plot(arc_b, label=f"{movie_b_id} (Result)", linestyle='--', linewidth=2)
    
    plt.title(f"Narrative Arc Comparison: {movie_a_id} vs {movie_b_id}")
    plt.xlabel("Narrative Progress (%)")
    plt.ylabel("Sentiment Valence (Positive/Negative)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"narrative_comparison_{movie_a_id}_vs_{movie_b_id}.png")

if __name__ == "__main__":
    # 測試畫出 空軍一號 vs 刀鋒戰士
    plot_comparison('air-force-one', 'blade-trinity')