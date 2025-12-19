import json
from src.evaluator import Evaluator
from main import load_or_process_features, StructRetrieval

feature_db = load_or_process_features()
retriever = StructRetrieval(feature_db)

# read 原本的 metadata JSON 檔案
with open('data/movies.json', 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)

# evaluator = Evaluator(feature_db, metadata_list)
evaluator = Evaluator(feature_db, metadata_list, threshold_mode="strict")

# Precision Recall Curve
evaluator.plot_11point_curve(retriever)


# 如果要用 Precision at K
# print("----- Evaluate by narrative ------")
# evaluator.precision_at_k(retriever, method="narrative", k=1223)
# print("----- Evaluate by topology ------")
# evaluator.precision_at_k(retriever, method="topology", k=1223)
# print("----- Evaluate by hybrid ------")
# evaluator.precision_at_k(retriever, k=1223)