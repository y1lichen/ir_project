import os
import glob
from preprocessing.tfidf import build_query_vector, dictionary_lookup, cosine, build_index
from pathlib import Path

class QuerySimilarity:
    def __init__(self, DATA_DIR):
        # Documents
        self.SCRIPTS_DIR = os.path.join(DATA_DIR, 'scripts')
        self.txt_files = glob.glob(f"{self.SCRIPTS_DIR}/*.txt")
        self.N = len(self.txt_files)  # N = 1223

        # if os.path.join(DATA_DIR, 'tf-idf'):
        #     build_index()

        TFIDF_DIR = os.path.join(DATA_DIR, 'tf-idf')
        if not os.path.exists(TFIDF_DIR):
            os.makedirs(TFIDF_DIR)

        # if not os.path.exists(os.path.join(DATA_DIR, "tf-idf/dictionary.txt")):
            print("Computing TF-IDF....")
            build_index()

        self.dictionary_info = dictionary_lookup()


    def retrieve_top_k(self, user_query, k=5):
        build_query_vector(user_query, self.dictionary_info, self.N)

        scores = {}
        for file_path in self.txt_files:
            # store the unique terms extracted from user query
            file_id = file_path.replace(f"{self.SCRIPTS_DIR}/", "")
            movie_name = file_id.replace(".txt", "")
            score = cosine("user_query.txt", file_id)
            scores[movie_name] = score

        scores = dict(sorted(scores.items(), key=lambda x : x[1], reverse=True))

        topk = list(scores.keys())[:k]

        return topk