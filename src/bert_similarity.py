import os
import glob
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

class BertQuerySimilarity:
    def __init__(self, DATA_DIR, model_name='all-MiniLM-L6-v2'):
        """
        model_name: 'all-MiniLM-L6-v2' (輕量快) / 'all-mpnet-base-v2' (準度高)
        """
        self.SCRIPTS_DIR = os.path.join(DATA_DIR, 'scripts')
        self.EMBEDDING_DIR = os.path.join(DATA_DIR, 'embeddings')
        self.model_name = model_name
        
        print(f"Loading BERT model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        self.txt_files = glob.glob(f"{self.SCRIPTS_DIR}/*.txt")
        self.file_ids = [os.path.basename(f).replace(".txt", "") for f in self.txt_files]
        self.embeddings_path = os.path.join(self.EMBEDDING_DIR, f"corpus_embeddings_{model_name}.npy")
        
        if not os.path.exists(self.EMBEDDING_DIR):
            os.makedirs(self.EMBEDDING_DIR)
            
        self.corpus_embeddings = self._load_or_build_index()

    def _read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _load_or_build_index(self):
        if os.path.exists(self.embeddings_path):
            print("Loading existing embeddings...")
            return np.load(self.embeddings_path)
        
        print("Building BERT index (encoding documents)... This may take a while.")
        
        # read all documents
        documents = [self._read_file(f) for f in self.txt_files]
        
        # Encode: turn documents to vector
        # convert_to_numpy=True -> return numpy array
        embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        
        # 儲存向量供下次使用
        np.save(self.embeddings_path, embeddings)
        print(f"Index built and saved to {self.embeddings_path}")
        
        return embeddings

    def retrieve_top_k(self, user_query, k=5):
        """
        search top k similar documents(movie)
        """
        # turn user query to vector
        query_embedding = self.model.encode(user_query, convert_to_numpy=True)
        
        # calculate cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # sort and retrieve Top K
        top_results = torch.topk(cos_scores, k=k) # use torch.topk to find top k quickly
        
        results = []
        
        for score, idx in zip(top_results.values, top_results.indices):
            doc_id = self.file_ids[idx]
            score_val = score.item()
            results.append((doc_id, score_val))
            # print(f"Document: {doc_id} \t Score: {score_val:.4f}")
            
        # return movie's id list
        return [r[0] for r in results]


# --- Example ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    engine = BertQuerySimilarity(DATA_DIR)
    results = engine.retrieve_top_k("A romantic movie about love and destiny", k=3)