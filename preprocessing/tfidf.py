import re
import math
import glob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TFIDF_DIR = DATA_DIR / "tf-idf"

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stemmer = PorterStemmer()

    # Step 1: Lowercasing
    text = text.lower()

    # Step 2: Remove Punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Step 3: Tokenization
    tokens = word_tokenize(text)

    # Step 4: Remove Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 5: Stemming
    processed_tokens = [stemmer.stem(token) for token in tokens]

    return processed_tokens


def extract_unique_term(tokens: list):
    term_frequency = {}

    for t in tokens:
        term_frequency[t] = term_frequency.get(t, 0) + 1
    
    unique_terms = list(term_frequency.keys())

    return unique_terms, term_frequency


def preporcess_scripts(folder_path, txt_files):
    processed_dictionary = []
    doc_terms = {}

    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()

        processed_tokens = preprocess_text(document)

        # unique term extraction for every script
        unique_terms, _ = extract_unique_term(processed_tokens)


        processed_dictionary += unique_terms
        processed_dictionary.sort()

        # store the unique terms extracted from each document
        file_id = file_path.replace(folder_path + "/", "")
        doc_terms[file_id] = unique_terms

    return processed_dictionary, doc_terms


def build_dictionary(processed_dictionary):
    dictionary, document_frequency = extract_unique_term(processed_dictionary)

    with open(f"{TFIDF_DIR}/dictionary.txt", "w") as f:
        f.write("t_index   term   df\n")
        t_index = 1
        for k, v in list(document_frequency.items()):
            f.write(f"{t_index}      {k}      {v}\n")
            t_index += 1

    print(f"Result saved to dictionary.txt ({len(dictionary)} unique terms)")


def dictionary_lookup():
    with open(f"{TFIDF_DIR}/dictionary.txt", "r") as f:
        terms_info = [line.strip() for line in f if line.strip()]
        terms_info = terms_info[1:] # skip header

    dictionary_info = {}
    for term_info in terms_info:
        t = term_info.split()
        dictionary_info[t[1]] = {
            "t_index": t[0],
            "df": t[2]
        }

    return dictionary_info


def compute_tf(doc_unique_terms):
    term_frequency = {}
    
    for t in doc_unique_terms:
        term_frequency[t] = term_frequency.get(t, 0) + 1

    return term_frequency

def build_vector(term_frequency, dictionary_info, N, doc_name):
    vector = []
    for t, tf in term_frequency.items():
        if t not in dictionary_info:
            continue   # ignore OOV term
        
        df = dictionary_info[t].get('df')
        idf = math.log(N / int(df))
        tfidf = round(tf*idf, 4)
        
        vector.append({
            "t_index": dictionary_info[t].get('t_index'),
            "tf-idf": tfidf
        })

    with open(f"{TFIDF_DIR}/{doc_name}", "w") as f:
        f.write(f"{len(vector)}\nt_index   tf-idf\n")
        for v in vector:
            f.write(f"{v['t_index']}   {v['tf-idf']}\n")


def build_query_vector(query, dictionary_info, N):
    processed_tokens = preprocess_text(query)

    unique_terms, _ = extract_unique_term(processed_tokens)

    term_frequency = compute_tf(unique_terms)

    build_vector(term_frequency, dictionary_info, N, "user_query.txt")


# Sparse Vector Construction
def sparse_vector(doc):
    # get tf-idf vector
    with open(f"{TFIDF_DIR}/{doc}", "r") as f:
        lines = [line.strip() for line in f]
        lines = lines[2:]

    vector = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue

        t_index = int(parts[0])
        tfidf = float(parts[1])

        vector[t_index] = tfidf
    
    return vector

# Cosine Similarity Calculation
def cosine(doc_x, doc_y):
    """
    Compute cosine similarity between two document vectors.
    Input: document filenames (e.g., '1.txt', '2.txt')
    Output: cosine similarity (float between 0 and 1)
    """
    vector_x = sparse_vector(doc_x)
    vector_y = sparse_vector(doc_y)
    dot = sum(vector_x[k] * vector_y[k] for k in vector_x if k in vector_y)
    
    x_length = math.sqrt(sum(v*v for v in vector_x.values()))
    y_length = math.sqrt(sum(v*v for v in vector_y.values()))
    
    cosine = dot/(x_length*y_length)

    return cosine

# main function
def build_index():
    """
    Build TF-IDF dictionary and document vectors
    """
    
    # Documents
    folder_path = f"{DATA_DIR}/scripts"
    txt_files = glob.glob(f"{folder_path}/*.txt")
    N = len(txt_files)  # N = 1223

    processed_dictionary, doc_terms = preporcess_scripts(folder_path, txt_files)
    build_dictionary(processed_dictionary)

    dictionary_info = dictionary_lookup()

    doc_terms = dict(sorted(doc_terms.items()))
    doc_term_frequency = {}

    # Iterate through all documents to compute term frequency (TF)
    for doc, terms in doc_terms.items():
        term_frequency = compute_tf(terms)
        doc_term_frequency[doc] = term_frequency


    for d, tfs in doc_term_frequency.items():
        build_vector(tfs, dictionary_info, N, d)


if __name__ == "__main__":
    build_index()