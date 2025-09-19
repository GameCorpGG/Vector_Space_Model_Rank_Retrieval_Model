import os
import math
from collections import defaultdict, Counter

import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def tokenize(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_alpha and token.text not in stop_words:
            tokens.append(token.lemma_)  # Keep lemmas instead of raw words
    return tokens

def preprocess_query(query):
    doc = nlp(query.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return tokens

# Soundex
def soundex(word):
    codes = {
        "B": "1","F": "1","P": "1","V": "1",
        "C": "2","G": "2","J": "2","K": "2","Q": "2","S": "2","X": "2","Z": "2",
        "D": "3","T": "3",
        "L": "4",
        "M": "5","N": "5",
        "R": "6"
    }
    if not word:
        return "0000"
    word = word.upper()
    first = word[0]
    out = first
    prev = codes.get(first, "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code == prev or code == "":
            prev = code
            continue
        out += code
        prev = code
        if len(out) == 4:
            break
    return out.ljust(4, "0")[:4]

# Indexer
class VSMIndexer:
    def __init__(self):
        self.inverted = {}
        self.N = 0
        self.doc_norms = {}
        self.doc_term_norm_weight = defaultdict(dict)
        self.soundex_map = defaultdict(set)
    
    def index_documents(self, docs):
        self.N = len(docs)
        term_doc_tf = defaultdict(lambda: defaultdict(int))
        
        for docID, text in docs.items():
            tokens = tokenize(text)
            counts = Counter(tokens)
            for term, tf in counts.items():
                term_doc_tf[term][docID] = tf
        
        for term, doc_tf_map in term_doc_tf.items():
            postings = sorted(doc_tf_map.items(), key=lambda x: x[0])
            df = len(postings)
            self.inverted[term] = {"df": df, "postings": postings}
            self.soundex_map[soundex(term)].add(term)
        
        for term, info in self.inverted.items():
            for docID, tf in info["postings"]:
                tf_w = 1.0 + math.log10(tf)
                self.doc_norms.setdefault(docID, 0.0)
                self.doc_norms[docID] += tf_w * tf_w
                self.doc_term_norm_weight[docID][term] = tf_w
        
        for docID, sqsum in self.doc_norms.items():
            norm = math.sqrt(sqsum) if sqsum > 0 else 1.0
            self.doc_norms[docID] = norm
            for term, raw_w in self.doc_term_norm_weight[docID].items():
                self.doc_term_norm_weight[docID][term] = raw_w / norm
    
    def get_postings(self, term):
        return self.inverted.get(term, None)
    
    def soundex_terms(self, term):
        return self.soundex_map.get(soundex(term), set())

# Searcher (lnc.ltc)
class VSMSearcher:
    def __init__(self, indexer):
        self.idx = indexer
        self.N = indexer.N
    
    def _query_vector(self, tokens):
        q_counts = Counter(tokens)
        q_weights = {}
        for term, tf_q in q_counts.items():
            tf_q_w = 1.0 + math.log10(tf_q)
            p = self.idx.get_postings(term)
            if p is None:
                q_weights[term] = None
            else:
                df = p["df"]
                idf = math.log10(self.N / df) if df > 0 else 0
                q_weights[term] = tf_q_w * idf
        return q_weights
    
    def search(self, query, top_k=10, soundex_fallback=True):
        tokens = tokenize(query)
        q_raw = self._query_vector(tokens)
        
        q_term_weights = {}
        for term, w in q_raw.items():
            if w is not None:
                q_term_weights[term] = w
        
        if soundex_fallback:
            for term, w in q_raw.items():
                if w is None:
                    tf_q = Counter(tokens)[term]
                    tf_q_w = 1.0 + math.log10(tf_q)
                    for cand in self.idx.soundex_terms(term):
                        p = self.idx.get_postings(cand)
                        if p:
                            df_cand = p["df"]
                            idf_cand = math.log10(self.N / df_cand) if df_cand > 0 else 0
                            q_term_weights[cand] = q_term_weights.get(cand, 0) + tf_q_w * idf_cand
        
        if not q_term_weights:
            return []
        
        q_norm = math.sqrt(sum(w*w for w in q_term_weights.values())) or 1.0
        for t in q_term_weights:
            q_term_weights[t] /= q_norm
        
        scores = defaultdict(float)
        for term, q_w in q_term_weights.items():
            posting = self.idx.get_postings(term)
            if not posting:
                continue
            for docID, _ in posting["postings"]:
                d_w = self.idx.doc_term_norm_weight[docID].get(term, 0.0)
                scores[docID] += q_w * d_w
        
        results = [(docID, s) for docID, s in scores.items() if s > 0.]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

# Load corpus from folder
def load_corpus_from_folder(folder_path):
    docs = {}
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    docs[filename] = f.read()
    return docs

if __name__ == "__main__":
    folder = "C:\\Users\\dell\\Downloads\\Corpus-20230203T210935Z-001\\Corpus"  #The Folder Path of the Corpus(Replace it accordingly)
    docs = load_corpus_from_folder(folder)
    
    print(f"Loaded {len(docs)} documents.")
    
    idx = VSMIndexer()
    idx.index_documents(docs)
    searcher = VSMSearcher(idx)
    
    while True:
        q = input("\nEnter query (or 'exit'): ")
        if q.lower() == "exit":
            break
        results = searcher.search(q)
        if not results:
            print("No results found.")
        else:
            for docID, score in results:
                print(f"Doc {docID} (score={score:.8f})")
