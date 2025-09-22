"""
Vector Space Model (VSM) Implementation with Novel Output Features
==================================================================

Features:
- TF-IDF weighting (lnc.ltc)
- Cosine similarity for document ranking
- Soundex algorithm for phonetic query expansion
- Advanced text preprocessing with lemmatization
- NEW: Bar chart of scores for each search
- NEW: Google-style snippets with highlighted query terms
"""

import os
import math
import string
from collections import defaultdict, Counter

import spacy
from nltk.corpus import stopwords

# ---------------- NEW LIBRARIES ----------------
import matplotlib.pyplot as plt
from termcolor import colored  # pip install termcolor once in Colab
# ------------------------------------------------

# Load spaCy language model and NLTK stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if (token.is_alpha and
            token.text not in stop_words and
            len(token.text) > 2 and
            not token.is_space):
            tokens.append(token.lemma_)
    return tokens

def preprocess_query(query):
    doc = nlp(query.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

# Soundex implementation
def soundex(word):
    codes = {
        "B": "1","F": "1","P": "1","V": "1",
        "C": "2","G": "2","J": "2","K": "2","Q": "2","S": "2","X": "2","Z": "2",
        "D": "3","T": "3",
        "L": "4",
        "M": "5","N": "5",
        "R": "6"
    }
    if not word: return "0000"
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

# Searcher
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
        q_term_weights = {term: w for term, w in q_raw.items() if w is not None}
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

# ----------------- NEW FEATURES -----------------

def plot_scores(results):
    """Bar chart of search scores."""
    docIDs = [d for d,_ in results]
    scores = [s for _,s in results]
    plt.barh(docIDs[::-1], scores[::-1])
    plt.xlabel("Score")
    plt.ylabel("Document")
    plt.title("Search Result Scores")
    plt.show()

def snippet_with_highlights(text, query_terms, length=150):
    """Return snippet of text with query terms highlighted."""
    t_lower = text.lower()
    for t in query_terms:
        idx = t_lower.find(t)
        if idx != -1:
            start = max(0, idx - length//2)
            end = min(len(text), idx + length//2)
            snippet = text[start:end].replace("\n", " ")
            for qt in query_terms:
                snippet = snippet.replace(qt, colored(qt, 'red'))
                snippet = snippet.replace(qt.capitalize(), colored(qt.capitalize(), 'red'))
            return snippet + "..."
    return text[:length] + "..."

# ------------------------------------------------

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
    folder = "./Corpus"  # path to your corpus folder
    print("Loading document corpus...")
    docs = load_corpus_from_folder(folder)
    print(f"Loaded {len(docs)} documents.")
    if len(docs) == 0:
        print("Error: No documents found.")
        exit()

    print("Building inverted index...")
    idx = VSMIndexer()
    idx.index_documents(docs)
    print("Index construction completed.")
    searcher = VSMSearcher(idx)

    print("\n" + "="*60)
    print("Vector Space Model Search Engine")
    print("="*60)
    print("Now with bar charts and highlighted snippets!")
    print("="*60)

    while True:
        try:
            q = input("\nEnter query (or 'exit'): ")
            if q.lower() == "exit":
                print("Thank you for using the VSM Search Engine!")
                break
            results = searcher.search(q)
            if not results:
                print("No results found.")
            else:
                print(f"\nFound {len(results)} relevant documents:")
                print("-"*50)
                tokens = tokenize(q)
                for i,(docID,score) in enumerate(results,1):
                    print(f"{i:2d}. {docID:<25} (Relevance: {score:.8f})")
                    print(snippet_with_highlights(docs[docID], tokens))
                    print()
                # plot the scores visually
                plot_scores(results)
        except KeyboardInterrupt:
            print("\n\nSearch session terminated by user.")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            print("Please try again with a different query.")
