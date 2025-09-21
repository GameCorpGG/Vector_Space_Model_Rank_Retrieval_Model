"""
Vector Space Model (VSM) Implementation for Information Retrieval
================================================================

This module implements a complete Vector Space Model search engine with the following features:
- TF-IDF weighting scheme (lnc.ltc): log-normalized term frequency for documents, 
  log-normalized term frequency with IDF for queries
- Cosine similarity for document ranking
- Soundex algorithm for phonetic query expansion
- Advanced text preprocessing with lemmatization
"""

import os
import math
import string
from collections import defaultdict, Counter

import spacy
from nltk.corpus import stopwords

# Initialize spaCy language model for advanced NLP preprocessing
# en_core_web_sm provides tokenization, POS tagging, and lemmatization
nlp = spacy.load("en_core_web_sm")

# Load English stopwords from NLTK corpus
# Stopwords are high-frequency words (the, and, is, etc.) that carry little semantic meaning
stop_words = set(stopwords.words("english"))

def tokenize(text):
    """
    Advanced text tokenization with linguistic preprocessing.
    
    Performs comprehensive text normalization including:
    1. Punctuation removal for cleaner token extraction
    2. Lowercase conversion for case-insensitive matching
    3. Stopword filtering to remove common, non-discriminative terms
    4. Lemmatization to reduce words to their root forms (e.g., 'running' -> 'run')
    5. Length filtering to exclude very short tokens that may not be meaningful
    
    Args:
        text (str): Raw input text to be tokenized
        
    Returns:
        list[str]: List of processed tokens (lemmatized, filtered terms)
    """
    # Remove punctuation marks that don't contribute to semantic meaning
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Process text through spaCy's NLP pipeline for advanced tokenization
    doc = nlp(text.lower())
    tokens = []
    
    # Filter tokens based on multiple linguistic and semantic criteria
    for token in doc:
        if (token.is_alpha and                    # Only alphabetic tokens (no numbers/symbols)
            token.text not in stop_words and      # Exclude common stopwords
            len(token.text) > 2 and               # Filter out very short tokens
            not token.is_space):                  # Exclude whitespace tokens
            tokens.append(token.lemma_)           # Use lemmatized form for better matching
    return tokens

def preprocess_query(query):
    """
    Lightweight query preprocessing for fast query processing.
    
    This function provides a simplified tokenization pipeline specifically
    optimized for query terms. Uses spaCy's built-in stopword detection
    for consistent preprocessing with the main tokenization pipeline.
    
    Args:
        query (str): User's search query
        
    Returns:
        list[str]: List of lemmatized query terms without stopwords
    """
    doc = nlp(query.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return tokens

# Soundex Algorithm Implementation
def soundex(word):
    """
    Implements the classic Soundex phonetic algorithm for query expansion.
    
    Soundex converts words to a 4-character code based on phonetic similarity,
    enabling retrieval of documents containing phonetically similar terms.
    This is particularly useful for handling spelling variations and typos.
    
    Algorithm steps:
    1. Keep the first letter as-is
    2. Map consonants to numeric codes based on phonetic groups
    3. Remove consecutive duplicates and vowels/h/w/y
    4. Pad with zeros to ensure 4-character output
    
    Args:
        word (str): Input word to encode
        
    Returns:
        str: 4-character Soundex code (e.g., "smith" -> "S530")
        
    Example:
        soundex("smith") -> "S530"
        soundex("smyth") -> "S530"  (phonetically similar)
    """
    # Phonetic mapping table - groups consonants by similar sounds
    codes = {
        "B": "1","F": "1","P": "1","V": "1",        # Labials
        "C": "2","G": "2","J": "2","K": "2","Q": "2","S": "2","X": "2","Z": "2",  # Sibilants/Gutturals
        "D": "3","T": "3",                          # Dentals
        "L": "4",                                   # Liquid
        "M": "5","N": "5",                          # Nasals
        "R": "6"                                    # Liquid
    }
    
    # Handle edge case of empty input
    if not word:
        return "0000"
    
    # Convert to uppercase for consistent processing
    word = word.upper()
    first = word[0]  # Preserve first letter
    out = first
    prev = codes.get(first, "")  # Track previous code to avoid duplicates
    
    # Process remaining characters
    for ch in word[1:]:
        code = codes.get(ch, "")
        # Skip if same as previous code or if vowel/h/w/y (empty code)
        if code == prev or code == "":
            prev = code
            continue
        out += code
        prev = code
        # Stop when we have 4 characters
        if len(out) == 4:
            break
    
    # Pad with zeros if needed and ensure exactly 4 characters
    return out.ljust(4, "0")[:4]

# Vector Space Model Indexer
class VSMIndexer:
    """
    Builds and maintains an inverted index for the Vector Space Model.
    
    This class implements the document indexing component of VSM with:
    - Inverted index construction for efficient term lookup
    - TF-IDF weight calculation using lnc scheme (log-normalized TF, no IDF, cosine normalization)
    - Document vector normalization for cosine similarity
    - Soundex indexing for phonetic query expansion
    
    The indexer uses the 'lnc' weighting scheme for documents:
    - l: log-normalized term frequency (1 + log(tf))
    - n: no IDF weighting for documents  
    - c: cosine normalization of document vectors
    """
    
    def __init__(self):
        """Initialize indexer data structures."""
        self.inverted = {}                          # Main inverted index: {term: {df, postings}}
        self.N = 0                                  # Total number of documents in collection
        self.doc_norms = {}                         # Document vector norms for cosine normalization
        self.doc_term_norm_weight = defaultdict(dict)  # Normalized document term weights
        self.soundex_map = defaultdict(set)         # Soundex code to terms mapping
    
    def index_documents(self, docs):
        """
        Build the complete inverted index from document collection.
        
        Process flow:
        1. Calculate raw term frequencies for all documents
        2. Build inverted index with document frequency statistics
        3. Compute log-normalized TF weights for document terms
        4. Normalize document vectors using cosine normalization
        5. Create Soundex mappings for phonetic matching
        
        Args:
            docs (dict): Document collection {doc_id: content}
        """
        self.N = len(docs)
        # Temporary storage for term frequencies across all documents
        term_doc_tf = defaultdict(lambda: defaultdict(int))
        
        # Phase 1: Extract and count terms from all documents
        for docID, text in docs.items():
            tokens = tokenize(text)  # Apply full text preprocessing pipeline
            counts = Counter(tokens)  # Count term occurrences
            for term, tf in counts.items():
                term_doc_tf[term][docID] = tf
        
        # Phase 2: Build inverted index with document frequency statistics
        for term, doc_tf_map in term_doc_tf.items():
            # Sort postings by document ID for consistent ordering
            postings = sorted(doc_tf_map.items(), key=lambda x: x[0])
            df = len(postings)  # Document frequency (number of docs containing term)
            
            # Store inverted index entry
            self.inverted[term] = {"df": df, "postings": postings}
            
            # Build Soundex mapping for phonetic query expansion
            self.soundex_map[soundex(term)].add(term)
        
        # Phase 3: Calculate log-normalized TF weights and accumulate for normalization
        for term, info in self.inverted.items():
            for docID, tf in info["postings"]:
                # Apply log normalization: 1 + log10(tf) 
                # This reduces impact of high-frequency terms within documents
                tf_w = 1.0 + math.log10(tf)
                
                # Accumulate squared weights for L2 norm calculation
                self.doc_norms.setdefault(docID, 0.0)
                self.doc_norms[docID] += tf_w * tf_w
                
                # Store raw log-normalized weight (before normalization)
                self.doc_term_norm_weight[docID][term] = tf_w
        
        # Phase 4: Apply cosine normalization to all document vectors
        for docID, sqsum in self.doc_norms.items():
            # Calculate L2 norm (Euclidean length) of document vector
            norm = math.sqrt(sqsum) if sqsum > 0 else 1.0
            self.doc_norms[docID] = norm
            
            # Normalize all term weights by document vector length
            for term, raw_w in self.doc_term_norm_weight[docID].items():
                self.doc_term_norm_weight[docID][term] = raw_w / norm
    
    def get_postings(self, term):
        """
        Retrieve postings list for a given term.
        
        Args:
            term (str): Query term
            
        Returns:
            dict or None: Postings info {df, postings} or None if term not found
        """
        return self.inverted.get(term, None)
    
    def soundex_terms(self, term):
        """
        Find phonetically similar terms using Soundex algorithm.
        
        Args:
            term (str): Original term
            
        Returns:
            set: Set of terms with same Soundex code (phonetically similar)
        """
        return self.soundex_map.get(soundex(term), set())

# Vector Space Model Searcher (lnc.ltc weighting scheme)
class VSMSearcher:
    """
    Implements query processing and document ranking for Vector Space Model.
    
    This class handles the query-side processing using the 'ltc' weighting scheme:
    - l: log-normalized term frequency for query terms
    - t: IDF (inverse document frequency) weighting for discriminative power
    - c: cosine normalization of query vector
    
    The searcher computes cosine similarity between normalized query and document vectors
    to rank documents by relevance.
    """
    
    def __init__(self, indexer):
        """
        Initialize searcher with a pre-built index.
        
        Args:
            indexer (VSMIndexer): Pre-built inverted index with document weights
        """
        self.idx = indexer                          # Reference to the inverted index
        self.N = indexer.N                          # Total number of documents in collection
    
    def _query_vector(self, tokens):
        """
        Compute TF-IDF weights for query terms using 'ltc' scheme.
        
        For each unique query term, calculates:
        1. Log-normalized term frequency: 1 + log10(tf_query)
        2. Inverse document frequency: log10(N / df)
        3. Combined weight: tf_weight * idf_weight
        
        Args:
            tokens (list): Preprocessed query tokens
            
        Returns:
            dict: Mapping of {term: weight} for query terms, None for OOV terms
        """
        # Count term frequencies in query
        q_counts = Counter(tokens)
        q_weights = {}
        
        for term, tf_q in q_counts.items():
            # Apply log-normalization to query term frequency
            tf_q_w = 1.0 + math.log10(tf_q)
            
            # Retrieve term's document frequency from index
            p = self.idx.get_postings(term)
            if p is None:
                # Term not found in collection (Out-of-Vocabulary)
                q_weights[term] = None
            else:
                # Calculate IDF: log(N/df) - higher for rare terms
                df = p["df"]
                idf = math.log10(self.N / df) if df > 0 else 0
                q_weights[term] = tf_q_w * idf
        return q_weights
    
    def search(self, query, top_k=10, soundex_fallback=True):
        """
        Execute search query against document collection.
        
        Search process:
        1. Tokenize and preprocess query
        2. Calculate TF-IDF weights for query terms
        3. Handle out-of-vocabulary terms with Soundex expansion (optional)
        4. Normalize query vector using cosine normalization
        5. Compute cosine similarity with all relevant documents
        6. Rank and return top-k results
        
        Args:
            query (str): Raw search query from user
            top_k (int): Maximum number of results to return
            soundex_fallback (bool): Enable phonetic matching for OOV terms
            
        Returns:
            list: Ranked list of (document_id, similarity_score) tuples
        """
        # Step 1: Tokenize query using same preprocessing as documents
        tokens = tokenize(query)
        
        # Step 2: Calculate initial TF-IDF weights for query terms
        q_raw = self._query_vector(tokens)
        
        # Step 3: Collect terms that exist in the index
        q_term_weights = {}
        for term, w in q_raw.items():
            if w is not None:  # Term found in collection
                q_term_weights[term] = w
        
        # Step 4: Handle out-of-vocabulary terms with Soundex expansion
        if soundex_fallback:
            for term, w in q_raw.items():
                if w is None:  # Term not found in collection
                    # Calculate TF weight for the original query term
                    tf_q = Counter(tokens)[term]
                    tf_q_w = 1.0 + math.log10(tf_q)
                    
                    # Find phonetically similar terms using Soundex
                    for cand in self.idx.soundex_terms(term):
                        p = self.idx.get_postings(cand)
                        if p:
                            # Use IDF of the phonetically similar term
                            df_cand = p["df"]
                            idf_cand = math.log10(self.N / df_cand) if df_cand > 0 else 0
                            # Add contribution from expanded term (accumulate if multiple matches)
                            q_term_weights[cand] = q_term_weights.get(cand, 0) + tf_q_w * idf_cand
        
        # Step 5: Early termination if no valid query terms
        if not q_term_weights:
            return []
        
        # Step 6: Apply cosine normalization to query vector
        q_norm = math.sqrt(sum(w*w for w in q_term_weights.values())) or 1.0
        for t in q_term_weights:
            q_term_weights[t] /= q_norm
        
        # Step 7: Calculate cosine similarity scores for all relevant documents
        scores = defaultdict(float)
        for term, q_w in q_term_weights.items():
            # Retrieve postings list for current query term
            posting = self.idx.get_postings(term)
            if not posting:
                continue
            
            # Accumulate similarity scores for documents containing this term
            for docID, _ in posting["postings"]:
                # Get normalized document weight for this term
                d_w = self.idx.doc_term_norm_weight[docID].get(term, 0.0)
                # Add dot product contribution: q_weight * doc_weight
                scores[docID] += q_w * d_w
        
        # Step 8: Filter, sort, and return top-k results
        results = [(docID, s) for docID, s in scores.items() if s > 0.]
        # Sort by score (descending), then by document ID (ascending) for tie-breaking
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

# Document Corpus Loading Utilities
def load_corpus_from_folder(folder_path):
    """
    Load all text documents from a specified directory.
    
    Recursively searches through the directory structure to find all .txt files
    and loads their content into memory for indexing. Handles encoding issues
    gracefully by ignoring problematic characters.
    
    Args:
        folder_path (str): Path to the directory containing document corpus
        
    Returns:
        dict: Mapping of {filename: document_content} for all .txt files found
        
    Note:
        Uses UTF-8 encoding with error handling for robust file reading
    """
    docs = {}
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    docs[filename] = f.read()
    return docs

if __name__ == "__main__":
    # Configuration: Set corpus directory path
    folder = "./Corpus"  # Unzip the corpus in the same directory as this file
    
    # Phase 1: Document Collection Loading
    print("Loading document corpus...")
    docs = load_corpus_from_folder(folder)
    
    print(f"Loaded {len(docs)} documents.")
    
    # Early exit if no documents found
    if len(docs) == 0:
        print("Error: No documents found. Please check the corpus folder path.")
        exit()
    
    # Phase 2: Index Construction
    print("Building inverted index...")
    idx = VSMIndexer()
    idx.index_documents(docs)
    print("Index construction completed.")
    
    # Phase 3: Search System Initialization
    searcher = VSMSearcher(idx)
    
    # Phase 4: Interactive Search Loop
    print("\n" + "="*60)
    print("Vector Space Model Search Engine")
    print("="*60)
    print("Features:")
    print("• TF-IDF weighting with cosine similarity")
    print("• Soundex-based query expansion for typos")
    print("• Advanced text preprocessing with lemmatization")
    print("="*60)
    print("Enter your search queries (type 'exit' to quit)")
    
    while True:
        try:
            # Get user query
            q = input("\nEnter query (or 'exit'): ")
            
            # Check for exit condition
            if q.lower() == "exit":
                print("Thank you for using the VSM Search Engine!")
                break
            
            # Perform search and display results
            results = searcher.search(q)
            if not results:
                print("No results found.")
                print("Suggestions:")
                print("• Try different keywords")
                print("• Check spelling (Soundex will help with minor typos)")
                print("• Use more general terms")
            else:
                print(f"\nFound {len(results)} relevant documents:")
                print("-" * 50)
                for i, (docID, score) in enumerate(results, 1):
                    print(f"{i:2d}. {docID:<25} (Relevance: {score:.8f})")
        
        except KeyboardInterrupt:
            print("\n\nSearch session terminated by user.")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            print("Please try again with a different query.")
