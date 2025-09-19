# Vector_Space_Model_Rank_Retrieval_Model

Vector Space Model Search Engine
This project implements a Vector Space Model (VSM)–based search engine in Python. 
It indexes a corpus of text files, processes queries and retrieves ranked results using the lnc.ltc weighting (logarithmic term frequency and inverse document frequency).

Features:
Tokenization and Lemmatization using spaCy
Stopword removal using NLTK
Soundex-based fallback for handling spelling variations
Inverted index construction with term frequencies and document frequencies
TF–IDF weighting and cosine similarity for ranking

Simple command-line interface to load a corpus and query it

Project Structure
project/
├── main.py                 # Main script (indexer + searcher)
├── README.md
└── Corpus/                 # Folder containing .txt files (your corpus)

Requirements
Python 3.7+
spaCy and English model (en_core_web_sm)
NLTK(for stopwords)

Install dependencies:
pip install spacy nltk
python -m spacy download en_core_web_sm

Download NLTK stopwords (run once):
import nltk
nltk.download('stopwords')

Usage
Prepare corpus:
Place your .txt files inside the Corpus/ folder.

Run the script:
python main.py

Interact via CLI:
Enter a query to retrieve ranked documents.
Type exit to quit.
Example:
Loaded 50 documents.
Enter query (or 'exit'): information retrieval
Doc doc1.txt (score=0.53211428)
Doc doc5.txt (score=0.47629402)

How It Works
Documents are tokenized, lemmatized, and filtered for stopwords.
An inverted index is built with TF counts.
Document vectors are normalized.
Query vectors are computed with TF–IDF weighting.
Cosine similarity ranks documents by relevance.
If a query term is unseen, similar terms are tried using Soundex codes.

License
This project is open-source under the MIT License.
