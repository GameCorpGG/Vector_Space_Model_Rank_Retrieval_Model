"""
Streamlit Web Interface for Vector Space Model Search Engine
===========================================================

This module provides a web interface for the VSM search engine with deterministic behavior.
All query processing and result display operations are performed in sorted order to ensure
consistent results across multiple runs with the same query.

Features:
- Interactive search interface with real-time results
- Score visualization with matplotlib charts  
- Document snippets with query term highlighting
- Deterministic query processing and result display
"""

import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vsm_search import (
    VSMIndexer, VSMSearcher, load_corpus_from_folder, 
    tokenize, snippet_with_highlights, plot_scores, 
    enhanced_search_with_snippets
)

# Set page configuration
st.set_page_config(
    page_title="VSM Search Engine",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_search_engine():
    """Initialize and cache the VSM search engine"""
    with st.spinner("Loading corpus and building index..."):
        folder = "./Corpus"
        docs = load_corpus_from_folder(folder)
        
        if len(docs) == 0:
            st.error(f"No documents found in '{folder}'. Please check the corpus folder.")
            return None, None, 0
        
        indexer = VSMIndexer()
        indexer.index_documents(docs)
        
        searcher = VSMSearcher(indexer)
        
        return searcher, docs, len(docs)

def main():
    st.title("Vector Space Model Search Engine")
    st.markdown("---")
    
    searcher, docs, doc_count = initialize_search_engine()
    
    if searcher is None:
        st.stop()
    
    st.success(f"Loaded {doc_count} documents and built search index")
    
    st.subheader("Search Documents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., smartphone technology, social media platform, etc.",
            key="search_query"
        )
    with col2:
        show_visualization = st.checkbox("Show score chart", value=False)
    
    if st.button("Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching..."):
                results = searcher.search(query.strip(), top_k=10)
            
            if results:
                st.subheader(f"Search Results ({len(results)} documents found)")
                
                if show_visualization:
                    st.subheader("Relevance Score Visualization")
                    
                    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.3)))
                    docIDs = [d for d, _ in results]
                    scores = [s for _, s in results]
                    
                    bars = ax.barh(docIDs[::-1], scores[::-1], color='skyblue', alpha=0.7)
                    ax.set_xlabel("Relevance Score")
                    ax.set_ylabel("Documents") 
                    ax.set_title(f"Search Results for: '{query.strip()}'")
                    ax.grid(axis='x', alpha=0.3)
                    
                    for bar, score in zip(bars, scores[::-1]):
                        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'{score:.4f}', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown("---")
                
                query_tokens = tokenize(query.strip())
                
                for i, (doc_id, score) in enumerate(results, 1):
                    with st.expander(f"{i}. **{doc_id}** (Score: {score:.6f})", expanded=False):
                        if doc_id in docs:
                            snippet = snippet_with_highlights(docs[doc_id], query_tokens, length=300)
                            
                            clean_snippet = re.sub(r'\x1b\[[0-9;]*m', '', snippet)
                            
                            st.markdown("**Document Preview:**")
                            st.markdown(f"*{clean_snippet}*")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Document Length", f"{len(docs[doc_id])} chars")
                            with col2:
                                matches = [term for term in sorted(query_tokens) if term.lower() in docs[doc_id].lower()]
                                st.metric("Query Terms Found", len(matches))
                            with col3:
                                st.metric("Relevance Score", f"{score:.6f}")
                            
                            highlighted_snippet = clean_snippet
                            for term in sorted(query_tokens):  # Sort terms for consistent processing
                                highlighted_snippet = highlighted_snippet.replace(
                                    term, f"**:red[{term}]**"
                                ).replace(
                                    term.capitalize(), f"**:red[{term.capitalize()}]**"
                                ).replace(
                                    term.upper(), f"**:red[{term.upper()}]**"
                                )
                            
                            if highlighted_snippet != clean_snippet:
                                st.markdown("**🔍 Highlighted Preview:**")
                                st.markdown(highlighted_snippet)
                            
                            if st.button(f"View Full Document", key=f"full_{i}"):
                                st.text_area(
                                    "Full Document Content:",
                                    docs[doc_id],
                                    height=300,
                                    key=f"full_content_{i}"
                                )
            else:
                st.warning("No results found for your query.")
                st.info("""
                **Suggestions:**
                - Try different keywords
                - Check spelling (the system uses Soundex for typo tolerance)
                - Use more general terms
                - Try single words instead of phrases
                """)
        else:
            st.warning("Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Enhanced Features:**
        - TF-IDF weighting with cosine similarity
        - Soundex-based query expansion for typos
        - Advanced text preprocessing with lemmatization
        - Smart snippet generation with query highlighting
        - Interactive score visualization
        - Document metadata and statistics
        """
    )

if __name__ == "__main__":
    main()
