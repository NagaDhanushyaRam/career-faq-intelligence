"""
TF-IDF Retrieval Module
Implements the TF-IDF baseline retriever as shown in Listing 3 of the Midterm Report.

Features:
- build_index(): Build TF-IDF vectors from FAQ corpus
- search(): Find top-k similar FAQs using cosine similarity
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS_DIR, 
    PROCESSED_DATA_DIR, 
    FAQ_CORPUS_FILE,
    TFIDF_VECTORIZER_FILE,
    TFIDF_MATRIX_FILE,
    DEFAULT_TOP_K
)
from src.preprocessing import preprocess_for_tfidf


class TFIDFRetriever:
    """
    TF-IDF based FAQ retriever.
    
    Uses cosine similarity between TF-IDF vectors to find
    the most similar FAQs to a user query.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to extract (1,2) means unigrams and bigrams
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency (fraction) for terms
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.tfidf_matrix = None
        self.corpus_df = None
        self.is_fitted = False
    
    def build_index(self, corpus_df: Optional[pd.DataFrame] = None) -> None:
        """
        Build the TF-IDF index from the FAQ corpus.
        
        Args:
            corpus_df: DataFrame with 'question' and 'answer' columns.
                      If None, loads from the default processed data path.
        """
        print("\n📊 Building TF-IDF Index...")
        
        # Load corpus if not provided
        if corpus_df is None:
            corpus_path = PROCESSED_DATA_DIR / FAQ_CORPUS_FILE
            if not corpus_path.exists():
                raise FileNotFoundError(
                    f"FAQ corpus not found at {corpus_path}. "
                    "Please run the data pipeline first."
                )
            corpus_df = pd.read_csv(corpus_path)
        
        self.corpus_df = corpus_df.reset_index(drop=True)
        
        # Preprocess questions
        print("  🔄 Preprocessing questions...")
        processed_questions = [
            preprocess_for_tfidf(q) for q in self.corpus_df['question'].tolist()
        ]
        
        # Fit and transform
        print("  🔄 Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
        
        self.is_fitted = True
        
        print(f"  ✅ Index built successfully!")
        print(f"     - Documents: {self.tfidf_matrix.shape[0]}")
        print(f"     - Features: {self.tfidf_matrix.shape[1]}")
        print(f"     - Matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4f}")
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar FAQs to the query.
        
        Args:
            query: User's search query
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries containing:
            - faq_id: Unique identifier
            - question: Original question
            - answer: Answer text
            - score: Cosine similarity score
            - source: Data source
        """
        if not self.is_fitted:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Preprocess query
        processed_query = preprocess_for_tfidf(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            # Skip if below threshold
            if score < threshold:
                continue
            
            row = self.corpus_df.iloc[idx]
            results.append({
                'faq_id': int(row.get('faq_id', idx)),
                'question': row['question'],
                'answer': row['answer'],
                'score': round(score, 4),
                'source': row.get('source', 'unknown'),
                'retrieval_method': 'tfidf'
            })
        
        return results
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_index() first.")
        
        if path is None:
            path = MODELS_DIR
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = path / TFIDF_VECTORIZER_FILE
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save TF-IDF matrix
        matrix_path = path / TFIDF_MATRIX_FILE
        with open(matrix_path, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        # Save corpus
        corpus_path = path / "tfidf_corpus.pkl"
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.corpus_df, f)
        
        print(f"✅ Model saved to {path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load a fitted model from disk."""
        if path is None:
            path = MODELS_DIR
        
        path = Path(path)
        
        # Load vectorizer
        vectorizer_path = path / TFIDF_VECTORIZER_FILE
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load TF-IDF matrix
        matrix_path = path / TFIDF_MATRIX_FILE
        with open(matrix_path, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        # Load corpus
        corpus_path = path / "tfidf_corpus.pkl"
        with open(corpus_path, 'rb') as f:
            self.corpus_df = pickle.load(f)
        
        self.is_fitted = True
        print(f"✅ Model loaded from {path}")
    
    def get_feature_names(self) -> List[str]:
        """Get the vocabulary (feature names) from the vectorizer."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.vectorizer.get_feature_names_out().tolist()


# Create singleton instance for easy import
_retriever_instance = None


def get_tfidf_retriever() -> TFIDFRetriever:
    """Get or create the TF-IDF retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = TFIDFRetriever()
    return _retriever_instance


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("TF-IDF RETRIEVER DEMO")
    print("="*60)
    
    retriever = TFIDFRetriever()
    
    try:
        # Try to build index
        retriever.build_index()
        
        # Test search
        test_queries = [
            "How do I write a good resume?",
            "What are common interview questions?",
            "How to negotiate salary?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-"*40)
            
            results = retriever.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [Score: {result['score']:.3f}]")
                print(f"   Q: {result['question'][:80]}...")
                print(f"   A: {result['answer'][:100]}...")
        
        # Save model
        retriever.save()
        
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
        print("Run the data pipeline first: python src/data_pipeline.py")

