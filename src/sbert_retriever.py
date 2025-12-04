"""
Sentence-BERT + FAISS Retrieval Module
Implements semantic search using Sentence-BERT embeddings and FAISS indexing.

Features:
- Semantic understanding of queries
- Fast approximate nearest neighbor search with FAISS
- Better handling of paraphrases and synonyms
"""
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    FAQ_CORPUS_FILE,
    FAISS_INDEX_DIR,
    FAISS_INDEX_FILE,
    EMBEDDINGS_FILE,
    SBERT_MODEL_NAME,
    DEFAULT_TOP_K
)
from src.preprocessing import preprocess_for_sbert


class SBERTRetriever:
    """
    Sentence-BERT based FAQ retriever with FAISS indexing.
    
    Uses dense embeddings from Sentence-BERT for semantic similarity
    and FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        model_name: str = SBERT_MODEL_NAME,
        use_gpu: bool = False
    ):
        """
        Initialize the SBERT retriever.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
            use_gpu: Whether to use GPU for encoding (requires CUDA)
        """
        self.model_name = model_name
        self.device = 'cuda' if use_gpu else 'cpu'
        
        print(f"📦 Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.embeddings = None
        self.faiss_index = None
        self.corpus_df = None
        self.is_fitted = False
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def build_index(
        self,
        corpus_df: Optional[pd.DataFrame] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """
        Build the FAISS index from the FAQ corpus.
        
        Args:
            corpus_df: DataFrame with 'question' and 'answer' columns
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        """
        print("\n📊 Building SBERT + FAISS Index...")
        
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
            preprocess_for_sbert(q) for q in self.corpus_df['question'].tolist()
        ]
        
        # Encode questions
        print("  🔄 Encoding questions with SBERT...")
        self.embeddings = self.model.encode(
            processed_questions,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        # Build FAISS index
        print("  🔄 Building FAISS index...")
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add vectors to index
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        self.is_fitted = True
        
        print(f"  ✅ Index built successfully!")
        print(f"     - Documents: {len(self.embeddings)}")
        print(f"     - Embedding dimension: {self.embedding_dim}")
        print(f"     - FAISS index size: {self.faiss_index.ntotal}")
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for the most semantically similar FAQs to the query.
        
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
        processed_query = preprocess_for_sbert(query)
        
        # Encode query
        query_embedding = self.model.encode(
            [processed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices
            if idx < 0:
                continue
            
            # Skip if below threshold
            if score < threshold:
                continue
            
            row = self.corpus_df.iloc[idx]
            results.append({
                'faq_id': int(row.get('faq_id', idx)),
                'question': row['question'],
                'answer': row['answer'],
                'score': round(float(score), 4),
                'source': row.get('source', 'unknown'),
                'retrieval_method': 'sbert'
            })
        
        return results
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_index() first.")
        
        if path is None:
            path = FAISS_INDEX_DIR
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / FAISS_INDEX_FILE
        faiss.write_index(self.faiss_index, str(index_path))
        
        # Save embeddings
        embeddings_path = path / EMBEDDINGS_FILE
        np.save(embeddings_path, self.embeddings)
        
        # Save corpus
        corpus_path = path / "sbert_corpus.pkl"
        import pickle
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.corpus_df, f)
        
        print(f"✅ Model saved to {path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load a fitted model from disk."""
        if path is None:
            path = FAISS_INDEX_DIR
        
        path = Path(path)
        
        # Load FAISS index
        index_path = path / FAISS_INDEX_FILE
        self.faiss_index = faiss.read_index(str(index_path))
        
        # Load embeddings
        embeddings_path = path / EMBEDDINGS_FILE
        self.embeddings = np.load(embeddings_path)
        
        # Load corpus
        corpus_path = path / "sbert_corpus.pkl"
        import pickle
        with open(corpus_path, 'rb') as f:
            self.corpus_df = pickle.load(f)
        
        self.is_fitted = True
        print(f"✅ Model loaded from {path}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]


# Create singleton instance for easy import
_retriever_instance = None


def get_sbert_retriever() -> SBERTRetriever:
    """Get or create the SBERT retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SBERTRetriever()
    return _retriever_instance


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("SBERT + FAISS RETRIEVER DEMO")
    print("="*60)
    
    retriever = SBERTRetriever()
    
    try:
        # Try to build index
        retriever.build_index()
        
        # Test search
        test_queries = [
            "How do I write a good resume?",
            "What questions will they ask in an interview?",
            "How to ask for more money in a job offer?",
            "Tips for career change"
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

