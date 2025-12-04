"""
Main script to run the complete FAQ pipeline:
1. Data ingestion and unification
2. Build TF-IDF index
3. Build SBERT + FAISS index

Run this script first to prepare all data and models.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import run_pipeline
from src.tfidf_retriever import TFIDFRetriever
from src.sbert_retriever import SBERTRetriever


def main():
    print("\n" + "="*70)
    print("   FAQ RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Run data pipeline
    print("\n" + "📊 STEP 1: DATA PIPELINE")
    print("-"*70)
    corpus = run_pipeline()
    
    if corpus.empty:
        print("\n❌ Data pipeline failed. Please check your data files.")
        return
    
    # Step 2: Build TF-IDF index
    print("\n" + "📊 STEP 2: TF-IDF INDEX")
    print("-"*70)
    tfidf = TFIDFRetriever()
    tfidf.build_index(corpus)
    tfidf.save()
    
    # Step 3: Build SBERT + FAISS index
    print("\n" + "📊 STEP 3: SBERT + FAISS INDEX")
    print("-"*70)
    sbert = SBERTRetriever()
    sbert.build_index(corpus)
    sbert.save()
    
    # Summary
    print("\n" + "="*70)
    print("   ✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"""
    📈 Summary:
    - Total FAQs: {len(corpus)}
    - TF-IDF features: {tfidf.tfidf_matrix.shape[1]}
    - SBERT embedding dim: {sbert.embedding_dim}
    
    🚀 Next Steps:
    1. Start the API:
       python -m uvicorn api.main:app --reload
    
    2. Start the Streamlit UI:
       streamlit run streamlit_app.py
    
    3. Open your browser:
       - API docs: http://localhost:8000/docs
       - Streamlit UI: http://localhost:8501
    """)


if __name__ == "__main__":
    main()

