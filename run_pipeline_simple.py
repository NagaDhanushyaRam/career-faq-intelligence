"""
Simplified Pipeline - Uses only Entry Level Career QA and CareerVillage datasets.
Run this if the full pipeline has issues with the large HR Interview JSON file.
"""
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR, FAQ_CORPUS_FILE
from src.data_pipeline import (
    load_entry_level_career_qa, 
    load_careervillage_data,
    RAW_DATA_DIR
)
from src.tfidf_retriever import TFIDFRetriever
from src.sbert_retriever import SBERTRetriever


def main():
    print("\n" + "="*70)
    print("   FAQ RECOMMENDATION SYSTEM - SIMPLIFIED PIPELINE")
    print("   (Skipping HR Interview dataset)")
    print("="*70)
    
    # Step 1: Load only Entry Level and CareerVillage datasets
    print("\n📊 STEP 1: LOADING DATASETS")
    print("-"*70)
    
    entry_level_df = load_entry_level_career_qa(RAW_DATA_DIR)
    careervillage_df = load_careervillage_data(RAW_DATA_DIR)
    
    # Combine datasets
    dfs_to_combine = []
    
    if not entry_level_df.empty:
        if 'question' in entry_level_df.columns and 'answer' in entry_level_df.columns:
            entry_level_df = entry_level_df[['question', 'answer', 'source']]
            dfs_to_combine.append(entry_level_df)
            print(f"  ✓ Entry Level Career QA: {len(entry_level_df)} records")
    
    if not careervillage_df.empty:
        if 'question' in careervillage_df.columns and 'answer' in careervillage_df.columns:
            careervillage_df = careervillage_df[['question', 'answer', 'source']]
            dfs_to_combine.append(careervillage_df)
            print(f"  ✓ CareerVillage: {len(careervillage_df)} records")
    
    if not dfs_to_combine:
        print("\n❌ No data loaded! Check your dataset files.")
        return
    
    # Combine and clean
    print("\n📊 STEP 2: COMBINING & CLEANING")
    print("-"*70)
    
    corpus = pd.concat(dfs_to_combine, ignore_index=True)
    
    # Clean data
    corpus = corpus.dropna(subset=['question', 'answer'])
    corpus['question'] = corpus['question'].astype(str).str.strip()
    corpus['answer'] = corpus['answer'].astype(str).str.strip()
    corpus = corpus[corpus['question'].str.len() > 10]  # Filter very short questions
    corpus = corpus[corpus['answer'].str.len() > 10]    # Filter very short answers
    
    # Remove duplicates
    initial_count = len(corpus)
    corpus = corpus.drop_duplicates(subset=['question'], keep='first')
    print(f"  ✓ Removed {initial_count - len(corpus)} duplicates")
    
    # Add IDs
    corpus = corpus.reset_index(drop=True)
    corpus['faq_id'] = range(len(corpus))
    corpus = corpus[['faq_id', 'question', 'answer', 'source']]
    
    # Save corpus
    output_path = PROCESSED_DATA_DIR / FAQ_CORPUS_FILE
    corpus.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(corpus)} FAQs to {output_path}")
    
    # Step 3: Build TF-IDF index
    print("\n📊 STEP 3: BUILDING TF-IDF INDEX")
    print("-"*70)
    
    tfidf = TFIDFRetriever()
    tfidf.build_index(corpus)
    tfidf.save()
    
    # Step 4: Build SBERT index
    print("\n📊 STEP 4: BUILDING SBERT + FAISS INDEX")
    print("-"*70)
    
    sbert = SBERTRetriever()
    sbert.build_index(corpus)
    sbert.save()
    
    # Summary
    print("\n" + "="*70)
    print("   ✅ SIMPLIFIED PIPELINE COMPLETE!")
    print("="*70)
    print(f"""
    📈 Summary:
    - Total FAQs: {len(corpus)}
    - Sources: {corpus['source'].value_counts().to_dict()}
    - TF-IDF features: {tfidf.tfidf_matrix.shape[1]}
    - SBERT embedding dim: {sbert.embedding_dim}
    
    🚀 Next Steps:
    1. Start the API:
       python -m uvicorn api.main:app --reload --port 8000
    
    2. Start the Streamlit UI (in another terminal):
       streamlit run streamlit_app.py
    
    3. Open your browser:
       - API docs: http://localhost:8000/docs
       - Streamlit UI: http://localhost:8501
    """)
    
    # Show sample data
    print("\n📋 Sample FAQs:")
    print("-"*70)
    for i, row in corpus.head(3).iterrows():
        print(f"\nQ: {row['question'][:100]}...")
        print(f"A: {row['answer'][:150]}...")


if __name__ == "__main__":
    main()

