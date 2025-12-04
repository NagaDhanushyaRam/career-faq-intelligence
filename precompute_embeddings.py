"""
Pre-compute SBERT embeddings locally.
Run this once, then upload the embeddings file to GitHub.
This makes Streamlit Cloud deployment much faster.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

print("=" * 60)
print("SBERT Embeddings Pre-computation")
print("=" * 60)

# Load corpus
corpus_path = Path("data/processed/faq_corpus.csv")
print(f"\n📂 Loading corpus from {corpus_path}...")
df = pd.read_csv(corpus_path)
print(f"✅ Loaded {len(df):,} FAQ entries")

# Load SBERT model
print("\n🔄 Loading SBERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Compute embeddings
print(f"\n🧠 Computing embeddings for {len(df):,} questions...")
print("   (This may take a few minutes...)")
start = time.time()

questions = df['question'].tolist()
embeddings = model.encode(
    questions,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=64
)

elapsed = time.time() - start
print(f"✅ Embeddings computed in {elapsed:.1f} seconds")
print(f"   Shape: {embeddings.shape}")

# Save embeddings
output_path = Path("data/processed/sbert_embeddings.npy")
print(f"\n💾 Saving embeddings to {output_path}...")
np.save(output_path, embeddings)

# Check file size
file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"✅ Saved! File size: {file_size_mb:.1f} MB")

if file_size_mb > 100:
    print("\n⚠️  WARNING: File is larger than 100MB!")
    print("   GitHub has a 100MB file limit.")
    print("   Consider using fewer FAQ entries or Git LFS.")
else:
    print("\n✅ File is under 100MB - safe to push to GitHub!")

print("\n" + "=" * 60)
print("Done! Now run:")
print("  git add data/processed/sbert_embeddings.npy")
print("  git commit -m 'Add pre-computed SBERT embeddings'")
print("  git push")
print("=" * 60)

