"""
Configuration settings for the FAQ Recommendation System.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"
FAISS_INDEX_DIR = MODELS_DIR / "faiss_index"

# Logs
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FAISS_INDEX_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Model Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
DEFAULT_RETRIEVAL_MODE = os.getenv("DEFAULT_RETRIEVAL_MODE", "sbert")

# Sentence-BERT Model
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and accurate

# File names
FAQ_CORPUS_FILE = "faq_corpus.csv"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE = "tfidf_matrix.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.npy"

