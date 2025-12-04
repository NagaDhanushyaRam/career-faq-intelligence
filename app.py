"""
FAQ Recommendation System - Streamlit Cloud Deployment Version
This standalone app works without the FastAPI backend.
Models are built/cached on first run.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import re
import string
import nltk

# Download NLTK data
@st.cache_resource
def download_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

download_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import sentence transformers and faiss
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data/processed/faq_corpus.csv")
SAMPLE_DATA_URL = "https://raw.githubusercontent.com/your-repo/main/data/processed/faq_corpus.csv"

# ============================================================
# TEXT PREPROCESSING
# ============================================================
try:
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = set()

LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str):
        text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and filter
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords and short tokens
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens 
              if t not in STOP_WORDS and len(t) > 2]
    
    return ' '.join(tokens)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_corpus():
    """Load the FAQ corpus."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    else:
        st.error("❌ FAQ corpus not found. Please ensure data/processed/faq_corpus.csv exists.")
        return None

# ============================================================
# TF-IDF MODEL
# ============================================================
@st.cache_resource
def build_tfidf_model(_corpus_df):
    """Build TF-IDF model with caching."""
    with st.spinner("🔄 Building TF-IDF index (first time only)..."):
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        processed = [preprocess_text(q) for q in _corpus_df['question'].tolist()]
        tfidf_matrix = vectorizer.fit_transform(processed)
        
    return vectorizer, tfidf_matrix

def search_tfidf(query: str, corpus_df, vectorizer, tfidf_matrix, top_k: int = 5):
    """Search using TF-IDF."""
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'question': corpus_df.iloc[idx]['question'],
            'answer': corpus_df.iloc[idx]['answer'],
            'score': float(similarities[idx]),
            'source': corpus_df.iloc[idx].get('source', 'unknown')
        })
    return results

# ============================================================
# SBERT MODEL
# ============================================================
@st.cache_resource
def build_sbert_model(_corpus_df):
    """Build SBERT + FAISS model with caching."""
    if not SBERT_AVAILABLE:
        return None, None
    
    with st.spinner("🔄 Building SBERT index (first time only, may take a few minutes)..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        questions = _corpus_df['question'].tolist()
        embeddings = model.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
    return model, index

def search_sbert(query: str, corpus_df, model, index, top_k: int = 5):
    """Search using SBERT."""
    if model is None or index is None:
        return []
    
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                'question': corpus_df.iloc[idx]['question'],
                'answer': corpus_df.iloc[idx]['answer'],
                'score': float(score),
                'source': corpus_df.iloc[idx].get('source', 'unknown')
            })
    return results

# ============================================================
# CUSTOM CSS
# ============================================================
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .sub-title {
            font-size: 1rem;
            color: #a0aec0;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .result-card {
            background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .question-box {
            background: rgba(102, 126, 234, 0.15);
            border-left: 3px solid #667eea;
            padding: 0.8rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        
        .answer-box {
            background: rgba(72, 187, 120, 0.15);
            border-left: 3px solid #48bb78;
            padding: 0.8rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        
        .score-high { color: #48bb78; font-weight: 600; }
        .score-medium { color: #ecc94b; font-weight: 600; }
        .score-low { color: #fc8181; font-weight: 600; }
        
        .source-tag {
            display: inline-block;
            background: rgba(118, 75, 162, 0.3);
            color: #b794f6;
            padding: 0.2rem 0.6rem;
            border-radius: 10px;
            font-size: 0.75rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(
        page_title="Career FAQ Search",
        page_icon="🎯",
        layout="wide"
    )
    
    inject_css()
    
    # Header
    st.markdown('<h1 class="main-title">🎯 Career FAQ Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-powered answers to your career questions</p>', unsafe_allow_html=True)
    
    # Load data
    corpus_df = load_corpus()
    
    if corpus_df is None:
        st.stop()
    
    # Build models
    vectorizer, tfidf_matrix = build_tfidf_model(corpus_df)
    
    sbert_model, faiss_index = None, None
    if SBERT_AVAILABLE:
        sbert_model, faiss_index = build_sbert_model(corpus_df)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        if SBERT_AVAILABLE and sbert_model is not None:
            mode = st.radio(
                "Search Mode",
                ["sbert", "tfidf"],
                format_func=lambda x: "🧠 Semantic (SBERT)" if x == "sbert" else "📝 Keyword (TF-IDF)"
            )
        else:
            mode = "tfidf"
            st.info("ℹ️ Using TF-IDF mode (SBERT not available)")
        
        top_k = st.slider("Results", 1, 10, 5)
        
        st.markdown("---")
        st.header("📊 Stats")
        st.metric("Total FAQs", f"{len(corpus_df):,}")
        
        if 'source' in corpus_df.columns:
            st.markdown("**Sources:**")
            for src, cnt in corpus_df['source'].value_counts().items():
                st.text(f"• {src}: {cnt:,}")
    
    # Main content
    st.markdown("#### 💡 Try an example:")
    col1, col2, col3, col4 = st.columns(4)
    
    examples = [
        ("📝", "How to write a resume?"),
        ("🎤", "Interview preparation tips"),
        ("💰", "Salary negotiation"),
        ("🚀", "Career change advice")
    ]
    
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    for i, (icon, text) in enumerate(examples):
        with [col1, col2, col3, col4][i]:
            if st.button(f"{icon} {text}", key=f"ex_{i}", use_container_width=True):
                st.session_state.query = text
                st.rerun()
    
    # Search input
    query = st.text_input(
        "🔍 Ask your career question:",
        value=st.session_state.query,
        placeholder="e.g., How do I prepare for a technical interview?"
    )
    
    if query != st.session_state.query:
        st.session_state.query = query
    
    st.markdown("---")
    
    # Search
    if query:
        start = time.time()
        
        if mode == "sbert" and sbert_model is not None:
            results = search_sbert(query, corpus_df, sbert_model, faiss_index, top_k)
        else:
            results = search_tfidf(query, corpus_df, vectorizer, tfidf_matrix, top_k)
        
        elapsed = (time.time() - start) * 1000
        
        if results:
            st.markdown(f"### Found {len(results)} results ({elapsed:.0f}ms)")
            
            for i, r in enumerate(results, 1):
                score = r['score']
                if score >= 0.5:
                    score_class, icon = "score-high", "🟢"
                elif score >= 0.25:
                    score_class, icon = "score-medium", "🟡"
                else:
                    score_class, icon = "score-low", "🔴"
                
                st.markdown(f"**#{i}** {icon} Score: `{score:.3f}` | <span class='source-tag'>{r['source']}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='question-box'><b>❓ Question:</b><br>{r['question']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer-box'><b>✅ Answer:</b><br>{r['answer']}</div>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("No results found. Try rephrasing.")
    else:
        st.markdown("""
        ### 👋 Welcome!
        
        Ask any career question to get instant answers from our FAQ database.
        
        **Topics covered:** Resume writing • Interview prep • Salary negotiation • Career growth • Job applications
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("<center><small>Career FAQ Intelligence • COSC 757 Data Mining Project</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

