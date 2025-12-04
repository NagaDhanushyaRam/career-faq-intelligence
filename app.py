"""
FAQ Recommendation System - Streamlit Cloud Deployment Version
Full-featured app with Search, Analytics, Data Explorer, and About tabs.
Works standalone without FastAPI backend.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import re
import string
import json
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

# Try to import SBERT
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
EMBEDDINGS_PATH = Path("data/processed/sbert_embeddings.npy")
EVAL_PATH = Path("data/processed/evaluation_results.json")
VIZ_DIR = Path("logs/visualizations")

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
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
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
        st.error("❌ FAQ corpus not found.")
        return None

@st.cache_data
def load_evaluation_results():
    """Load evaluation results."""
    if EVAL_PATH.exists():
        with open(EVAL_PATH, 'r') as f:
            return json.load(f)
    return None

# ============================================================
# TF-IDF MODEL
# ============================================================
@st.cache_resource
def build_tfidf_model(_corpus_df):
    """Build TF-IDF model."""
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
    """Load SBERT model and pre-computed embeddings."""
    if not SBERT_AVAILABLE:
        return None, None
    
    if not EMBEDDINGS_PATH.exists():
        # Try to compute embeddings if file doesn't exist
        with st.spinner("🔄 Building SBERT index (first time only)..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            questions = _corpus_df['question'].tolist()
            embeddings = model.encode(
                questions,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
        return model, index
    
    with st.spinner("🔄 Loading SBERT model..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = np.load(EMBEDDINGS_PATH)
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
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
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
        
        .metric-card {
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #a0aec0;
            text-transform: uppercase;
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
        
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .custom-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            margin: 1.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# TAB: SEARCH
# ============================================================
def render_search_tab(corpus_df, vectorizer, tfidf_matrix, sbert_model, faiss_index, mode, top_k):
    """Render the search tab."""
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
    
    query = st.text_input(
        "🔍 Ask your career question:",
        value=st.session_state.query,
        placeholder="e.g., How do I prepare for a technical interview?"
    )
    
    if query != st.session_state.query:
        st.session_state.query = query
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if query:
        start = time.time()
        
        if mode == "sbert" and sbert_model is not None:
            results = search_sbert(query, corpus_df, sbert_model, faiss_index, top_k)
        else:
            results = search_tfidf(query, corpus_df, vectorizer, tfidf_matrix, top_k)
        
        elapsed = (time.time() - start) * 1000
        
        if results:
            st.markdown(f"### Found {len(results)} results ({elapsed:.0f}ms) | Mode: `{mode.upper()}`")
            
            for i, r in enumerate(results, 1):
                score = r['score']
                if score >= 0.5:
                    icon = "🟢"
                elif score >= 0.25:
                    icon = "🟡"
                else:
                    icon = "🔴"
                
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
        
        **Topics covered:** Resume writing • Interview prep • Salary negotiation • Career growth
        """)

# ============================================================
# TAB: ANALYTICS
# ============================================================
def render_analytics_tab(corpus_df):
    """Render the analytics tab."""
    eval_results = load_evaluation_results()
    
    st.markdown("### 📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if corpus_df is not None:
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(corpus_df):,}</div>
                <div class="metric-label">Total FAQs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{corpus_df['source'].nunique()}</div>
                <div class="metric-label">Data Sources</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_len = int(corpus_df['question'].str.len().mean())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_len}</div>
                <div class="metric-label">Avg Question Length</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if eval_results:
                mrr = eval_results.get('sbert_metrics', {}).get('MRR', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mrr:.3f}</div>
                    <div class="metric-label">SBERT MRR</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">--</div>
                    <div class="metric-label">SBERT MRR</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Evaluation Results
    if eval_results:
        st.markdown("### 🔬 Model Evaluation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### TF-IDF Performance")
            tfidf_metrics = eval_results.get('tfidf_metrics', {})
            tfidf_df = pd.DataFrame([
                {"Metric": k, "Score": f"{v:.4f}"}
                for k, v in tfidf_metrics.items()
            ])
            st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### SBERT Performance")
            sbert_metrics = eval_results.get('sbert_metrics', {})
            sbert_df = pd.DataFrame([
                {"Metric": k, "Score": f"{v:.4f}"}
                for k, v in sbert_metrics.items()
            ])
            st.dataframe(sbert_df, use_container_width=True, hide_index=True)
        
        # Comparison
        st.markdown("#### 📈 Model Comparison")
        comparison = eval_results.get('comparison', [])
        if comparison:
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            sbert_wins = sum(1 for c in comparison if c.get('Winner') == 'SBERT')
            tfidf_wins = sum(1 for c in comparison if c.get('Winner') == 'TF-IDF')
            ties = sum(1 for c in comparison if c.get('Winner') == 'Tie')
            
            if sbert_wins > tfidf_wins:
                st.success(f"🏆 **Overall Winner: SBERT** (won {sbert_wins}/{len(comparison)} metrics)")
            elif tfidf_wins > sbert_wins:
                st.success(f"🏆 **Overall Winner: TF-IDF** (won {tfidf_wins}/{len(comparison)} metrics)")
            else:
                st.info(f"🤝 **It's a Tie!** (SBERT: {sbert_wins}, TF-IDF: {tfidf_wins})")
    else:
        st.info("📊 Evaluation results not available. Run `python run_evaluation.py` locally to generate.")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📊 Visualizations")
    
    if VIZ_DIR.exists():
        viz_files = sorted(VIZ_DIR.glob("*.png"))
        if viz_files:
            for i in range(0, len(viz_files), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(viz_files):
                        with col:
                            st.image(str(viz_files[i + j]), 
                                   caption=viz_files[i + j].stem.replace('_', ' ').title(),
                                   use_container_width=True)
        else:
            st.info("📊 Visualizations not available. Run `python run_evaluation.py` locally.")
    else:
        st.info("📊 Visualizations not available. Run `python run_evaluation.py` locally.")

# ============================================================
# TAB: DATA EXPLORER
# ============================================================
def render_data_tab(corpus_df):
    """Render the data exploration tab."""
    if corpus_df is not None:
        st.markdown("### 📋 Data Overview")
        
        # Source distribution
        st.markdown("#### Source Distribution")
        source_counts = corpus_df['source'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(source_counts)
        with col2:
            for source, count in source_counts.items():
                pct = count / len(corpus_df) * 100
                st.markdown(f"**{source}**")
                st.progress(pct / 100)
                st.caption(f"{count:,} ({pct:.1f}%)")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Sample data
        st.markdown("#### 📝 Sample FAQs")
        n_samples = st.slider("Number of samples", 5, 20, 10)
        sample = corpus_df[['question', 'answer', 'source']].sample(n=min(n_samples, len(corpus_df)), random_state=42)
        st.dataframe(sample, use_container_width=True, hide_index=True)
    else:
        st.warning("Corpus not found.")

# ============================================================
# TAB: ABOUT
# ============================================================
def render_about_tab():
    """Render the about tab."""
    st.markdown("""
    ### 🎯 About This System
    
    The **Career FAQ Intelligence** system is a semantic search engine designed to help users 
    find relevant answers to career-related questions.
    
    ---
    
    #### 🔧 Technical Architecture
    
    | Component | Technology |
    |-----------|------------|
    | **Frontend** | Streamlit |
    | **Keyword Search** | TF-IDF (scikit-learn) |
    | **Semantic Search** | Sentence-BERT |
    | **Vector Index** | FAISS |
    | **Data Processing** | Pandas, NLTK |
    
    ---
    
    #### 📚 Data Sources
    
    1. **Entry Level Career QA** - Career questions for entry-level positions
    2. **CareerVillage** - Community Q&A about careers and education
    3. **HR Interview Questions** - Common interview questions with ideal answers
    
    ---
    
    #### 🔬 Retrieval Methods
    
    **TF-IDF (Term Frequency-Inverse Document Frequency)**
    - Fast keyword-based matching
    - Good for exact term searches
    - No semantic understanding
    
    **SBERT (Sentence-BERT)**
    - Understands meaning and context
    - Handles synonyms and paraphrases
    - Better for natural language questions
    
    ---
    
    #### 📊 Evaluation Metrics
    
    - **Precision@K**: How many of the top K results are relevant
    - **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result
    - **Hit Rate@K**: Whether a relevant result appears in top K
    
    ---
    
    *Built for COSC 757 - Data Mining*
    """)

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(
        page_title="Career FAQ Intelligence",
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
    with st.spinner("🔄 Loading TF-IDF model..."):
        vectorizer, tfidf_matrix = build_tfidf_model(corpus_df)
    
    sbert_model, faiss_index = None, None
    if SBERT_AVAILABLE:
        sbert_model, faiss_index = build_sbert_model(corpus_df)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if SBERT_AVAILABLE and sbert_model is not None:
            mode = st.radio(
                "Search Mode",
                ["sbert", "tfidf"],
                format_func=lambda x: "🧠 Semantic (SBERT)" if x == "sbert" else "📝 Keyword (TF-IDF)",
                index=0
            )
        else:
            mode = "tfidf"
            st.info("📝 Using TF-IDF keyword search")
        
        top_k = st.slider("Results", 1, 10, 5)
        
        st.markdown("---")
        st.header("📊 Stats")
        st.metric("Total FAQs", f"{len(corpus_df):,}")
        
        if 'source' in corpus_df.columns:
            st.markdown("**Sources:**")
            for src, cnt in corpus_df['source'].value_counts().items():
                st.text(f"• {src}: {cnt:,}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search", 
        "📊 Analytics", 
        "📋 Data Explorer",
        "ℹ️ About"
    ])
    
    with tab1:
        render_search_tab(corpus_df, vectorizer, tfidf_matrix, sbert_model, faiss_index, mode, top_k)
    
    with tab2:
        render_analytics_tab(corpus_df)
    
    with tab3:
        render_data_tab(corpus_df)
    
    with tab4:
        render_about_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("<center><small>Career FAQ Intelligence • COSC 757 Data Mining Project</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
