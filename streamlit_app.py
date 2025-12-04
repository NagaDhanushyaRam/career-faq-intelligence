"""
FAQ Recommendation System - Professional UI
A modern, professional interface for career FAQ search with analytics.
"""
import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path
from typing import Optional
import time

# Configuration
API_URL = "http://localhost:8000"
PROCESSED_DATA_DIR = Path("data/processed")
LOGS_DIR = Path("logs")
VIZ_DIR = LOGS_DIR / "visualizations"

# ============================================================
# API FUNCTIONS
# ============================================================

def search_faqs(query: str, mode: str, top_k: int) -> Optional[dict]:
    """Call the API to search for FAQs."""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "mode": mode, "top_k": top_k, "threshold": 0.0},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.RequestException:
        return None


def get_stats() -> Optional[dict]:
    """Get corpus statistics from API."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None


def load_evaluation_results():
    """Load evaluation results if available."""
    eval_path = PROCESSED_DATA_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            return json.load(f)
    return None


def load_corpus():
    """Load FAQ corpus."""
    corpus_path = PROCESSED_DATA_DIR / "faq_corpus.csv"
    if corpus_path.exists():
        return pd.read_csv(corpus_path)
    return None


# ============================================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# ============================================================

def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Header Styles */
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        .sub-title {
            font-size: 1.1rem;
            color: #a0aec0;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Card Styles */
        .result-card {
            background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
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
            letter-spacing: 0.05em;
        }
        
        /* Score Badge */
        .score-high { color: #48bb78; font-weight: 600; }
        .score-medium { color: #ecc94b; font-weight: 600; }
        .score-low { color: #fc8181; font-weight: 600; }
        
        /* Question/Answer Boxes */
        .question-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        
        .answer-box {
            background: rgba(72, 187, 120, 0.1);
            border-left: 4px solid #48bb78;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        
        /* Status Indicators */
        .status-online {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(72, 187, 120, 0.2);
            color: #48bb78;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .status-offline {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(252, 129, 129, 0.2);
            color: #fc8181;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Tab Styles */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Sidebar Styles */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Divider */
        .custom-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            margin: 1.5rem 0;
        }
        
        /* Source tag */
        .source-tag {
            display: inline-block;
            background: rgba(118, 75, 162, 0.2);
            color: #b794f6;
            padding: 0.2rem 0.8rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-title">🎯 Career FAQ Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-powered answers to your career questions • Powered by SBERT & TF-IDF</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings and status."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Search mode
        mode = st.selectbox(
            "Search Algorithm",
            options=["sbert", "tfidf"],
            format_func=lambda x: "🧠 Semantic (SBERT)" if x == "sbert" else "📝 Keyword (TF-IDF)",
            help="SBERT understands meaning and context. TF-IDF matches exact keywords."
        )
        
        # Number of results
        top_k = st.slider("Results to show", 1, 10, 5)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown("## 📊 System Status")
        stats = get_stats()
        
        if stats:
            st.markdown('<span class="status-online">● API Online</span>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("FAQs", f"{stats['total_faqs']:,}")
            with col2:
                st.metric("Sources", len(stats.get('sources', {})))
            
            # Model status
            st.markdown("**Models:**")
            col1, col2 = st.columns(2)
            with col1:
                if stats.get('tfidf_ready'):
                    st.success("TF-IDF ✓")
                else:
                    st.error("TF-IDF ✗")
            with col2:
                if stats.get('sbert_ready'):
                    st.success("SBERT ✓")
                else:
                    st.error("SBERT ✗")
        else:
            st.markdown('<span class="status-offline">● API Offline</span>', unsafe_allow_html=True)
            st.info("Start API: `python -m uvicorn api.main:app --reload`")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Info
        st.markdown("## 💡 Tips")
        st.markdown("""
        - **Semantic** mode understands synonyms
        - **Keyword** mode is faster
        - Try different phrasings for better results
        """)
        
        return mode, top_k


def render_search_tab(mode: str, top_k: int):
    """Render the search tab."""
    # Example queries
    st.markdown("#### Quick Examples")
    col1, col2, col3, col4 = st.columns(4)
    
    examples = [
        ("📝", "Resume tips"),
        ("🎤", "Interview prep"),
        ("💰", "Salary negotiation"),
        ("🚀", "Career growth")
    ]
    
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    for i, (icon, text) in enumerate(examples):
        with [col1, col2, col3, col4][i]:
            if st.button(f"{icon} {text}", key=f"ex_{i}", use_container_width=True):
                st.session_state.current_query = text
                st.rerun()
    
    st.markdown("")
    
    # Search input
    query = st.text_input(
        "🔍 Ask your career question",
        value=st.session_state.current_query,
        placeholder="e.g., How do I prepare for a technical interview?",
        label_visibility="collapsed"
    )
    
    if query != st.session_state.current_query:
        st.session_state.current_query = query
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Results
    if query:
        with st.spinner("🔍 Searching..."):
            start_time = time.time()
            results = search_faqs(query, mode, top_k)
            search_time = (time.time() - start_time) * 1000
        
        if results and results.get('results'):
            # Results header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"### Found {results['total_results']} results")
            with col2:
                st.markdown(f"**Mode:** `{results['mode'].upper()}`")
            with col3:
                st.markdown(f"**Time:** `{search_time:.0f}ms`")
            
            st.markdown("")
            
            # Results
            for i, result in enumerate(results['results'], 1):
                with st.container():
                    # Header row
                    col1, col2, col3 = st.columns([0.5, 2, 1])
                    
                    with col1:
                        st.markdown(f"### #{i}")
                    
                    with col2:
                        score = result['score']
                        if score >= 0.5:
                            score_class = "score-high"
                            score_icon = "🟢"
                        elif score >= 0.25:
                            score_class = "score-medium"
                            score_icon = "🟡"
                        else:
                            score_class = "score-low"
                            score_icon = "🔴"
                        
                        st.markdown(f'{score_icon} <span class="{score_class}">Score: {score:.3f}</span>', 
                                  unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'<span class="source-tag">{result["source"]}</span>', 
                                  unsafe_allow_html=True)
                    
                    # Question
                    st.markdown(f'<div class="question-box"><strong>❓ Question:</strong><br>{result["question"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Answer
                    st.markdown(f'<div class="answer-box"><strong>✅ Answer:</strong><br>{result["answer"]}</div>', 
                              unsafe_allow_html=True)
                    
                    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        elif results:
            st.warning("No results found. Try rephrasing your question.")
        else:
            st.error("❌ Cannot connect to API. Make sure the backend is running.")
    
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to Career FAQ Intelligence</h2>
            <p style="color: #a0aec0; font-size: 1.1rem;">
                Ask any career-related question and get instant, relevant answers.
            </p>
            <br>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div>📝 Resume Writing</div>
                <div>🎤 Interview Prep</div>
                <div>💰 Salary Negotiation</div>
                <div>🚀 Career Growth</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_analytics_tab():
    """Render the analytics tab."""
    eval_results = load_evaluation_results()
    corpus = load_corpus()
    
    # Metrics row
    st.markdown("### 📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if corpus is not None:
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(corpus):,}</div>
                <div class="metric-label">Total FAQs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{corpus['source'].nunique()}</div>
                <div class="metric-label">Data Sources</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_len = int(corpus['question'].str.len().mean())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_len}</div>
                <div class="metric-label">Avg Question Length</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if eval_results:
                mrr = eval_results['sbert_metrics'].get('MRR', 0)
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
            tfidf_df = pd.DataFrame([
                {"Metric": k, "Score": f"{v:.4f}"}
                for k, v in eval_results['tfidf_metrics'].items()
            ])
            st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### SBERT Performance")
            sbert_df = pd.DataFrame([
                {"Metric": k, "Score": f"{v:.4f}"}
                for k, v in eval_results['sbert_metrics'].items()
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
                st.success(f"🏆 **Overall Winner: SBERT** (won {sbert_wins}/{len(comparison)} metrics, {ties} ties)")
            elif tfidf_wins > sbert_wins:
                st.success(f"🏆 **Overall Winner: TF-IDF** (won {tfidf_wins}/{len(comparison)} metrics, {ties} ties)")
            else:
                st.info(f"🤝 **It's a Tie!** (SBERT: {sbert_wins}, TF-IDF: {tfidf_wins}, Ties: {ties})")
    
    else:
        st.info("📊 Run `python run_evaluation.py` to generate evaluation results.")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📊 Visualizations")
    
    if VIZ_DIR.exists():
        viz_files = sorted(VIZ_DIR.glob("*.png"))
        if viz_files:
            # Display in 2-column grid
            for i in range(0, len(viz_files), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(viz_files):
                        with col:
                            st.image(str(viz_files[i + j]), 
                                   caption=viz_files[i + j].stem.replace('_', ' ').title(),
                                   use_container_width=True)
        else:
            st.info("📊 Run `python run_evaluation.py` to generate visualizations.")
    else:
        st.info("📊 Run `python run_evaluation.py` to generate visualizations.")


def render_data_tab():
    """Render the data exploration tab."""
    corpus = load_corpus()
    
    if corpus is not None:
        st.markdown("### 📋 Data Overview")
        
        # Source distribution
        st.markdown("#### Source Distribution")
        source_counts = corpus['source'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(source_counts)
        with col2:
            for source, count in source_counts.items():
                pct = count / len(corpus) * 100
                st.markdown(f"**{source}**")
                st.progress(pct / 100)
                st.caption(f"{count:,} ({pct:.1f}%)")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Sample data
        st.markdown("#### 📝 Sample FAQs")
        n_samples = st.slider("Number of samples", 5, 20, 10)
        
        sample = corpus[['question', 'answer', 'source']].sample(n=n_samples, random_state=42)
        st.dataframe(sample, use_container_width=True, hide_index=True)
    
    else:
        st.warning("Corpus not found. Run the data pipeline first.")


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
    | **Backend** | FastAPI + Uvicorn |
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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Header
    render_header()
    
    # Sidebar
    mode, top_k = render_sidebar()
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search", 
        "📊 Analytics", 
        "📋 Data Explorer",
        "ℹ️ About"
    ])
    
    with tab1:
        render_search_tab(mode, top_k)
    
    with tab2:
        render_analytics_tab()
    
    with tab3:
        render_data_tab()
    
    with tab4:
        render_about_tab()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666; font-size: 0.85rem;">
        Career FAQ Intelligence • COSC 757 Data Mining Project • 
        Powered by SBERT, FAISS, FastAPI & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
