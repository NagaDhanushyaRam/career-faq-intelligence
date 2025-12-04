"""
Visualization Module for FAQ Recommendation System
Creates charts and visualizations for analysis and reporting.

Visualizations include:
- Word clouds from FAQ corpus
- Score distribution comparisons
- TF-IDF vs SBERT performance comparison
- Data source distribution
- Query response time analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from collections import Counter
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, FAQ_CORPUS_FILE, LOGS_DIR
from src.preprocessing import TextPreprocessor

# Create output directory for visualizations
VIZ_DIR = LOGS_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pd.DataFrame:
    """Load the FAQ corpus."""
    corpus_path = PROCESSED_DATA_DIR / FAQ_CORPUS_FILE
    return pd.read_csv(corpus_path)


def plot_data_source_distribution(corpus: pd.DataFrame, save_path: Path = None):
    """
    Create a pie chart showing distribution of FAQs by data source.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    source_counts = corpus['source'].value_counts()
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    explode = [0.02] * len(source_counts)
    
    wedges, texts, autotexts = ax.pie(
        source_counts.values,
        labels=source_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(source_counts)],
        explode=explode,
        shadow=True,
        startangle=90
    )
    
    # Style
    plt.setp(autotexts, size=12, weight='bold', color='white')
    plt.setp(texts, size=11)
    
    ax.set_title('FAQ Distribution by Data Source', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with counts
    legend_labels = [f'{src}: {count:,}' for src, count in source_counts.items()]
    ax.legend(wedges, legend_labels, title="Sources", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_question_length_distribution(corpus: pd.DataFrame, save_path: Path = None):
    """
    Create histogram of question lengths.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Question length (characters)
    corpus['q_len_chars'] = corpus['question'].str.len()
    axes[0].hist(corpus['q_len_chars'], bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Question Length (characters)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Question Lengths', fontsize=14, fontweight='bold')
    axes[0].axvline(corpus['q_len_chars'].median(), color='red', linestyle='--', 
                    label=f'Median: {corpus["q_len_chars"].median():.0f}')
    axes[0].legend()
    
    # Question length (words)
    corpus['q_len_words'] = corpus['question'].str.split().str.len()
    axes[1].hist(corpus['q_len_words'], bins=30, color='#A23B72', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Question Length (words)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Question Word Counts', fontsize=14, fontweight='bold')
    axes[1].axvline(corpus['q_len_words'].median(), color='red', linestyle='--',
                    label=f'Median: {corpus["q_len_words"].median():.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_top_words(corpus: pd.DataFrame, n_words: int = 20, save_path: Path = None):
    """
    Create bar chart of most frequent words in questions.
    """
    # Preprocess and tokenize
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    all_words = []
    for question in corpus['question'].tolist():
        tokens = preprocessor.preprocess(question, return_tokens=True)
        all_words.extend(tokens)
    
    # Count words
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(n_words)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    words = [w[0] for w in top_words]
    counts = [w[1] for w in top_words]
    
    bars = ax.barh(range(len(words)), counts, color='#2E86AB', edgecolor='white')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis()
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {n_words} Most Frequent Words in Questions', fontsize=14, fontweight='bold')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 50, i, f'{count:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_evaluation_comparison(results_path: Path = None, save_path: Path = None):
    """
    Create bar chart comparing TF-IDF vs SBERT evaluation metrics.
    """
    if results_path is None:
        results_path = PROCESSED_DATA_DIR / 'evaluation_results.json'
    
    if not results_path.exists():
        print("   ⚠️ Evaluation results not found. Run evaluation first.")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    tfidf_metrics = results['tfidf_metrics']
    sbert_metrics = results['sbert_metrics']
    
    # Select key metrics
    metrics_to_plot = ['P@1', 'P@3', 'P@5', 'MRR', 'Hit@1']
    
    tfidf_values = [tfidf_metrics.get(m, 0) for m in metrics_to_plot]
    sbert_values = [sbert_metrics.get(m, 0) for m in metrics_to_plot]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tfidf_values, width, label='TF-IDF', color='#2E86AB', edgecolor='white')
    bars2 = ax.bar(x + width/2, sbert_values, width, label='SBERT', color='#A23B72', edgecolor='white')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('TF-IDF vs SBERT: Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_score_distribution(save_path: Path = None):
    """
    Compare score distributions between TF-IDF and SBERT for sample queries.
    """
    from src.tfidf_retriever import TFIDFRetriever
    from src.sbert_retriever import SBERTRetriever
    
    # Load retrievers
    tfidf = TFIDFRetriever()
    sbert = SBERTRetriever()
    
    try:
        tfidf.load()
        sbert.load()
    except:
        print("   ⚠️ Models not loaded. Cannot create score distribution.")
        return None
    
    # Sample queries
    queries = [
        "How to write a resume?",
        "Interview preparation tips",
        "Career change advice",
        "Salary negotiation strategies",
        "Skills for data science"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, query in enumerate(queries):
        if i >= len(axes) - 1:
            break
            
        tfidf_results = tfidf.search(query, top_k=10)
        sbert_results = sbert.search(query, top_k=10)
        
        tfidf_scores = [r['score'] for r in tfidf_results]
        sbert_scores = [r['score'] for r in sbert_results]
        
        x = range(1, len(tfidf_scores) + 1)
        
        axes[i].plot(x, tfidf_scores, 'o-', label='TF-IDF', color='#2E86AB', linewidth=2, markersize=8)
        axes[i].plot(x, sbert_scores, 's-', label='SBERT', color='#A23B72', linewidth=2, markersize=8)
        
        axes[i].set_xlabel('Rank', fontsize=10)
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].set_title(f'"{query[:25]}..."', fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].set_xticks(x)
    
    # Summary in last subplot
    axes[-1].text(0.5, 0.5, 
                  'Score Comparison Summary\n\n'
                  'SBERT scores typically show:\n'
                  '• Higher absolute scores\n'
                  '• Smoother decay across ranks\n'
                  '• Better semantic matching\n\n'
                  'TF-IDF scores show:\n'
                  '• Keyword-based matching\n'
                  '• Sharper score differences\n'
                  '• Faster computation',
                  ha='center', va='center', fontsize=11,
                  transform=axes[-1].transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    axes[-1].axis('off')
    
    plt.suptitle('Score Distribution: TF-IDF vs SBERT', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_response_time_comparison(n_queries: int = 20, save_path: Path = None):
    """
    Compare response times between TF-IDF and SBERT.
    """
    from src.tfidf_retriever import TFIDFRetriever
    from src.sbert_retriever import SBERTRetriever
    
    # Load retrievers
    tfidf = TFIDFRetriever()
    sbert = SBERTRetriever()
    
    try:
        tfidf.load()
        sbert.load()
    except:
        print("   ⚠️ Models not loaded.")
        return None
    
    # Load corpus for queries
    corpus = load_corpus()
    sample_queries = corpus['question'].sample(n=n_queries, random_state=42).tolist()
    
    # Measure times
    tfidf_times = []
    sbert_times = []
    
    for query in sample_queries:
        # TF-IDF
        start = time.time()
        tfidf.search(query, top_k=5)
        tfidf_times.append((time.time() - start) * 1000)  # Convert to ms
        
        # SBERT
        start = time.time()
        sbert.search(query, top_k=5)
        sbert_times.append((time.time() - start) * 1000)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    axes[0].boxplot([tfidf_times, sbert_times], labels=['TF-IDF', 'SBERT'])
    axes[0].set_ylabel('Response Time (ms)', fontsize=12)
    axes[0].set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Bar chart of averages
    avg_times = [np.mean(tfidf_times), np.mean(sbert_times)]
    bars = axes[1].bar(['TF-IDF', 'SBERT'], avg_times, color=['#2E86AB', '#A23B72'], edgecolor='white')
    axes[1].set_ylabel('Average Response Time (ms)', fontsize=12)
    axes[1].set_title('Average Response Time Comparison', fontsize=14, fontweight='bold')
    
    # Add labels
    for bar, avg in zip(bars, avg_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{avg:.2f} ms', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.close()
    return fig


def generate_all_visualizations():
    """
    Generate all visualizations and save to the visualizations folder.
    """
    print("\n" + "📊"*20)
    print("   GENERATING VISUALIZATIONS")
    print("📊"*20 + "\n")
    
    # Load corpus
    corpus = load_corpus()
    print(f"📂 Loaded corpus: {len(corpus)} FAQs\n")
    
    # 1. Data source distribution
    print("1️⃣ Creating data source distribution chart...")
    plot_data_source_distribution(corpus, VIZ_DIR / "data_source_distribution.png")
    
    # 2. Question length distribution
    print("2️⃣ Creating question length distribution...")
    plot_question_length_distribution(corpus, VIZ_DIR / "question_length_distribution.png")
    
    # 3. Top words
    print("3️⃣ Creating top words chart...")
    plot_top_words(corpus, n_words=20, save_path=VIZ_DIR / "top_words.png")
    
    # 4. Evaluation comparison (if results exist)
    print("4️⃣ Creating evaluation comparison chart...")
    plot_evaluation_comparison(save_path=VIZ_DIR / "evaluation_comparison.png")
    
    # 5. Score distribution
    print("5️⃣ Creating score distribution comparison...")
    plot_score_distribution(save_path=VIZ_DIR / "score_distribution.png")
    
    # 6. Response time comparison
    print("6️⃣ Creating response time comparison...")
    plot_response_time_comparison(n_queries=20, save_path=VIZ_DIR / "response_time_comparison.png")
    
    print("\n" + "="*60)
    print(f"   ✅ All visualizations saved to: {VIZ_DIR}")
    print("="*60)
    
    return VIZ_DIR


if __name__ == "__main__":
    generate_all_visualizations()

