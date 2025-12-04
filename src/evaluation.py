"""
Evaluation Module for FAQ Recommendation System
Implements standard IR evaluation metrics:
- Precision@K
- Mean Reciprocal Rank (MRR)
- Recall@K
- Hit Rate@K

Also provides comparison between TF-IDF and SBERT retrievers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, FAQ_CORPUS_FILE
from src.tfidf_retriever import TFIDFRetriever
from src.sbert_retriever import SBERTRetriever


class RetrievalEvaluator:
    """
    Evaluator for FAQ retrieval systems.
    
    Uses a subset of the corpus as test queries and measures
    how well the system retrieves the original FAQ.
    """
    
    def __init__(self, corpus_path: Optional[Path] = None):
        """Initialize with FAQ corpus."""
        if corpus_path is None:
            corpus_path = PROCESSED_DATA_DIR / FAQ_CORPUS_FILE
        
        self.corpus = pd.read_csv(corpus_path)
        print(f"📊 Loaded corpus with {len(self.corpus)} FAQs")
    
    def create_test_set(
        self, 
        n_samples: int = 100, 
        random_state: int = 42,
        paraphrase: bool = True
    ) -> pd.DataFrame:
        """
        Create a test set by sampling from the corpus.
        The test set contains queries (questions) and their expected FAQ IDs.
        
        If paraphrase=True, modifies queries to test semantic understanding:
        - Removes some words
        - Truncates questions
        - Rephrases slightly
        This creates a more realistic evaluation scenario.
        """
        np.random.seed(random_state)
        
        # Sample from corpus
        test_indices = np.random.choice(
            len(self.corpus), 
            size=min(n_samples, len(self.corpus)), 
            replace=False
        )
        
        test_set = self.corpus.iloc[test_indices].copy()
        test_set = test_set.reset_index(drop=True)
        
        if paraphrase:
            # Store original questions for reference
            test_set['original_question'] = test_set['question']
            
            # Modify queries to simulate real user queries
            modified_queries = []
            for idx, row in test_set.iterrows():
                query = row['question']
                modification_type = idx % 5  # Cycle through 5 modification types
                
                if modification_type == 0:
                    # Remove first few words (simulate partial query)
                    words = query.split()
                    if len(words) > 5:
                        modified = ' '.join(words[2:])
                    else:
                        modified = query
                        
                elif modification_type == 1:
                    # Remove last few words
                    words = query.split()
                    if len(words) > 5:
                        modified = ' '.join(words[:-2])
                    else:
                        modified = query
                        
                elif modification_type == 2:
                    # Remove question marks and make lowercase
                    modified = query.replace('?', '').replace('!', '').lower()
                    
                elif modification_type == 3:
                    # Take only middle portion
                    words = query.split()
                    if len(words) > 6:
                        mid = len(words) // 2
                        modified = ' '.join(words[mid-2:mid+3])
                    else:
                        modified = query
                        
                else:
                    # Remove common words and keep key terms
                    stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which',
                                 'is', 'are', 'was', 'were', 'do', 'does', 'did',
                                 'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at',
                                 'i', 'you', 'we', 'they', 'it', 'my', 'your', 'can'}
                    words = query.lower().replace('?', '').split()
                    key_words = [w for w in words if w not in stop_words]
                    modified = ' '.join(key_words[:6]) if key_words else query
                
                modified_queries.append(modified.strip())
            
            test_set['question'] = modified_queries
            print(f"✅ Created test set with {len(test_set)} PARAPHRASED queries")
            print(f"   (Original questions stored for ground truth matching)")
        else:
            print(f"✅ Created test set with {len(test_set)} queries")
        
        return test_set
    
    def precision_at_k(
        self, 
        retrieved_ids: List[int], 
        relevant_id: int, 
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        For FAQ retrieval, there's typically 1 relevant document.
        """
        top_k = retrieved_ids[:k]
        return 1.0 if relevant_id in top_k else 0.0
    
    def recall_at_k(
        self, 
        retrieved_ids: List[int], 
        relevant_id: int, 
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        Same as Precision@K when there's 1 relevant document.
        """
        return self.precision_at_k(retrieved_ids, relevant_id, k)
    
    def reciprocal_rank(
        self, 
        retrieved_ids: List[int], 
        relevant_id: int
    ) -> float:
        """
        Calculate Reciprocal Rank.
        RR = 1/rank of first relevant document.
        """
        try:
            rank = retrieved_ids.index(relevant_id) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    def hit_rate_at_k(
        self, 
        retrieved_ids: List[int], 
        relevant_id: int, 
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K (same as Recall@K for single relevant doc).
        """
        return 1.0 if relevant_id in retrieved_ids[:k] else 0.0
    
    def evaluate_retriever(
        self,
        retriever,
        test_set: pd.DataFrame,
        k_values: List[int] = [1, 3, 5, 10],
        retriever_name: str = "Retriever"
    ) -> Dict[str, float]:
        """
        Evaluate a retriever on the test set.
        
        Args:
            retriever: TFIDFRetriever or SBERTRetriever instance
            test_set: DataFrame with 'question' and 'faq_id' columns
            k_values: List of K values for Precision@K and Recall@K
            retriever_name: Name for display
            
        Returns:
            Dictionary of metric names to values
        """
        print(f"\n📊 Evaluating {retriever_name}...")
        
        metrics = {f'P@{k}': [] for k in k_values}
        metrics.update({f'R@{k}': [] for k in k_values})
        metrics['MRR'] = []
        metrics['Hit@1'] = []
        
        for idx, row in tqdm(test_set.iterrows(), total=len(test_set), desc="Evaluating"):
            query = row['question']
            relevant_id = row['faq_id']
            
            # Get retrieval results
            results = retriever.search(query, top_k=max(k_values))
            retrieved_ids = [r['faq_id'] for r in results]
            
            # Calculate metrics
            for k in k_values:
                metrics[f'P@{k}'].append(
                    self.precision_at_k(retrieved_ids, relevant_id, k)
                )
                metrics[f'R@{k}'].append(
                    self.recall_at_k(retrieved_ids, relevant_id, k)
                )
            
            metrics['MRR'].append(
                self.reciprocal_rank(retrieved_ids, relevant_id)
            )
            metrics['Hit@1'].append(
                self.hit_rate_at_k(retrieved_ids, relevant_id, 1)
            )
        
        # Calculate averages
        avg_metrics = {name: np.mean(values) for name, values in metrics.items()}
        
        return avg_metrics
    
    def compare_retrievers(
        self,
        tfidf_retriever: TFIDFRetriever,
        sbert_retriever: SBERTRetriever,
        n_test_samples: int = 100,
        k_values: List[int] = [1, 3, 5, 10],
        use_paraphrased: bool = True
    ) -> Tuple[Dict, Dict, pd.DataFrame]:
        """
        Compare TF-IDF and SBERT retrievers.
        
        Args:
            use_paraphrased: If True, uses modified queries to test semantic understanding.
                           This creates a more realistic evaluation where SBERT should outperform TF-IDF.
        
        Returns:
            Tuple of (tfidf_metrics, sbert_metrics, comparison_df)
        """
        print("\n" + "="*60)
        print("   RETRIEVER COMPARISON EVALUATION")
        print("="*60)
        
        # Create test set with paraphrased queries for realistic evaluation
        test_set = self.create_test_set(n_samples=n_test_samples, paraphrase=use_paraphrased)
        
        if use_paraphrased:
            print("\n📝 Sample query modifications:")
            for i in range(min(3, len(test_set))):
                orig = test_set.iloc[i].get('original_question', test_set.iloc[i]['question'])[:60]
                modified = test_set.iloc[i]['question'][:60]
                print(f"   Original: {orig}...")
                print(f"   Modified: {modified}...")
                print()
        
        # Evaluate TF-IDF
        tfidf_metrics = self.evaluate_retriever(
            tfidf_retriever, test_set, k_values, "TF-IDF"
        )
        
        # Evaluate SBERT
        sbert_metrics = self.evaluate_retriever(
            sbert_retriever, test_set, k_values, "SBERT"
        )
        
        # Create comparison DataFrame
        comparison_data = []
        for metric in tfidf_metrics.keys():
            tfidf_val = tfidf_metrics[metric]
            sbert_val = sbert_metrics[metric]
            diff = sbert_val - tfidf_val
            
            # Determine winner (with tie handling)
            if abs(diff) < 0.001:  # Essentially equal
                winner = 'Tie'
            elif sbert_val > tfidf_val:
                winner = 'SBERT'
            else:
                winner = 'TF-IDF'
            
            comparison_data.append({
                'Metric': metric,
                'TF-IDF': round(tfidf_val, 4),
                'SBERT': round(sbert_val, 4),
                'Difference': round(diff, 4),
                'Winner': winner
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return tfidf_metrics, sbert_metrics, comparison_df


def run_evaluation(n_samples: int = 100, use_paraphrased: bool = True) -> Dict:
    """
    Run complete evaluation and return results.
    
    Args:
        n_samples: Number of test queries to use
        use_paraphrased: If True, modifies queries to test semantic understanding.
                        This is more realistic and will show SBERT's advantage.
    """
    print("\n" + "🔬"*20)
    print("   FAQ RETRIEVAL SYSTEM EVALUATION")
    print("🔬"*20 + "\n")
    
    if use_paraphrased:
        print("📝 Using PARAPHRASED queries to test semantic understanding")
        print("   This simulates real user queries that may not match exactly.\n")
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator()
    
    # Load retrievers
    print("\n📦 Loading retrievers...")
    
    tfidf = TFIDFRetriever()
    try:
        tfidf.load()
        print("   ✅ TF-IDF loaded")
    except:
        print("   ⚠️ TF-IDF not found, building...")
        tfidf.build_index()
    
    sbert = SBERTRetriever()
    try:
        sbert.load()
        print("   ✅ SBERT loaded")
    except:
        print("   ⚠️ SBERT not found, building...")
        sbert.build_index()
    
    # Run comparison
    tfidf_metrics, sbert_metrics, comparison_df = evaluator.compare_retrievers(
        tfidf, sbert, n_test_samples=n_samples, use_paraphrased=use_paraphrased
    )
    
    # Print results
    print("\n" + "="*60)
    print("   📈 EVALUATION RESULTS")
    print("="*60)
    
    print("\n📊 TF-IDF Metrics:")
    for metric, value in tfidf_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print("\n📊 SBERT Metrics:")
    for metric, value in sbert_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print("\n📊 Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Calculate overall winner
    sbert_wins = sum(1 for _, row in comparison_df.iterrows() if row['Winner'] == 'SBERT')
    tfidf_wins = sum(1 for _, row in comparison_df.iterrows() if row['Winner'] == 'TF-IDF')
    ties = sum(1 for _, row in comparison_df.iterrows() if row['Winner'] == 'Tie')
    
    if sbert_wins > tfidf_wins:
        winner = 'SBERT'
        winner_count = sbert_wins
    elif tfidf_wins > sbert_wins:
        winner = 'TF-IDF'
        winner_count = tfidf_wins
    else:
        winner = 'Tie'
        winner_count = ties
    
    print(f"\n🏆 Overall Winner: {winner}")
    print(f"   SBERT won: {sbert_wins} metrics")
    print(f"   TF-IDF won: {tfidf_wins} metrics")
    print(f"   Ties: {ties} metrics")
    
    # Key insight
    mrr_diff = sbert_metrics.get('MRR', 0) - tfidf_metrics.get('MRR', 0)
    if mrr_diff > 0.05:
        print(f"\n💡 Insight: SBERT shows {mrr_diff:.1%} better MRR - semantic understanding helps!")
    elif mrr_diff < -0.05:
        print(f"\n💡 Insight: TF-IDF shows {-mrr_diff:.1%} better MRR - keywords matter for these queries!")
    else:
        print(f"\n💡 Insight: Both models perform similarly for this test set.")
    
    # Save results
    results = {
        'tfidf_metrics': tfidf_metrics,
        'sbert_metrics': sbert_metrics,
        'comparison': comparison_df.to_dict('records'),
        'n_test_samples': n_samples
    }
    
    results_path = PROCESSED_DATA_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_evaluation(n_samples=100)

