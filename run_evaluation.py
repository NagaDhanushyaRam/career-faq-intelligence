"""
Run Evaluation and Generate Visualizations
This script:
1. Evaluates TF-IDF vs SBERT using IR metrics (Precision@K, MRR, etc.)
2. Generates all visualizations for the project report

Run this after the pipeline is complete.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import run_evaluation
from src.visualization import generate_all_visualizations


def main():
    print("\n" + "="*70)
    print("   FAQ RECOMMENDATION SYSTEM - EVALUATION & VISUALIZATION")
    print("="*70)
    
    # Step 1: Run evaluation
    print("\n" + "📊 STEP 1: RUNNING EVALUATION")
    print("-"*70)
    
    try:
        results = run_evaluation(n_samples=100)
        print("\n✅ Evaluation complete!")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        results = None
    
    # Step 2: Generate visualizations
    print("\n" + "📊 STEP 2: GENERATING VISUALIZATIONS")
    print("-"*70)
    
    try:
        viz_dir = generate_all_visualizations()
        print(f"\n✅ Visualizations saved to: {viz_dir}")
    except Exception as e:
        print(f"\n❌ Visualization generation failed: {e}")
        viz_dir = None
    
    # Summary
    print("\n" + "="*70)
    print("   ✅ EVALUATION & VISUALIZATION COMPLETE!")
    print("="*70)
    
    if results:
        print("\n📈 Key Findings:")
        tfidf = results['tfidf_metrics']
        sbert = results['sbert_metrics']
        
        print(f"\n   TF-IDF Performance:")
        print(f"     • Precision@1: {tfidf.get('P@1', 0):.3f}")
        print(f"     • MRR: {tfidf.get('MRR', 0):.3f}")
        
        print(f"\n   SBERT Performance:")
        print(f"     • Precision@1: {sbert.get('P@1', 0):.3f}")
        print(f"     • MRR: {sbert.get('MRR', 0):.3f}")
        
        winner = "SBERT" if sbert.get('MRR', 0) > tfidf.get('MRR', 0) else "TF-IDF"
        print(f"\n   🏆 Better Overall: {winner}")
    
    if viz_dir:
        print(f"\n📁 Visualization files:")
        for f in Path(viz_dir).glob("*.png"):
            print(f"     • {f.name}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

