"""
Structural Evaluation Entry Point

This script runs the structural evaluation framework that validates
the movie recommendation system based on narrative and topology consistency
with academic grounding.

Academic References:
1. Reagan et al. (2016) - Emotional arc classification (6 types)
2. Kaminskas & Bridge (2017) - Beyond-accuracy metrics (ILD, Novelty, Coverage)
3. Boyd et al. (2020) - Narrative rhythm correlation
4. Tsitsulin et al. (2018) - NetLSD graph similarity
5. Ziegler et al. (2005) - Intra-list diversity
6. Ge et al. (2010) - Coverage and serendipity
7. Zhou et al. (2010) - Novelty score
8. Bauer & Zangerle (2020) - User-based LOO evaluation

Uses precomputed distance matrices for ~100x faster evaluation.
"""

import json
from src.structural_evaluator import StructuralEvaluator
from src.precomputed_retrieval import PrecomputedRetrieval
from main import load_or_process_features


def main():
    # Load features
    print("[INFO] Loading features...")
    feature_db = load_or_process_features()

    # Initialize precomputed retriever (fast!)
    print("[INFO] Initializing precomputed retriever...")
    retriever = PrecomputedRetrieval(feature_db, cache_path='data/distance_cache.pkl')

    # Load metadata
    print("[INFO] Loading metadata...")
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    # Initialize structural evaluator
    evaluator = StructuralEvaluator(
        feature_db=feature_db,
        metadata_list=metadata_list,
        parsed_dir='data/parsed'
    )

    # Run full evaluation with baselines
    print("\n[INFO] Running full evaluation with academic-grounded metrics...")
    results = evaluator.generate_full_report(
        retriever,
        k=5,
        include_baselines=True
    )

    # Save results
    print("\n[INFO] Saving results to structural_eval_results.json...")
    with open('structural_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Evaluation complete!")
    print("\n" + "=" * 70)
    print("CITATION GUIDE")
    print("=" * 70)
    print("""
When citing these evaluation metrics in your paper, please use:

Structural Consistency Metrics:
- ATC: Reagan, A. J., et al. (2016). "The emotional arcs of stories are
       dominated by six basic shapes." EPJ Data Science, 5(1), 31.
- NRC: Boyd, R. L., et al. (2020). "The narrative arc: Revealing core
       narrative structures through text analysis." Science Advances, 6(32).
- GFS: Tsitsulin, A., et al. (2018). "NetLSD: Hearing the shape of a graph."
       KDD 2018.

Beyond-Accuracy Metrics:
- ILD: Ziegler, C. N., et al. (2005). "Improving recommendation lists through
       topic diversification." WWW 2005.
- Framework: Kaminskas, M., & Bridge, D. (2017). "Diversity, serendipity,
             novelty, and coverage." ACM TIIS, 7(1), 1-42.
- Novelty: Zhou, T., et al. (2010). "Solving the apparent diversity-accuracy
           dilemma of recommender systems."
- Coverage: Ge, M., et al. (2010). "Beyond accuracy: Evaluating recommender
            systems by coverage and serendipity." RecSys 2010.

Why Genre-based PR Curve is not appropriate:
- McNee, S. M., et al. (2006). "Being accurate is not enough: How accuracy
  metrics have hurt recommender systems." CHI Extended Abstracts.

User-Based Evaluation (LOO Protocol):
- Hit@K, NDCG@K: Bauer, C., & Zangerle, E. (2020). "Offline evaluation options
                 for recommender systems." Information Retrieval Journal.
- Evaluation Framework: Zangerle, E., et al. (2022). "Evaluating Recommender
                        Systems: Survey and Framework." ACM Computing Surveys.
""")


if __name__ == '__main__':
    main()
