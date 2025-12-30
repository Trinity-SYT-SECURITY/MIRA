"""
Multi-Model Comparison Mode

Compare attack success rates across multiple models.
"""

from mira.analysis.comparison import MultiModelRunner, get_recommended_models


def run_multi_model_comparison():
    """Run multi-model comparison mode."""
    print("\n" + "="*70)
    print("  MULTI-MODEL COMPARISON MODE")
    print("="*70 + "\n")
    
    # Get max model size
    try:
        max_size = input("  Max model size in GB (default: 1.0): ").strip()
        max_size = float(max_size) if max_size else 1.0
    except:
        max_size = 1.0
    
    # Get number of attacks
    try:
        num_attacks = input("  Attacks per model (default: 5): ").strip()
        num_attacks = int(num_attacks) if num_attacks else 5
    except:
        num_attacks = 5
    
    print()
    
    # Run comparison
    runner = MultiModelRunner()
    models = get_recommended_models(max_size_gb=max_size)
    
    if not models:
        print(f"  No models found under {max_size} GB")
        return
    
    report = runner.run_comparison(
        models=models,
        num_attacks=num_attacks,
        max_model_size_gb=max_size,
    )
    
    print("\n" + report.summary_table())
    print(f"\n  Report saved to: results/comparison/")
