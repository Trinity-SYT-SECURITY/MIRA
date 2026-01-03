"""
Analysis script for comparing PAIR vs GCG attack signatures

Generates statistical comparisons and visualizations for paper Section 4.8
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_experiment_results(results_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """Load both PAIR and GCG experimental results"""
    
    results = {}
    
    # Load PAIR results
    pair_files = list(Path(results_dir).glob("pair_validation/*_pair_results.json"))
    if pair_files:
        pair_data = []
        for f in pair_files:
            with open(f) as file:
                pair_data.extend(json.load(file))
        results['pair'] = pd.DataFrame(pair_data)
        print(f"Loaded {len(pair_data)} PAIR samples")
    
    # Load GCG results (from existing experiments)
    gcg_files = list(Path(results_dir).glob("**/gcg_results.json"))
    if gcg_files:
        gcg_data = []
        for f in gcg_files:
            with open(f) as file:
                gcg_data.extend(json.load(file))
        results['gcg'] = pd.DataFrame(gcg_data)
        print(f"Loaded {len(gcg_data)} GCG samples")
    
    return results


def compute_signature_statistics(df: pd.DataFrame, attack_type: str) -> Dict:
    """Compute summary statistics for representational signatures"""
    
    # Filter successful attacks only
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        return {'attack_type': attack_type, 'n': 0, 'asr': 0.0}
    
    stats_dict = {
        'attack_type': attack_type,
        'n_total': len(df),
        'n_successful': len(successful),
        'asr': len(successful) / len(df),
    }
    
    # KL Divergence statistics
    if 'signatures' in successful.columns:
        kl_values = [r.get('kl_drift_layer0') 
                     for r in successful['signatures'] 
                     if r and r.get('kl_drift_layer0') is not None]
        
        if kl_values:
            stats_dict.update({
                'kl_mean': np.mean(kl_values),
                'kl_std': np.std(kl_values),
                'kl_median': np.median(kl_values),
                'kl_min': np.min(kl_values),
                'kl_max': np.max(kl_values),
                'kl_ci_lower': np.percentile(kl_values, 2.5),
                'kl_ci_upper': np.percentile(kl_values, 97.5),
            })
        
        # Probe accuracy
        probe_values = [r.get('probe_accuracy') 
                        for r in successful['signatures'] 
                        if r and r.get('probe_accuracy') is not None]
        
        if probe_values:
            stats_dict.update({
                'probe_mean': np.mean(probe_values),
                'probe_std': np.std(probe_values),
            })
    
    return stats_dict


def statistical_comparison(pair_df: pd.DataFrame, gcg_df: pd.DataFrame) -> Dict:
    """Perform statistical tests comparing PAIR vs GCG signatures"""
    
    # Extract KL values from successful attacks
    pair_successful = pair_df[pair_df['success'] == True]
    gcg_successful = gcg_df[gcg_df['success'] == True]
    
    pair_kl = [r.get('kl_drift_layer0') 
               for r in pair_successful['signatures'] 
               if r and r.get('kl_drift_layer0') is not None]
    
    gcg_kl = [r.get('kl_drift_layer0') 
              for r in gcg_successful['signatures'] 
              if r and r.get('kl_drift_layer0') is not None]
    
    results = {}
    
    # Two-sample t-test
    if len(pair_kl) > 1 and len(gcg_kl) > 1:
        t_stat, p_value = stats.ttest_ind(pair_kl, gcg_kl)
        results['ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(pair_kl)**2 + np.std(gcg_kl)**2) / 2)
        cohens_d = (np.mean(gcg_kl) - np.mean(pair_kl)) / pooled_std
        results['effect_size'] = cohens_d
    
    # Test PAIR vs baseline
    baseline_kl = 2.3  # From our baseline experiments
    if len(pair_kl) > 0:
        t_stat_baseline, p_baseline = stats.ttest_1samp(pair_kl, baseline_kl)
        results['pair_vs_baseline'] = {
            't_statistic': t_stat_baseline,
            'p_value': p_baseline,
            'significant': p_baseline < 0.05
        }
    
    return results


def create_comparison_plots(pair_df: pd.DataFrame, gcg_df: pd.DataFrame, output_dir: str = "figures"):
    """Generate comparison visualizations for paper"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    def get_signatures(df, attack_name):
        successful = df[df['success'] == True]
        data = []
        for _, row in successful.iterrows():
            sigs = row.get('signatures', {})
            if sigs and sigs.get('kl_drift_layer0'):
                data.append({
                    'attack': attack_name,
                    'kl_drift': sigs['kl_drift_layer0'],
                    'probe_acc': sigs.get('probe_accuracy', None)
                })
        return pd.DataFrame(data)
    
    pair_sigs = get_signatures(pair_df, 'PAIR')
    gcg_sigs = get_signatures(gcg_df, 'GCG')
    combined = pd.concat([pair_sigs, gcg_sigs], ignore_index=True)
    
    # Figure 1: KL Divergence Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot
    ax = axes[0]
    sns.boxplot(data=combined, x='attack', y='kl_drift', ax=ax)
    ax.axhline(2.3, color='red', linestyle='--', label='Baseline', linewidth=2)
    ax.set_ylabel('Layer 0 KL Divergence')
    ax.set_title('(a) KL Drift Distribution')
    ax.legend()
    ax.set_ylim(0, 35)
    
    # Violin plot
    ax = axes[1]
    sns.violinplot(data=combined, x='attack', y='kl_drift', ax=ax)
    ax.set_ylabel('Layer 0 KL Divergence')
    ax.set_title('(b) KL Drift Density')
    ax.set_ylim(0, 35)
    
    # Probe accuracy
    ax = axes[2]
    probe_data = combined[combined['probe_acc'].notna()]
    sns.boxplot(data=probe_data, x='attack', y='probe_acc', ax=ax)
    ax.axhline(0.85, color='red', linestyle='--', label='Safe Threshold')
    ax.axhline(0.50, color='orange', linestyle='--', label='Random Chance')
    ax.set_ylabel('Probe Classification Accuracy')
    ax.set_title('(c) Safety Probe Accuracy')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pair_vs_gcg_signatures.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/pair_vs_gcg_signatures.png")
    
    plt.close()


def generate_latex_table(pair_stats: Dict, gcg_stats: Dict, test_results: Dict) -> str:
    """Generate LaTeX table for paper"""
    
    latex = r"""\begin{table}[t]
\centering
\caption{Attack Signature Comparison: PAIR vs GCG}
\label{tab:pair_gcg_comparison}
\begin{tabular}{lcc}
\toprule
Metric & PAIR & GCG \\
\midrule
Attack Success Rate & $VALUE$ & $VALUE$ \\
Layer 0 KL Drift (mean ± std) & $VALUE$ & $VALUE$ \\
Layer 0 KL Drift (95\% CI) & $VALUE$ & $VALUE$ \\
Probe Accuracy (mean ± std) & $VALUE$ & $VALUE$ \\
\midrule
\multicolumn{3}{l}{\textit{Statistical Tests}} \\
Two-sample t-test & \multicolumn{2}{c}{$t=VALUE$, $p<VALUE$} \\
Effect size (Cohen's d) & \multicolumn{2}{c}{$d=VALUE$} \\
\bottomrule
\end{tabular}
\end{table}"""
    
    # Fill in values (simplified)
    print("\n=== LaTeX Table Template ===")
    print(latex)
    print("\n=== Fill with actual values ===\n")
    
    return latex


def main():
    """Run complete analysis pipeline"""
    
    print("\n" + "="*60)
    print("PAIR vs GCG Signature Analysis")
    print("="*60 + "\n")
    
    # Load data
    results = load_experiment_results()
    
    if 'pair' not in results or 'gcg' not in results:
        print("ERROR: Missing experimental data")
        print("  Need both PAIR and GCG results")
        return
    
    pair_df = results['pair']
    gcg_df = results['gcg']
    
    # Compute statistics
    print("\n--- Computing Statistics ---\n")
    pair_stats = compute_signature_statistics(pair_df, 'PAIR')
    gcg_stats = compute_signature_statistics(gcg_df, 'GCG')
    
    print(f"PAIR Statistics (n={pair_stats['n_successful']}):")
    for k, v in pair_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    
    print(f"\nGCG Statistics (n={gcg_stats['n_successful']}):")
    for k, v in gcg_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    
    # Statistical tests
    print("\n--- Statistical Comparison ---\n")
    test_results = statistical_comparison(pair_df, gcg_df)
    
    if 'ttest' in test_results:
        print(f"Two-sample t-test:")
        print(f"  t = {test_results['ttest']['t_statistic']:.2f}")
        print(f"  p = {test_results['ttest']['p_value']:.4f}")
        print(f"  Significant: {test_results['ttest']['significant']}")
        print(f"\nEffect size (Cohen's d): {test_results['effect_size']:.2f}")
    
    if 'pair_vs_baseline' in test_results:
        print(f"\nPAIR vs Baseline:")
        print(f"  t = {test_results['pair_vs_baseline']['t_statistic']:.2f}")
        print(f"  p = {test_results['pair_vs_baseline']['p_value']:.4f}")
        print(f"  PAIR shows significant elevation: {test_results['pair_vs_baseline']['significant']}")
    
    # Generate plots
    print("\n--- Generating Visualizations ---\n")
    create_comparison_plots(pair_df, gcg_df)
    
    # Generate LaTeX table
    generate_latex_table(pair_stats, gcg_stats, test_results)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
