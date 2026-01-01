"""
Multi-Run Analyzer for MIRA.

Aggregates and compares results across multiple experiment runs to
identify consistent patterns and validate findings. Essential for
establishing statistical significance and reproducibility.

Key analyses:
- Cross-run ASR consistency
- Probe accuracy stability across layers
- Statistical significance testing
- Pattern correlation analysis
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from scipy import stats


@dataclass
class RunSummary:
    """Summary of a single experiment run."""
    run_id: str
    run_dir: str
    model: str
    timestamp: str
    duration_seconds: float
    gradient_asr: float
    probe_bypass_rate: float
    probe_accuracy: float
    total_probes: int
    attack_records: Optional[pd.DataFrame] = None


class MultiRunAnalyzer:
    """
    Multi-Run Analysis Tool.
    
    Aggregates data from multiple experiment runs to identify
    consistent patterns and compute statistical measures.
    
    Key capabilities:
    - Load and parse multiple run directories
    - Compare ASR metrics across runs
    - Analyze probe accuracy by layer
    - Statistical significance testing
    - Generate comparison reports
    """
    
    def __init__(self):
        """Initialize Multi-Run Analyzer."""
        self.runs: List[RunSummary] = []
        self.comparison_data = {}
        
    def load_run(self, run_dir: str) -> Optional[RunSummary]:
        """
        Load a single run directory.
        
        Args:
            run_dir: Path to run directory
            
        Returns:
            RunSummary or None if loading fails
        """
        run_path = Path(run_dir)
        
        if not run_path.exists():
            print(f"Warning: Run directory not found: {run_dir}")
            return None
        
        # Extract run ID from directory name
        run_id = run_path.name
        
        # Load summary.json
        summary_path = run_path / "summary.json"
        if not summary_path.exists():
            print(f"Warning: No summary.json in {run_dir}")
            return None
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Try to load attack records
        records_df = None
        records_path = run_path / "mira" / "data" / "records.csv"
        if records_path.exists():
            try:
                records_df = pd.read_csv(records_path)
            except Exception as e:
                print(f"Warning: Could not load records.csv: {e}")
        
        run_summary = RunSummary(
            run_id=run_id,
            run_dir=str(run_path),
            model=summary.get('model', 'unknown'),
            timestamp=summary.get('timestamp', ''),
            duration_seconds=summary.get('duration_seconds', 0),
            gradient_asr=summary.get('gradient_asr', 0),
            probe_bypass_rate=summary.get('probe_bypass_rate', 0),
            probe_accuracy=summary.get('probe_accuracy', 0),
            total_probes=summary.get('total_probes', 0),
            attack_records=records_df,
        )
        
        return run_summary
    
    def load_runs(self, run_dirs: List[str]) -> int:
        """
        Load multiple run directories.
        
        Args:
            run_dirs: List of paths to run directories
            
        Returns:
            Number of successfully loaded runs
        """
        self.runs = []
        
        for run_dir in run_dirs:
            run = self.load_run(run_dir)
            if run:
                self.runs.append(run)
        
        print(f"Loaded {len(self.runs)} runs successfully")
        return len(self.runs)
    
    def load_from_results_dir(self, results_dir: str, pattern: str = "run_*") -> int:
        """
        Load all runs from a results directory.
        
        Args:
            results_dir: Path to results directory
            pattern: Glob pattern to match run directories
            
        Returns:
            Number of successfully loaded runs
        """
        results_path = Path(results_dir)
        run_dirs = sorted(results_path.glob(pattern))
        return self.load_runs([str(d) for d in run_dirs])
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Get summary table of all runs.
        
        Returns:
            DataFrame with run summaries
        """
        data = []
        for run in self.runs:
            data.append({
                'Run ID': run.run_id,
                'Model': run.model,
                'Gradient ASR': f"{run.gradient_asr:.1%}",
                'Probe Bypass': f"{run.probe_bypass_rate:.1%}",
                'Probe Accuracy': f"{run.probe_accuracy:.1%}",
                'Duration (min)': f"{run.duration_seconds / 60:.1f}",
                'Total Attacks': run.total_probes,
            })
        return pd.DataFrame(data)
    
    def compare_asr_metrics(self) -> Dict:
        """
        Compare ASR metrics across runs.
        
        Returns:
            Dict with statistics: mean, std, min, max, etc.
        """
        gradient_asrs = [r.gradient_asr for r in self.runs]
        bypass_rates = [r.probe_bypass_rate for r in self.runs]
        
        comparison = {
            'gradient_asr': {
                'values': gradient_asrs,
                'mean': np.mean(gradient_asrs),
                'std': np.std(gradient_asrs),
                'min': np.min(gradient_asrs),
                'max': np.max(gradient_asrs),
                'median': np.median(gradient_asrs),
            },
            'probe_bypass_rate': {
                'values': bypass_rates,
                'mean': np.mean(bypass_rates),
                'std': np.std(bypass_rates),
                'min': np.min(bypass_rates),
                'max': np.max(bypass_rates),
                'median': np.median(bypass_rates),
            },
        }
        
        # Correlation between methods
        if len(gradient_asrs) >= 3:
            correlation, p_value = stats.pearsonr(gradient_asrs, bypass_rates)
            comparison['correlation'] = {
                'pearson_r': correlation,
                'p_value': p_value,
            }
        
        return comparison
    
    def group_by_model(self) -> Dict[str, List[RunSummary]]:
        """Group runs by model name."""
        groups = {}
        for run in self.runs:
            if run.model not in groups:
                groups[run.model] = []
            groups[run.model].append(run)
        return groups
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare metrics across different models.
        
        Returns:
            DataFrame with per-model statistics
        """
        groups = self.group_by_model()
        
        data = []
        for model, runs in groups.items():
            asrs = [r.gradient_asr for r in runs]
            bypasses = [r.probe_bypass_rate for r in runs]
            
            data.append({
                'Model': model,
                'N Runs': len(runs),
                'ASR Mean': np.mean(asrs),
                'ASR Std': np.std(asrs),
                'Bypass Mean': np.mean(bypasses),
                'Bypass Std': np.std(bypasses),
            })
        
        return pd.DataFrame(data)
    
    def statistical_tests(self) -> Dict:
        """
        Run statistical significance tests.
        
        Returns:
            Dict with test results
        """
        results = {}
        
        groups = self.group_by_model()
        
        # If we have multiple models, do pairwise comparisons
        if len(groups) >= 2:
            model_names = list(groups.keys())
            results['pairwise_tests'] = []
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a = model_names[i]
                    model_b = model_names[j]
                    
                    asrs_a = [r.gradient_asr for r in groups[model_a]]
                    asrs_b = [r.gradient_asr for r in groups[model_b]]
                    
                    if len(asrs_a) >= 2 and len(asrs_b) >= 2:
                        # Mann-Whitney U test (non-parametric)
                        stat, p_value = stats.mannwhitneyu(asrs_a, asrs_b, alternative='two-sided')
                        
                        # Effect size (Cohen's d approximation)
                        pooled_std = np.sqrt((np.var(asrs_a) + np.var(asrs_b)) / 2)
                        effect_size = (np.mean(asrs_a) - np.mean(asrs_b)) / (pooled_std + 1e-10)
                        
                        results['pairwise_tests'].append({
                            'model_a': model_a,
                            'model_b': model_b,
                            'statistic': stat,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant_005': p_value < 0.05,
                        })
        
        # Overall consistency test (coefficient of variation)
        all_asrs = [r.gradient_asr for r in self.runs]
        cv = np.std(all_asrs) / (np.mean(all_asrs) + 1e-10)
        results['coefficient_of_variation'] = cv
        results['is_consistent'] = cv < 0.3  # CV < 30% is typically considered consistent
        
        return results
    
    def find_consistent_patterns(self) -> Dict:
        """
        Identify patterns that are consistent across runs.
        
        Returns:
            Dict with consistent patterns and their statistics
        """
        patterns = {
            'high_asr_runs': [],
            'low_asr_runs': [],
            'model_rankings': [],
            'stable_metrics': [],
        }
        
        # Categorize runs by ASR
        mean_asr = np.mean([r.gradient_asr for r in self.runs])
        for run in self.runs:
            if run.gradient_asr > mean_asr:
                patterns['high_asr_runs'].append(run.run_id)
            else:
                patterns['low_asr_runs'].append(run.run_id)
        
        # Rank models by mean ASR
        model_stats = self.compare_models()
        if not model_stats.empty:
            model_stats = model_stats.sort_values('ASR Mean', ascending=False)
            patterns['model_rankings'] = model_stats[['Model', 'ASR Mean']].to_dict('records')
        
        # Identify stable metrics (low variance)
        comparison = self.compare_asr_metrics()
        for metric_name, metric_stats in comparison.items():
            if isinstance(metric_stats, dict) and 'std' in metric_stats:
                cv = metric_stats['std'] / (metric_stats['mean'] + 1e-10)
                if cv < 0.2:
                    patterns['stable_metrics'].append({
                        'metric': metric_name,
                        'mean': metric_stats['mean'],
                        'cv': cv,
                    })
        
        return patterns
    
    def plot_asr_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot ASR comparison across runs.
        
        Args:
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        run_ids = [r.run_id[-6:] for r in self.runs]  # Truncate for display
        gradient_asrs = [r.gradient_asr * 100 for r in self.runs]
        bypass_rates = [r.probe_bypass_rate * 100 for r in self.runs]
        models = [r.model.split('/')[-1] for r in self.runs]
        
        # Gradient ASR
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(models))))
        model_colors = {m: colors[i] for i, m in enumerate(set(models))}
        bar_colors = [model_colors[m] for m in models]
        
        axes[0].bar(run_ids, gradient_asrs, color=bar_colors)
        axes[0].axhline(y=np.mean(gradient_asrs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(gradient_asrs):.1f}%')
        axes[0].set_xlabel('Run')
        axes[0].set_ylabel('ASR (%)')
        axes[0].set_title('Gradient Attack Success Rate')
        axes[0].legend()
        axes[0].set_xticklabels(run_ids, rotation=45, ha='right')
        
        # Probe Bypass Rate
        axes[1].bar(run_ids, bypass_rates, color=bar_colors)
        axes[1].axhline(y=np.mean(bypass_rates), color='red', linestyle='--',
                       label=f'Mean: {np.mean(bypass_rates):.1f}%')
        axes[1].set_xlabel('Run')
        axes[1].set_ylabel('Bypass Rate (%)')
        axes[1].set_title('Probe Bypass Rate')
        axes[1].legend()
        axes[1].set_xticklabels(run_ids, rotation=45, ha='right')
        
        plt.suptitle('Multi-Run ASR Comparison', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot comparison across different models.
        
        Args:
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        model_stats = self.compare_models()
        
        if model_stats.empty:
            print("No model data to compare")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        models = model_stats['Model'].tolist()
        x = np.arange(len(models))
        width = 0.35
        
        asr_means = model_stats['ASR Mean'].tolist()
        asr_stds = model_stats['ASR Std'].tolist()
        bypass_means = model_stats['Bypass Mean'].tolist()
        bypass_stds = model_stats['Bypass Std'].tolist()
        
        ax.bar(x - width/2, [v*100 for v in asr_means], width, 
               yerr=[v*100 for v in asr_stds], label='Gradient ASR', capsize=5)
        ax.bar(x + width/2, [v*100 for v in bypass_means], width,
               yerr=[v*100 for v in bypass_stds], label='Probe Bypass', capsize=5)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Attack Success by Model (with std dev)')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def generate_report(self, output_dir: str = None) -> str:
        """
        Generate comprehensive multi-run analysis report.
        
        Args:
            output_dir: Directory to save report and plots
            
        Returns:
            Report as markdown string
        """
        lines = []
        lines.append("# MIRA Multi-Run Analysis Report\n")
        lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Total Runs Analyzed**: {len(self.runs)}\n\n")
        
        # Summary table
        lines.append("## Experiment Summary\n")
        summary_df = self.get_summary_table()
        lines.append(summary_df.to_markdown(index=False))
        lines.append("\n\n")
        
        # ASR Comparison
        lines.append("## ASR Metrics Comparison\n")
        comparison = self.compare_asr_metrics()
        
        for metric, stats in comparison.items():
            if isinstance(stats, dict) and 'mean' in stats:
                lines.append(f"### {metric.replace('_', ' ').title()}\n")
                lines.append(f"- Mean: {stats['mean']:.1%}\n")
                lines.append(f"- Std Dev: {stats['std']:.1%}\n")
                lines.append(f"- Range: {stats['min']:.1%} - {stats['max']:.1%}\n")
                lines.append(f"- Median: {stats['median']:.1%}\n\n")
        
        # Model Comparison
        if len(self.group_by_model()) > 1:
            lines.append("## Model Comparison\n")
            model_df = self.compare_models()
            lines.append(model_df.to_markdown(index=False))
            lines.append("\n\n")
        
        # Statistical Tests
        lines.append("## Statistical Analysis\n")
        stats_results = self.statistical_tests()
        
        cv = stats_results.get('coefficient_of_variation', 0)
        is_consistent = stats_results.get('is_consistent', False)
        lines.append(f"- Coefficient of Variation: {cv:.2%}\n")
        lines.append(f"- Results Consistent: {'✅ Yes' if is_consistent else '⚠️ No'}\n\n")
        
        if 'pairwise_tests' in stats_results:
            lines.append("### Pairwise Model Comparisons\n")
            for test in stats_results['pairwise_tests']:
                sig = "✅" if test['significant_005'] else "❌"
                lines.append(f"- {test['model_a']} vs {test['model_b']}: ")
                lines.append(f"p={test['p_value']:.4f}, effect={test['effect_size']:.2f} {sig}\n")
        
        # Consistent Patterns
        lines.append("\n## Consistent Patterns\n")
        patterns = self.find_consistent_patterns()
        
        if patterns['model_rankings']:
            lines.append("### Model Rankings by ASR\n")
            for i, rank in enumerate(patterns['model_rankings'], 1):
                lines.append(f"{i}. {rank['Model']}: {rank['ASR Mean']:.1%}\n")
        
        if patterns['stable_metrics']:
            lines.append("\n### Stable Metrics (Low Variance)\n")
            for metric in patterns['stable_metrics']:
                lines.append(f"- {metric['metric']}: CV={metric['cv']:.1%}\n")
        
        report = ''.join(lines)
        
        # Save report and plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save report
            report_path = output_path / "multi_run_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Saved report: {report_path}")
            
            # Generate plots
            self.plot_asr_comparison(str(output_path / "asr_comparison.png"))
            if len(self.group_by_model()) > 1:
                self.plot_model_comparison(str(output_path / "model_comparison.png"))
        
        return report


def analyze_runs(run_dirs: List[str], output_dir: str = None) -> Dict:
    """
    Convenience function to analyze multiple runs.
    
    Args:
        run_dirs: List of run directory paths
        output_dir: Optional directory to save results
        
    Returns:
        Dict with analyzer and report
    """
    analyzer = MultiRunAnalyzer()
    analyzer.load_runs(run_dirs)
    
    result = {
        'analyzer': analyzer,
        'summary': analyzer.get_summary_table(),
        'comparison': analyzer.compare_asr_metrics(),
        'patterns': analyzer.find_consistent_patterns(),
        'stats': analyzer.statistical_tests(),
    }
    
    if output_dir:
        result['report'] = analyzer.generate_report(output_dir)
    
    return result
