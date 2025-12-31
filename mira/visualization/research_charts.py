"""
Research chart generation module.

Automatically generates publication-quality charts during
experiment runs for research papers and reports.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Publication-quality style settings
RESEARCH_STYLE = {
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Color palettes for different use cases
COLORS = {
    "primary": "#3498db",
    "secondary": "#2ecc71",
    "accent": "#e74c3c",
    "neutral": "#95a5a6",
    "success": "#27ae60",
    "failure": "#c0392b",
    "warning": "#f39c12",  # Orange for medium ASR
    "before": "#3498db",
    "after": "#e74c3c",
}


class ResearchChartGenerator:
    """
    Generator for research-quality visualizations.
    
    Produces charts suitable for academic papers,
    with consistent styling and proper annotations.
    """
    
    def __init__(
        self,
        output_dir: str = "./charts",
        style: Optional[Dict[str, Any]] = None,
        interactive: bool = False,
    ):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory for saving charts
            style: Override default style settings
            interactive: Use Plotly for interactive charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = {**RESEARCH_STYLE, **(style or {})}
        self.interactive = interactive and HAS_PLOTLY
        
        if HAS_MATPLOTLIB:
            plt.rcParams.update(self.style)
    
    def plot_attack_success_rate(
        self,
        models: List[str],
        asr_values: List[float],
        title: str = "Attack Success Rate by Model",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate bar chart comparing ASR across models.
        
        Args:
            models: List of model names
            asr_values: ASR for each model
            title: Chart title
            save_name: Filename (without extension)
            
        Returns:
            Path to saved chart or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [COLORS["primary"] if asr < 0.5 else COLORS["accent"] 
                  for asr in asr_values]
        
        bars = ax.bar(models, asr_values, color=colors, edgecolor="white", linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, asr_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        
        ax.set_ylabel("Attack Success Rate")
        ax.set_xlabel("Model")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            from pathlib import Path
            # Ensure output directory exists
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            path = output_path / f"{save_name}.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight')
            plt.close()
            # Verify file was created
            if path.exists():
                return str(path)
            else:
                return None
        
        return None
    
    def plot_loss_curve(
        self,
        loss_history: List[float],
        title: str = "Attack Optimization Progress",
        save_name: Optional[str] = None,
        highlight_min: bool = True,
    ) -> Optional[str]:
        """
        Plot loss curve during attack optimization.
        
        Args:
            loss_history: Loss values over steps
            title: Chart title
            save_name: Filename
            highlight_min: Mark minimum point
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = list(range(len(loss_history)))
        ax.plot(steps, loss_history, color=COLORS["primary"], linewidth=2, alpha=0.8)
        ax.fill_between(steps, loss_history, alpha=0.2, color=COLORS["primary"])
        
        if highlight_min and loss_history:
            min_idx = np.argmin(loss_history)
            min_val = loss_history[min_idx]
            ax.scatter([min_idx], [min_val], color=COLORS["accent"], s=100, zorder=5)
            ax.annotate(
                f"Min: {min_val:.4f}",
                (min_idx, min_val),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=10,
                fontweight="bold",
            )
        
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_subspace_comparison(
        self,
        before_distances: List[float],
        after_distances: List[float],
        labels: Optional[List[str]] = None,
        title: str = "Subspace Distance Before vs After Attack",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Compare subspace distances before and after attack.
        
        Args:
            before_distances: Distances before attack
            after_distances: Distances after attack  
            labels: Optional labels for each sample
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n = len(before_distances)
        x = np.arange(n)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_distances, width, 
                       label="Before Attack", color=COLORS["before"], alpha=0.8)
        bars2 = ax.bar(x + width/2, after_distances, width,
                       label="After Attack", color=COLORS["after"], alpha=0.8)
        
        if labels:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax.set_xlabel("Sample Index")
        
        ax.set_ylabel("Subspace Distance")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Pattern",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate attention heatmap.
        
        Args:
            attention_matrix: 2D attention weights
            tokens: Token labels
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cmap = plt.cm.Blues
        im = ax.imshow(attention_matrix, cmap=cmap, aspect="auto")
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Attention Weight", fontsize=12)
        
        if tokens and len(tokens) <= 30:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=9)
            ax.set_yticklabels(tokens, fontsize=9)
        
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(title, fontweight="bold")
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_entropy_distribution(
        self,
        entropy_values: List[float],
        labels: Optional[List[str]] = None,
        title: str = "Entropy Distribution",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot entropy distribution histogram.
        
        Args:
            entropy_values: List of entropy values
            labels: Optional labels
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(entropy_values, bins=30, color=COLORS["primary"], 
                alpha=0.7, edgecolor="white")
        
        mean_entropy = np.mean(entropy_values)
        ax.axvline(mean_entropy, color=COLORS["accent"], linestyle="--", linewidth=2)
        ax.text(
            mean_entropy + 0.01 * (max(entropy_values) - min(entropy_values)),
            ax.get_ylim()[1] * 0.9,
            f"Mean: {mean_entropy:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_layer_activation_trajectory(
        self,
        trajectories: Dict[str, List[float]],
        title: str = "Activation Trajectory Across Layers",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot activation magnitude across layers.
        
        Args:
            trajectories: Dict mapping label to layer-wise values
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, (label, values) in enumerate(trajectories.items()):
            color = colors[i % len(colors)]
            layers = list(range(len(values)))
            ax.plot(layers, values, marker="o", label=label, 
                    color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel("Layer")
        ax.set_ylabel("Activation Magnitude")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_comparison_radar(
        self,
        categories: List[str],
        values_dict: Dict[str, List[float]],
        title: str = "Model Comparison",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate radar chart for multi-dimensional comparison.
        
        Args:
            categories: Metric categories
            values_dict: Dict mapping model to metric values
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        n = len(categories)
        angles = [i / float(n) * 2 * np.pi for i in range(n)]
        angles += angles[:1]  # Complete the circle
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, (label, values) in enumerate(values_dict.items()):
            values_plot = values + values[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, values_plot, "o-", linewidth=2, label=label, color=color)
            ax.fill(angles, values_plot, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def generate_experiment_summary(
        self,
        experiment_data: Dict[str, Any],
        save_name: str = "experiment_summary",
    ) -> List[str]:
        """
        Generate a complete set of charts for an experiment.
        
        Args:
            experiment_data: Dictionary with experiment results
            save_name: Base name for charts
            
        Returns:
            List of paths to generated charts
        """
        charts = []
        
        # Loss curve if available
        if "loss_history" in experiment_data:
            path = self.plot_loss_curve(
                experiment_data["loss_history"],
                save_name=f"{save_name}_loss",
            )
            if path:
                charts.append(path)
        
        # ASR comparison if available
        if "asr_by_model" in experiment_data:
            models = list(experiment_data["asr_by_model"].keys())
            values = list(experiment_data["asr_by_model"].values())
            path = self.plot_attack_success_rate(
                models, values,
                save_name=f"{save_name}_asr",
            )
            if path:
                charts.append(path)
        
        # Entropy distribution if available
        if "entropy_values" in experiment_data:
            path = self.plot_entropy_distribution(
                experiment_data["entropy_values"],
                save_name=f"{save_name}_entropy",
            )
            if path:
                charts.append(path)
        
        return charts
    
    def plot_asr_by_attack_type(
        self,
        attack_types: List[str],
        asr_values: List[float],
        model_name: Optional[str] = None,
        title: str = "ASR by Attack Type",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot ASR comparison across different attack types.
        
        Args:
            attack_types: List of attack type names (e.g., "Prompt Injection", "GCG", "Probe")
            asr_values: ASR for each attack type
            model_name: Optional model name for title
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = [COLORS["success"] if asr < 0.3 else COLORS["warning"] if asr < 0.7 else COLORS["failure"]
                  for asr in asr_values]
        
        bars = ax.bar(attack_types, asr_values, color=colors, edgecolor="white", linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, asr_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        
        ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
        ax.set_xlabel("Attack Type", fontsize=12)
        if model_name:
            ax.set_title(f"{title} - {model_name}", fontweight="bold", fontsize=14)
        else:
            ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5, label="50% threshold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if save_name:
            from pathlib import Path
            # Ensure output directory exists
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            path = output_path / f"{save_name}.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight')
            plt.close()
            # Verify file was created
            if path.exists():
                return str(path)
            else:
                return None
        
        return None
    
    def plot_phase_wise_asr(
        self,
        phases: List[str],
        asr_by_phase: Dict[str, List[float]],
        model_names: Optional[List[str]] = None,
        title: str = "Phase-wise ASR Comparison",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot ASR across different attack phases.
        
        Args:
            phases: List of phase names
            asr_by_phase: Dict mapping phase to ASR values (or single list if one model)
            model_names: Optional list of model names
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(phases))
        width = 0.35 if model_names and len(model_names) > 1 else 0.6
        
        if isinstance(asr_by_phase, dict) and model_names:
            # Multiple models
            colors = list(mcolors.TABLEAU_COLORS.values())
            for i, model in enumerate(model_names):
                if model in asr_by_phase:
                    values = asr_by_phase[model]
                    offset = (i - len(model_names)/2 + 0.5) * width / len(model_names)
                    ax.bar(x + offset, values, width/len(model_names), 
                          label=model, color=colors[i % len(colors)], alpha=0.8)
        else:
            # Single model or list
            if isinstance(asr_by_phase, list):
                values = asr_by_phase
            else:
                values = list(asr_by_phase.values())[0] if asr_by_phase else []
            
            bars = ax.bar(x, values, width, color=COLORS["primary"], alpha=0.8)
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        
        ax.set_ylabel("Attack Success Rate", fontsize=12)
        ax.set_xlabel("Attack Phase", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        if model_names and len(model_names) > 1:
            ax.legend()
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_cumulative_asr(
        self,
        steps: List[int],
        cumulative_asr: List[float],
        title: str = "Cumulative ASR Over Time",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot cumulative ASR curve showing method stability.
        
        Args:
            steps: Step numbers
            cumulative_asr: Cumulative ASR values
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(steps, cumulative_asr, color=COLORS["primary"], linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(steps, cumulative_asr, alpha=0.2, color=COLORS["primary"])
        
        # Add final value annotation
        if cumulative_asr:
            final_asr = cumulative_asr[-1]
            ax.annotate(
                f"Final ASR: {final_asr:.1%}",
                (steps[-1], final_asr),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["accent"], alpha=0.3),
            )
        
        ax.set_xlabel("Attack Attempts", fontsize=12)
        ax.set_ylabel("Cumulative ASR", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_perplexity_comparison(
        self,
        attack_types: List[str],
        pre_attack_perplexity: List[float],
        post_attack_perplexity: List[float],
        title: str = "Perplexity Before vs After Attack",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot perplexity distribution comparison (box plot).
        
        Args:
            attack_types: List of attack type names
            pre_attack_perplexity: Perplexity values before attack
            post_attack_perplexity: Perplexity values after attack
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(attack_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pre_attack_perplexity, width,
                      label="Pre-Attack", color=COLORS["before"], alpha=0.8)
        bars2 = ax.bar(x + width/2, post_attack_perplexity, width,
                      label="Post-Attack", color=COLORS["after"], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(pre_attack_perplexity + post_attack_perplexity) * 0.02,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.set_xlabel("Attack Type", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(attack_types, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_probe_accuracy_by_layer(
        self,
        layers: List[int],
        accuracy_by_model: Dict[str, List[float]],
        title: str = "Probe Accuracy Across Layers",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot probe accuracy curve showing which layer first distinguishes attacks.
        
        Args:
            layers: Layer indices
            accuracy_by_model: Dict mapping model name to accuracy values per layer
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, (model_name, accuracies) in enumerate(accuracy_by_model.items()):
            color = colors[i % len(colors)]
            ax.plot(layers, accuracies, marker="o", label=model_name,
                   color=color, linewidth=2, markersize=6)
            
            # Mark first layer with accuracy > 0.7
            for layer, acc in zip(layers, accuracies):
                if acc > 0.7:
                    ax.scatter([layer], [acc], color=color, s=150, zorder=5, 
                             marker="*", edgecolors="white", linewidths=1)
                    ax.annotate(
                        f"Layer {layer}\n({acc:.1%})",
                        (layer, acc),
                        textcoords="offset points",
                        xytext=(10, 10),
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2),
                    )
                    break
        
        ax.axhline(y=0.7, color=COLORS["neutral"], linestyle="--", alpha=0.5, label="70% threshold")
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Probe Accuracy", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
        return None
    
    def plot_layer_logit_distribution(
        self,
        layers: List[int],
        logit_distributions: Dict[str, Dict[int, List[float]]],
        title: str = "Layer-wise Logit Distribution",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot logit distribution heatmap across layers.
        
        Args:
            layers: Layer indices
            logit_distributions: Dict mapping condition ("clean"/"attack") to layer-wise logit values
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, len(logit_distributions), figsize=(6*len(logit_distributions), 8))
        if len(logit_distributions) == 1:
            axes = [axes]
        
        for idx, (condition, layer_data) in enumerate(logit_distributions.items()):
            ax = axes[idx]
            
            # Build heatmap data
            max_tokens = max(len(vals) for vals in layer_data.values()) if layer_data else 0
            heatmap_data = []
            
            for layer in layers:
                if layer in layer_data:
                    values = layer_data[layer]
                    # Pad or truncate to max_tokens
                    if len(values) < max_tokens:
                        values = list(values) + [0] * (max_tokens - len(values))
                    else:
                        values = values[:max_tokens]
                    heatmap_data.append(values)
                else:
                    heatmap_data.append([0] * max_tokens)
            
            if heatmap_data:
                im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")
                ax.set_xlabel("Token Position", fontsize=11)
                ax.set_ylabel("Layer", fontsize=11)
                ax.set_title(f"{condition.capitalize()} Prompts", fontweight="bold", fontsize=12)
                ax.set_yticks(range(len(layers)))
                ax.set_yticklabels([f"L{l}" for l in layers])
                plt.colorbar(im, ax=ax, label="Logit Value")
        
        fig.suptitle(title, fontweight="bold", fontsize=14, y=0.98)
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        
        return None
    
    def plot_timeseries_asr_comparison(
        self,
        systematic_steps: List[int],
        systematic_asr: List[float],
        random_steps: Optional[List[int]] = None,
        random_asr: Optional[List[float]] = None,
        title: str = "Time-Series ASR: Systematic vs Random",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot time-series ASR comparison between systematic and random methods.
        
        Args:
            systematic_steps: Step numbers for systematic method
            systematic_asr: Cumulative ASR for systematic method
            random_steps: Optional step numbers for random method
            random_asr: Optional cumulative ASR for random method
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Cumulative ASR
        ax1.plot(systematic_steps, systematic_asr, color=COLORS["success"], 
               linewidth=2.5, marker="o", markersize=4, label="Systematic Method")
        ax1.fill_between(systematic_steps, systematic_asr, alpha=0.2, color=COLORS["success"])
        
        if random_steps and random_asr:
            ax1.plot(random_steps, random_asr, color=COLORS["failure"], 
                   linewidth=2.5, marker="s", markersize=4, label="Random Baseline", linestyle="--")
            ax1.fill_between(random_steps, random_asr, alpha=0.2, color=COLORS["failure"])
        
        ax1.set_xlabel("Attack Steps", fontsize=12)
        ax1.set_ylabel("Cumulative ASR", fontsize=12)
        ax1.set_title("Cumulative ASR Over Time", fontweight="bold", fontsize=13)
        ax1.set_ylim(0, 1.1)
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Right: Convergence speed (rolling window ASR)
        if len(systematic_asr) >= 5:
            window_size = min(5, len(systematic_asr) // 2)
            rolling_asr = []
            for i in range(len(systematic_asr)):
                start = max(0, i - window_size + 1)
                rolling_asr.append(np.mean(systematic_asr[start:i+1]))
            
            ax2.plot(systematic_steps, rolling_asr, color=COLORS["primary"], 
                   linewidth=2.5, marker="o", markersize=4, label="Rolling ASR (window=5)")
            ax2.axhline(y=systematic_asr[-1] if systematic_asr else 0, 
                       color=COLORS["success"], linestyle="--", linewidth=2, 
                       label=f"Final ASR: {systematic_asr[-1]*100:.1f}%")
            
            # Mark convergence point (where variance < threshold)
            if len(rolling_asr) >= 3:
                for i in range(3, len(rolling_asr)):
                    window = rolling_asr[i-3:i]
                    if np.std(window) < 0.05:  # Stable within 5%
                        ax2.axvline(x=systematic_steps[i-1], color=COLORS["warning"], 
                                  linestyle=":", linewidth=2, label="Convergence Point")
                        break
            
            ax2.set_xlabel("Attack Steps", fontsize=12)
            ax2.set_ylabel("Rolling ASR", fontsize=12)
            ax2.set_title("Convergence Analysis", fontweight="bold", fontsize=13)
            ax2.set_ylim(0, 1.1)
            ax2.grid(alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Insufficient data\nfor convergence analysis", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Convergence Analysis", fontweight="bold", fontsize=13)
        
        fig.suptitle(title, fontweight="bold", fontsize=14, y=0.98)
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        
        return None