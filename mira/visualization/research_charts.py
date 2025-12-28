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
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path)
            plt.close()
            return str(path)
        
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
