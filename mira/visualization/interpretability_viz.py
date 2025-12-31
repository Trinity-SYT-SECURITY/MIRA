"""
Interpretability Visualization Module.

Generates explainable visualizations for attack analysis:
- Logit change heatmaps (success vs failure)
- SSR steering vector visualizations
- Attack path diagrams
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


class InterpretabilityVisualizer:
    """Generate interpretability visualizations for attack analysis."""
    
    def __init__(self, output_dir: str = "./results/interpretability"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_logit_change_heatmap(
        self,
        layers: List[int],
        clean_logits: Dict[int, List[float]],
        attack_logits: Dict[int, List[float]],
        top_k: int = 20,
        title: str = "Logit Change Heatmap (Attack vs Clean)",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot heatmap showing logit changes between clean and attack conditions.
        
        Args:
            layers: Layer indices
            clean_logits: Dict mapping layer_idx to list of logit values (top-k tokens)
            attack_logits: Dict mapping layer_idx to list of logit values (top-k tokens)
            top_k: Number of top tokens to display
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Compute logit differences
        logit_diffs = {}
        for layer_idx in layers:
            if layer_idx in clean_logits and layer_idx in attack_logits:
                clean = np.array(clean_logits[layer_idx][:top_k])
                attack = np.array(attack_logits[layer_idx][:top_k])
                
                # Pad or truncate to same length
                max_len = max(len(clean), len(attack))
                if len(clean) < max_len:
                    clean = np.pad(clean, (0, max_len - len(clean)), constant_values=0)
                if len(attack) < max_len:
                    attack = np.pad(attack, (0, max_len - len(attack)), constant_values=0)
                
                diff = attack - clean
                logit_diffs[layer_idx] = diff[:top_k]
        
        if not logit_diffs:
            return None
        
        # Build heatmap data
        heatmap_data = []
        layer_labels = []
        for layer_idx in layers:
            if layer_idx in logit_diffs:
                heatmap_data.append(logit_diffs[layer_idx])
                layer_labels.append(f"L{layer_idx}")
        
        if not heatmap_data:
            return None
        
        # Normalize to same length
        max_len = max(len(row) for row in heatmap_data)
        heatmap_data = [np.pad(row, (0, max_len - len(row)), constant_values=0) if len(row) < max_len else row[:max_len] 
                       for row in heatmap_data]
        
        heatmap_array = np.array(heatmap_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use diverging colormap (blue-white-red)
        cmap = plt.cm.RdBu_r
        vmax = np.abs(heatmap_array).max()
        vmin = -vmax
        
        im = ax.imshow(heatmap_array, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set labels
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels)
        ax.set_xlabel("Token Rank (Top-K)", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="Logit Change (Attack - Clean)")
        cbar.ax.axhline(y=0, color='black', linewidth=1)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, max_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(layer_labels), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        
        return None
    
    def plot_ssr_steering_vectors(
        self,
        refusal_directions: Dict[int, np.ndarray],
        acceptance_directions: Optional[Dict[int, np.ndarray]] = None,
        title: str = "SSR Steering Vectors",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Visualize SSR steering vectors using PCA projection.
        
        Args:
            refusal_directions: Dict mapping layer_idx to refusal direction vector
            acceptance_directions: Optional dict mapping layer_idx to acceptance direction vector
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return None
        
        if not refusal_directions:
            return None
        
        # Collect all vectors
        all_vectors = []
        layer_indices = []
        vector_types = []
        
        for layer_idx, vec in refusal_directions.items():
            if vec is not None and len(vec) > 0:
                all_vectors.append(vec.flatten() if vec.ndim > 1 else vec)
                layer_indices.append(layer_idx)
                vector_types.append("refusal")
        
        if acceptance_directions:
            for layer_idx, vec in acceptance_directions.items():
                if vec is not None and len(vec) > 0:
                    all_vectors.append(vec.flatten() if vec.ndim > 1 else vec)
                    layer_indices.append(layer_idx)
                    vector_types.append("acceptance")
        
        if not all_vectors:
            return None
        
        # Normalize to same dimension
        max_dim = max(len(v) for v in all_vectors)
        all_vectors = [np.pad(v, (0, max_dim - len(v)), constant_values=0) if len(v) < max_dim else v[:max_dim]
                      for v in all_vectors]
        
        vectors_array = np.array(all_vectors)
        
        # Apply PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors_array)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot refusal directions
        refusal_mask = np.array(vector_types) == "refusal"
        refusal_2d = vectors_2d[refusal_mask]
        refusal_layers = [layer_indices[i] for i in range(len(layer_indices)) if refusal_mask[i]]
        
        if len(refusal_2d) > 0:
            scatter1 = ax.scatter(
                refusal_2d[:, 0], refusal_2d[:, 1],
                c='#ef4444', s=150, alpha=0.7, marker='o',
                label='Refusal Direction', edgecolors='black', linewidths=1.5
            )
            
            # Annotate layers
            for i, (x, y) in enumerate(refusal_2d):
                ax.annotate(
                    f"L{refusal_layers[i]}",
                    (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#ef4444'
                )
        
        # Plot acceptance directions
        if acceptance_directions:
            acceptance_mask = np.array(vector_types) == "acceptance"
            acceptance_2d = vectors_2d[acceptance_mask]
            acceptance_layers = [layer_indices[i] for i in range(len(layer_indices)) if acceptance_mask[i]]
            
            if len(acceptance_2d) > 0:
                scatter2 = ax.scatter(
                    acceptance_2d[:, 0], acceptance_2d[:, 1],
                    c='#22c55e', s=150, alpha=0.7, marker='s',
                    label='Acceptance Direction', edgecolors='black', linewidths=1.5
                )
                
                # Annotate layers
                for i, (x, y) in enumerate(acceptance_2d):
                    ax.annotate(
                        f"L{acceptance_layers[i]}",
                        (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#22c55e'
                    )
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        
        return None
    
    def plot_attack_path_diagram(
        self,
        layers: List[int],
        clean_activations: List[float],
        attack_activations: List[float],
        refusal_scores: Optional[List[float]] = None,
        acceptance_scores: Optional[List[float]] = None,
        title: str = "Attack Path Diagram",
        save_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create diagram showing attack path through transformer layers.
        
        Args:
            layers: Layer indices
            clean_activations: Activation values for clean prompt
            attack_activations: Activation values for attack prompt
            refusal_scores: Optional refusal scores per layer
            acceptance_scores: Optional acceptance scores per layer
            title: Chart title
            save_name: Filename
            
        Returns:
            Path to saved chart
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if len(layers) != len(clean_activations) or len(layers) != len(attack_activations):
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Activation trajectory
        ax1.plot(layers, clean_activations, 'o-', color='#3498db', linewidth=2.5, 
                markersize=8, label='Clean', alpha=0.8)
        ax1.plot(layers, attack_activations, 's-', color='#e74c3c', linewidth=2.5,
                markersize=8, label='Attack', alpha=0.8)
        
        # Fill between
        ax1.fill_between(layers, clean_activations, attack_activations, 
                        alpha=0.2, color='#e74c3c')
        
        ax1.set_xlabel("Layer", fontsize=12)
        ax1.set_ylabel("Activation Norm", fontsize=12)
        ax1.set_title("Activation Trajectory", fontweight="bold", fontsize=13)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Right: Refusal/Acceptance scores
        if refusal_scores and acceptance_scores:
            x = np.arange(len(layers))
            width = 0.35
            
            ax2.bar(x - width/2, refusal_scores, width, label='Refusal Score', 
                   color='#ef4444', alpha=0.7)
            ax2.bar(x + width/2, acceptance_scores, width, label='Acceptance Score',
                   color='#22c55e', alpha=0.7)
            
            ax2.set_xlabel("Layer", fontsize=12)
            ax2.set_ylabel("Score", fontsize=12)
            ax2.set_title("Refusal vs Acceptance Scores", fontweight="bold", fontsize=13)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"L{l}" for l in layers])
            ax2.legend(loc='best', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            # Show activation difference instead
            diff = np.array(attack_activations) - np.array(clean_activations)
            colors = ['#ef4444' if d > 0 else '#22c55e' for d in diff]
            ax2.bar(range(len(layers)), diff, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel("Layer", fontsize=12)
            ax2.set_ylabel("Activation Change (Attack - Clean)", fontsize=12)
            ax2.set_title("Activation Change by Layer", fontweight="bold", fontsize=13)
            ax2.set_xticks(range(len(layers)))
            ax2.set_xticklabels([f"L{l}" for l in layers])
            ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontweight="bold", fontsize=15, y=0.98)
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        
        return None

