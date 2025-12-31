"""
Attack Signature Matrix Visualization.

Generates heatmaps and visualizations for the Attack Signature Matrix,
showing feature differences between baseline and attack prompts.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from mira.analysis.signature_matrix import SignatureMatrix


class SignatureVisualizer:
    """
    Visualizer for Attack Signature Matrix.
    
    Generates heatmaps showing feature differences between
    baseline and attack conditions.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize signature visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['savefig.dpi'] = 300
    
    def plot_signature_heatmap(
        self,
        signature_matrix: SignatureMatrix,
        title: str = "Attack Signature Matrix",
        save_name: str = "signature_matrix",
        normalize: bool = True,
    ) -> Optional[str]:
        """
        Plot Attack Signature Matrix as a heatmap.
        
        Args:
            signature_matrix: Computed SignatureMatrix
            title: Chart title
            save_name: Filename for saving
            normalize: Whether to normalize features (z-score)
            
        Returns:
            Path to saved chart, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Prepare data
        feature_names = signature_matrix.feature_names
        
        # Build matrix: rows = prompts, columns = features
        baseline_data = []
        attack_data = []
        
        for vector in signature_matrix.baseline_vectors:
            row = [
                vector.probe_refusal_score,
                vector.probe_acceptance_score,
                vector.attention_entropy,
                vector.attention_max_weight,
                vector.token_entropy,
                vector.top1_top2_gap,
            ]
            # Add layer norms
            for i in range(min(len(vector.layer_activation_norms), 6)):
                row.append(vector.layer_activation_norms[i] if i < len(vector.layer_activation_norms) else 0.0)
            baseline_data.append(row)
        
        for vector in signature_matrix.attack_vectors:
            row = [
                vector.probe_refusal_score,
                vector.probe_acceptance_score,
                vector.attention_entropy,
                vector.attention_max_weight,
                vector.token_entropy,
                vector.top1_top2_gap,
            ]
            # Add layer norms
            for i in range(min(len(vector.layer_activation_norms), 6)):
                row.append(vector.layer_activation_norms[i] if i < len(vector.layer_activation_norms) else 0.0)
            attack_data.append(row)
        
        baseline_array = np.array(baseline_data)
        attack_array = np.array(attack_data)
        
        # Compute differential (attack - baseline mean)
        baseline_mean = np.mean(baseline_array, axis=0)
        differential = attack_array - baseline_mean[None, :]
        
        # Normalize if requested
        if normalize:
            baseline_std = np.std(baseline_array, axis=0) + 1e-9
            differential = differential / baseline_std[None, :]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(attack_data) * 0.3)))
        
        # Left: Baseline heatmap
        sns.heatmap(
            baseline_array,
            xticklabels=feature_names,
            yticklabels=[f"Baseline {i+1}" for i in range(len(baseline_data))],
            cmap="Blues",
            center=0,
            cbar_kws={"label": "Feature Value"},
            ax=axes[0],
            fmt=".2f",
        )
        axes[0].set_title("Baseline Feature Distribution", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Features", fontsize=12)
        axes[0].set_ylabel("Prompts", fontsize=12)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")
        
        # Right: Differential heatmap (attack - baseline)
        sns.heatmap(
            differential,
            xticklabels=feature_names,
            yticklabels=[f"Attack {i+1}" for i in range(len(attack_data))],
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Δ (Attack - Baseline)"},
            ax=axes[1],
            fmt=".2f",
        )
        axes[1].set_title("Attack Signature (Δ from Baseline)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Features", fontsize=12)
        axes[1].set_ylabel("Attack Prompts", fontsize=12)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
        
        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        # Save
        path = self.output_dir / f"{save_name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        return str(path)
    
    def plot_feature_stability(
        self,
        signature_matrix: SignatureMatrix,
        title: str = "Feature Stability Analysis",
        save_name: str = "feature_stability",
    ) -> Optional[str]:
        """
        Plot feature stability and discriminative power.
        
        Args:
            signature_matrix: Computed SignatureMatrix
            title: Chart title
            save_name: Filename for saving
            
        Returns:
            Path to saved chart, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB or not signature_matrix.feature_stability:
            return None
        
        features = list(signature_matrix.feature_stability.keys())
        stability = [signature_matrix.feature_stability[f] for f in features]
        discriminative = [
            signature_matrix.feature_discriminative_power.get(f, 0.0)
            for f in features
        ]
        z_scores = [
            abs(signature_matrix.z_scores[i]) if signature_matrix.z_scores is not None else 0.0
            for i in range(len(features))
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Stability
        axes[0].barh(features, stability, color="steelblue")
        axes[0].set_xlabel("Stability Score", fontsize=12)
        axes[0].set_title("Feature Stability\n(Frequency in Attacks)", fontsize=12, fontweight="bold")
        axes[0].axvline(x=0.7, color="red", linestyle="--", alpha=0.5, label="Threshold (0.7)")
        axes[0].legend()
        
        # Z-scores
        colors = ["red" if z > 1.5 else "orange" if z > 1.0 else "gray" for z in z_scores]
        axes[1].barh(features, z_scores, color=colors)
        axes[1].set_xlabel("|Z-Score|", fontsize=12)
        axes[1].set_title("Feature Z-Score\n(Deviation from Baseline)", fontsize=12, fontweight="bold")
        axes[1].axvline(x=1.5, color="red", linestyle="--", alpha=0.5, label="Threshold (1.5)")
        axes[1].legend()
        
        # Discriminative power
        axes[2].barh(features, discriminative, color="green")
        axes[2].set_xlabel("Discriminative Power", fontsize=12)
        axes[2].set_title("Success vs Failure\nSeparation", fontsize=12, fontweight="bold")
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        path = self.output_dir / f"{save_name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        return str(path)
    
    def plot_stable_signatures(
        self,
        stable_signatures: Dict[str, Any],
        title: str = "Stable Attack Signatures",
        save_name: str = "stable_signatures",
    ) -> Optional[str]:
        """
        Plot identified stable attack signatures.
        
        Args:
            stable_signatures: Output from identify_stable_signatures
            title: Chart title
            save_name: Filename for saving
            
        Returns:
            Path to saved chart, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB or not stable_signatures.get("stable_signatures"):
            return None
        
        sigs = stable_signatures["stable_signatures"]
        if not sigs:
            return None
        
        features = [s["feature"] for s in sigs]
        stability = [s["stability"] for s in sigs]
        z_scores = [abs(s["z_score"]) for s in sigs]
        combined = [s["stability"] * abs(s["z_score"]) for s in sigs]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
        
        y_pos = np.arange(len(features))
        width = 0.35
        
        bars1 = ax.barh(y_pos - width/2, stability, width, label="Stability", color="steelblue")
        bars2 = ax.barh(y_pos + width/2, z_scores, width, label="|Z-Score|", color="coral")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        
        # Add combined score as text
        for i, (y, comb) in enumerate(zip(y_pos, combined)):
            ax.text(comb + 0.05, y, f"{comb:.2f}", va="center", fontsize=9)
        
        plt.tight_layout()
        
        path = self.output_dir / f"{save_name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        return str(path)

