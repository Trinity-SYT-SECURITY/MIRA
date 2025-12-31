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
    
    def plot_universal_signatures_comparison(
        self,
        signature_matrix: SignatureMatrix,
        universal_signatures: Dict[str, Any],
        title: str = "Universal Attack Signatures: Baseline vs Attack",
        save_name: str = "universal_signatures_comparison",
    ) -> Optional[str]:
        """
        Plot comparison of universal signatures between baseline and attack.
        
        Shows that these features appear ONLY during attacks and rarely in baseline.
        
        Args:
            signature_matrix: Computed SignatureMatrix
            universal_signatures: Output from identify_universal_attack_signatures
            title: Chart title
            save_name: Filename for saving
            
        Returns:
            Path to saved chart, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB or not universal_signatures.get("universal_signatures"):
            return None
        
        universal_list = universal_signatures["universal_signatures"]
        if not universal_list:
            return None
        
        # Extract feature data
        features = [s["feature"] for s in universal_list]
        
        # Get baseline and attack values for these features
        def vector_to_array(vector) -> np.ndarray:
            arr = [
                vector.probe_refusal_score,
                vector.probe_acceptance_score,
                vector.attention_entropy,
                vector.attention_max_weight,
                vector.token_entropy,
                vector.top1_top2_gap,
            ]
            for i in range(min(len(vector.layer_activation_norms), 6)):
                arr.append(vector.layer_activation_norms[i] if i < len(vector.layer_activation_norms) else 0.0)
            return np.array(arr)
        
        baseline_array = np.array([vector_to_array(v) for v in signature_matrix.baseline_vectors])
        attack_array = np.array([vector_to_array(v) for v in signature_matrix.attack_vectors])
        
        # Get feature indices
        feature_names = signature_matrix.feature_names
        feature_indices = [feature_names.index(f) for f in features]
        
        # Extract values for universal features
        baseline_values = baseline_array[:, feature_indices]
        attack_values = attack_array[:, feature_indices]
        
        # Compute statistics
        baseline_mean = np.mean(baseline_values, axis=0)
        baseline_std = np.std(baseline_values, axis=0)
        attack_mean = np.mean(attack_values, axis=0)
        attack_std = np.std(attack_values, axis=0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: Bar chart comparing means
        x_pos = np.arange(len(features))
        width = 0.35
        
        bars1 = axes[0].bar(x_pos - width/2, baseline_mean, width, 
                           yerr=baseline_std, label="Baseline (Normal Prompts)", 
                           color="steelblue", alpha=0.7, capsize=5)
        bars2 = axes[0].bar(x_pos + width/2, attack_mean, width,
                           yerr=attack_std, label="Attack Prompts",
                           color="coral", alpha=0.7, capsize=5)
        
        axes[0].set_xlabel("Universal Attack Signatures", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Feature Value", fontsize=12)
        axes[0].set_title("Feature Values: Baseline vs Attack\n(These features appear ONLY during attacks)", 
                         fontsize=13, fontweight="bold")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f.replace("_", " ").title() for f in features], rotation=45, ha="right")
        axes[0].legend(fontsize=11)
        axes[0].grid(axis="y", alpha=0.3)
        
        # Add false positive rate annotations
        for i, sig in enumerate(universal_list):
            fp_rate = sig.get("false_positive_rate", 0.0)
            axes[0].text(i, max(baseline_mean[i], attack_mean[i]) + 0.1,
                        f"FP: {fp_rate:.1%}", ha="center", fontsize=9, 
                        color="red" if fp_rate > 0.1 else "green", fontweight="bold")
        
        # Bottom: Distribution comparison (box plot)
        data_to_plot = []
        labels = []
        for i, feat in enumerate(features):
            data_to_plot.append(baseline_values[:, i])
            labels.append(f"{feat.replace('_', ' ').title()}\n(Baseline)")
            data_to_plot.append(attack_values[:, i])
            labels.append(f"{feat.replace('_', ' ').title()}\n(Attack)")
        
        bp = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True, 
                            widths=0.6, showmeans=True)
        
        # Color code: blue for baseline, red for attack
        colors = ["steelblue" if i % 2 == 0 else "coral" for i in range(len(bp["boxes"]))]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel("Feature Value Distribution", fontsize=12)
        axes[1].set_title("Distribution Comparison: Baseline vs Attack\n(Shows separation between normal and attack)", 
                         fontsize=13, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].tick_params(axis="x", rotation=45)
        
        plt.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        
        path = self.output_dir / f"{save_name}.png"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        
        return str(path)
    
    def plot_universal_signatures_detection_accuracy(
        self,
        signature_matrix: SignatureMatrix,
        universal_signatures: Dict[str, Any],
        title: str = "Universal Signatures Detection Accuracy",
        save_name: str = "universal_signatures_detection",
    ) -> Optional[str]:
        """
        Plot detection accuracy metrics for universal signatures.
        
        Shows true positive rate (in attacks) vs false positive rate (in baseline).
        
        Args:
            signature_matrix: Computed SignatureMatrix
            universal_signatures: Output from identify_universal_attack_signatures
            title: Chart title
            save_name: Filename for saving
            
        Returns:
            Path to saved chart, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB or not universal_signatures.get("universal_signatures"):
            return None
        
        universal_list = universal_signatures["universal_signatures"]
        if not universal_list:
            return None
        
        features = [s["feature"] for s in universal_list]
        cross_attack_stability = [s["cross_attack_stability"] for s in universal_list]
        false_positive_rates = [s["false_positive_rate"] for s in universal_list]
        
        # Create ROC-like plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot: False Positive Rate (x) vs True Positive Rate (y)
        scatter = ax.scatter(false_positive_rates, cross_attack_stability, 
                           s=200, alpha=0.7, c=range(len(features)), 
                           cmap="RdYlGn_r", edgecolors="black", linewidths=1.5)
        
        # Add feature labels
        for i, feat in enumerate(features):
            ax.annotate(feat.replace("_", " ").title()[:20], 
                       (false_positive_rates[i], cross_attack_stability[i]),
                       xytext=(5, 5), textcoords="offset points", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Add ideal region (low FP, high TP)
        ax.axhspan(0.8, 1.0, xmin=0, xmax=0.1, alpha=0.2, color="green", 
                  label="Ideal Region (High TP, Low FP)")
        ax.axvspan(0, 0.1, ymin=0.8, ymax=1.0, alpha=0.2, color="green")
        
        ax.set_xlabel("False Positive Rate (in Baseline)", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Cross-Attack Stability)", fontsize=12, fontweight="bold")
        ax.set_title(title + "\n(Features in top-right are best detectors)", 
                    fontsize=13, fontweight="bold")
        ax.set_xlim(-0.05, max(0.2, max(false_positive_rates) * 1.2))
        ax.set_ylim(0.7, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Classifier")
        
        plt.tight_layout()
        
        path = self.output_dir / f"{save_name}.png"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        
        return str(path)

