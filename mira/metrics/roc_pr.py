"""
ROC and Precision-Recall Curve Generation for Attack Success Prediction.

Evaluates attack success prediction based on internal model signals
such as probe scores, entropy metrics, and feature activations.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class AttackSuccessCurves:
    """
    Generate ROC and PR curves for predicting attack success.
    
    Predicts whether an attack will succeed based on internal signals:
    - Feature activation patterns
    - Probe refusal/acceptance scores
    - Entropy and uncertainty metrics
    - Distance to safety subspaces
    """
    
    def __init__(self):
        self.curves_data = {}
    
    def compute_roc_curve(
        self,
        true_labels: List[bool],
        prediction_scores: List[float],
        name: str = "default"
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data.
        
        Args:
            true_labels: Ground truth (True = attack succeeded)
            prediction_scores: Predicted scores (higher = more likely success)
            name: Identifier for this ROC curve
            
        Returns:
            Dict containing:
                fpr: False positive rates
                tpr: True positive rates
                thresholds: Decision thresholds
                auc: Area under curve
        """
        # Convert to numpy arrays
        y_true = np.array(true_labels, dtype=int)
        y_scores = np.array(prediction_scores, dtype=float)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        result = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc),
            "name": name
        }
        
        self.curves_data[f"roc_{name}"] = result
        return result
    
    def compute_pr_curve(
        self,
        true_labels: List[bool],
        prediction_scores: List[float],
        name: str = "default"
    ) -> Dict[str, Any]:
        """
        Compute Precision-Recall curve.
        
        Args:
            true_labels: Ground truth  
            prediction_scores: Predicted scores
            name: Identifier for this PR curve
            
        Returns:
            Dict containing:
                precision: Precision values
                recall: Recall values
                thresholds: Decision thresholds
                auc: Area under curve
        """
        y_true = np.array(true_labels, dtype=int)
        y_scores = np.array(prediction_scores, dtype=float)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        result = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(pr_auc),
            "name": name
        }
        
        self.curves_data[f"pr_{name}"] = result
        return result
    
    def plot_roc_curves(
        self,
        curves: List[Dict[str, Any]],
        output_path: str,
        title: str = "ROC Curve - Attack Success Prediction"
    ):
        """
        Plot ROC curves for multiple predictors.
        
        Args:
            curves: List of ROC curve data dicts
            output_path: Path to save plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
        
        for curve_data, color in zip(curves, colors):
            fpr = np.array(curve_data["fpr"])
            tpr = np.array(curve_data["tpr"])
            roc_auc = curve_data["auc"]
            name = curve_data["name"]
            
            ax.plot(
                fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})'
            )
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curves(
        self,
        curves: List[Dict[str, Any]],
        output_path: str,
        title: str = "Precision-Recall Curve - Attack Success Prediction"
    ):
        """Plot Precision-Recall curves."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
        
        for curve_data, color in zip(curves, colors):
            precision = np.array(curve_data["precision"])
            recall = np.array(curve_data["recall"])
            pr_auc = curve_data["auc"]
            name = curve_data["name"]
            
            ax.plot(
                recall, precision, color=color, lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})'
            )
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combined_roc_pr(
        self,
        roc_curves: List[Dict[str, Any]],
        pr_curves: List[Dict[str, Any]],
        output_path: str,
        title: str = "Attack Success Prediction Performance"
    ):
        """Plot ROC and PR curves side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))
        
        # ROC curves
        for curve_data, color in zip(roc_curves, colors):
            fpr = np.array(curve_data["fpr"])
            tpr = np.array(curve_data["tpr"])
            roc_auc = curve_data["auc"]
            name = curve_data["name"]
            
            ax1.plot(
                fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})'
            )
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=9)
        ax1.grid(alpha=0.3)
        
        # PR curves
        for curve_data, color in zip(pr_curves, colors):
            precision = np.array(curve_data["precision"])
            recall = np.array(curve_data["recall"])
            pr_auc = curve_data["auc"]
            name = curve_data["name"]
            
            ax2.plot(
                recall, precision, color=color, lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})'
            )
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def predict_attack_success_from_features(
        self,
        feature_vector: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Predict attack success probability from internal features.
        
        Args:
            feature_vector: Dict of feature names to values
            weights: Optional feature weights (defaults to equal weights)
            
        Returns:
            Predicted success probability (0-1)
        """
        if weights is None:
            # Default weights based on feature importance
            weights = {
                "probe_refusal_score": -1.5,  # Lower refusal = higher success
                "probe_acceptance_score": 2.0,  # Higher acceptance = higher success
                "token_entropy": 0.8,  # Higher entropy often indicates confusion
                "attention_entropy": 0.5,
                "distance_to_acceptance_subspace": -1.0,  # Lower distance = higher success
            }
        
        # Compute weighted score
        score = 0.0
        total_weight = 0.0
        
        for feature_name, feature_value in feature_vector.items():
            if feature_name in weights:
                score += weights[feature_name] * feature_value
                total_weight += abs(weights[feature_name])
        
        # Normalize and apply sigmoid
        if total_weight > 0:
            score = score / total_weight
        
        # Sigmoid to convert to probability
        probability = 1.0 / (1.0 + np.exp(-score))
        
        return float(probability)


def compute_attack_success_curves(
    attack_results: List[Dict[str, Any]],
    output_dir: str,
    predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute ROC/PR curves for attack success prediction.
    
    Args:
        attack_results: List of attack result dicts with:
            - success: bool (ground truth)
            - probe_refusal_score: float
            - probe_acceptance_score: float
            - token_entropy: float
            - Other feature values
        output_dir: Directory to save plots
        predictors: List of feature names to use as predictors
        
    Returns:
        Dict with curve data and plot paths
    """
    if predictors is None:
        predictors = [
            "probe_acceptance_score",
            "token_entropy",
            "combined_features"
        ]
    
    generator = AttackSuccessCurves()
    results = {}
    
    # Extract ground truth
    true_labels = [r.get("success", False) for r in attack_results]
    
    # Compute curves for each predictor
    roc_curves = []
    pr_curves = []
    
    for predictor in predictors:
        if predictor == "combined_features":
            # Use weighted combination
            scores = [
                generator.predict_attack_success_from_features({
                    k: v for k, v in r.items()
                    if k not in ["success", "prompt", "response"]
                })
                for r in attack_results
            ]
        else:
            # Use single feature
            scores = [r.get(predictor, 0.0) for r in attack_results]
        
        # Compute curves
        roc_data = generator.compute_roc_curve(true_labels, scores, name=predictor)
        pr_data = generator.compute_pr_curve(true_labels, scores, name=predictor)
        
        roc_curves.append(roc_data)
        pr_curves.append(pr_data)
    
    # Generate plots
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    roc_path = os.path.join(output_dir, "attack_success_roc.png")
    pr_path = os.path.join(output_dir, "attack_success_pr.png")
    combined_path = os.path.join(output_dir, "attack_success_curves.png")
    
    generator.plot_roc_curves(roc_curves, roc_path)
    generator.plot_pr_curves(pr_curves, pr_path)
    generator.plot_combined_roc_pr(roc_curves, pr_curves, combined_path)
    
    results["roc_curves"] = roc_curves
    results["pr_curves"] = pr_curves
    results["plot_paths"] = {
        "roc": roc_path,
        "pr": pr_path,
        "combined": combined_path
    }
    
    return results
