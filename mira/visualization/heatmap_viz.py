"""
Heatmap Visualization - Visualize baseline vs attack feature differences

Creates publication-quality heatmaps for:
- Layer-wise activation differences
- Attack success feature patterns
- Feature consistency analysis

Reference: CGC.md research methodology
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def plot_layer_feature_heatmap(
    delta_data: Dict[str, Any],
    title: str = "Attack vs Baseline Feature Delta",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot heatmap of layer-wise feature differences.
    
    Rows = Layers (0 to N)
    Columns = Features (residual, attention entropy, head dominance)
    Color = Normalized delta (red = increase, blue = decrease)
    
    Args:
        delta_data: Dict with 'layer_deltas' list
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if not delta_data or 'layer_deltas' not in delta_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    layer_deltas = delta_data['layer_deltas']
    num_layers = len(layer_deltas)
    
    # Prepare data matrix
    features = ['Residual Norm Δ', 'Attn Entropy Δ', 'Head Dominance Δ']
    data = np.zeros((num_layers, len(features)))
    
    for i, ld in enumerate(layer_deltas):
        data[i, 0] = ld.get('residual_delta', 0)
        data[i, 1] = ld.get('attn_entropy_delta', 0)
        data[i, 2] = ld.get('head_dominance_delta', 0)
    
    # Normalize for visualization (z-score per column)
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.std(col) > 0:
            data[:, j] = (col - np.mean(col)) / np.std(col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap: blue (decrease) -> white (no change) -> red (increase)
    cmap = plt.cm.RdBu_r
    
    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
    
    # Labels
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, fontsize=11, fontweight='bold')
    ax.set_yticks(range(num_layers))
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Z-Score Delta')
    cbar.ax.tick_params(labelsize=10)
    
    # Add value annotations
    for i in range(num_layers):
        for j in range(len(features)):
            val = data[i, j]
            color = 'white' if abs(val) > 1 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=8, color=color)
    
    # Grid
    ax.set_xticks(np.arange(len(features)+1)-.5, minor=True)
    ax.set_yticks(np.arange(num_layers+1)-.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attack_comparison_heatmap(
    baseline_features: Dict[str, np.ndarray],
    failed_attack_features: Dict[str, np.ndarray],
    success_attack_features: Dict[str, np.ndarray],
    feature_name: str = "residual_norms",
    title: str = "Baseline vs Attack Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison heatmap: Baseline | Failed Attack | Successful Attack
    
    Args:
        baseline_features: Features from clean prompts
        failed_attack_features: Features from failed attacks
        success_attack_features: Features from successful attacks
        feature_name: Which feature to plot
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    datasets = [
        (baseline_features, 'Baseline', 'Greens'),
        (failed_attack_features, 'Failed Attack', 'Blues'),
        (success_attack_features, 'Successful Attack', 'Reds'),
    ]
    
    vmin, vmax = None, None
    
    # Find global min/max for consistent colormap
    for data, _, _ in datasets:
        if feature_name in data:
            arr = data[feature_name]
            if vmin is None:
                vmin, vmax = arr.min(), arr.max()
            else:
                vmin = min(vmin, arr.min())
                vmax = max(vmax, arr.max())
    
    for ax, (data, label, cmap) in zip(axes, datasets):
        if feature_name not in data or len(data[feature_name]) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_title(label, fontsize=12, fontweight='bold')
            continue
        
        arr = data[feature_name]
        
        # Average across samples if needed
        if arr.ndim == 2:
            arr_mean = arr.mean(axis=0).reshape(-1, 1)
        else:
            arr_mean = arr.reshape(-1, 1)
        
        im = ax.imshow(arr_mean, cmap=cmap, aspect='auto', 
                       vmin=vmin, vmax=vmax if vmax else 1)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Avg', fontsize=10)
        
        if ax == axes[0]:
            ax.set_ylabel('Layer', fontsize=11)
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=feature_name.replace('_', ' ').title())
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_consistency_map(
    success_patterns: List[Dict[str, Any]],
    title: str = "Feature Consistency in Successful Attacks",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature activation frequency in successful attacks.
    
    Shows which features consistently appear when attacks succeed.
    
    Args:
        success_patterns: List of delta dicts from successful attacks
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if not success_patterns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No successful attacks to analyze', 
               ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Count how often each layer-feature combination shows significant change
    num_layers = max(len(p.get('layer_deltas', [])) for p in success_patterns)
    features = ['residual_delta', 'attn_entropy_delta', 'head_dominance_delta']
    feature_labels = ['Residual Δ', 'Attn Entropy Δ', 'Head Dom Δ']
    
    # Count significant changes (|z| > 1)
    freq_matrix = np.zeros((num_layers, len(features)))
    
    for pattern in success_patterns:
        for i, ld in enumerate(pattern.get('layer_deltas', [])):
            for j, feat in enumerate(features):
                val = ld.get(feat, 0)
                if abs(val) > 0.5:  # Significant threshold
                    freq_matrix[i, j] += 1
    
    # Normalize to percentage
    freq_matrix = freq_matrix / len(success_patterns) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(freq_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, label='Activation Frequency (%)')
    
    # Add annotations
    for i in range(num_layers):
        for j in range(len(features)):
            val = freq_matrix[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=9, color=color)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_tiered_asr_chart(
    rbr: float,
    nrgr: float,
    scr: float,
    model_name: str = "Model",
    attack_type: str = "GCG",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot three-tier ASR as stacked bar chart.
    
    Args:
        rbr: Refusal Bypass Rate (%)
        nrgr: Non-Refusal Generation Rate (%)
        scr: Semantic Compliance Rate (%)
        model_name: Name of target model
        attack_type: Type of attack
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Data
    metrics = ['L1: RBR\n(Refusal Bypass)', 
               'L2: NRGR\n(Generation Stable)', 
               'L3: SCR\n(Semantic Comply)']
    values = [rbr, nrgr, scr]
    colors = ['#ef4444', '#f59e0b', '#22c55e']  # Red, Amber, Green
    
    # Horizontal bar chart
    bars = ax.barh(metrics, values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 110)
    ax.set_xlabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Three-Tier ASR: {model_name} under {attack_type} Attack',
                fontsize=14, fontweight='bold', pad=15)
    
    # Add explanation box
    explanation = (
        "L1 RBR: No refusal detected (safety bypass)\n"
        "L2 NRGR: Coherent output generated\n"
        "L3 SCR: Semantically compliant response"
    )
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes,
           fontsize=9, va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_comprehensive_heatmap_report(
    results_dir: str,
    baseline_clean: Optional[Dict] = None,
    attack_failed: Optional[Dict] = None,
    attack_success: Optional[Dict] = None,
    tiered_asr: Optional[Dict] = None
) -> List[str]:
    """
    Generate all heatmap visualizations and save to results directory.
    
    Args:
        results_dir: Directory to save charts
        baseline_clean: Clean prompt features
        attack_failed: Failed attack features
        attack_success: Successful attack features
        tiered_asr: Tiered ASR results
        
    Returns:
        List of saved file paths
    """
    results_path = Path(results_dir)
    charts_dir = results_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # 1. Tiered ASR chart
    if tiered_asr:
        fig = plot_tiered_asr_chart(
            rbr=tiered_asr.get('rbr', 0),
            nrgr=tiered_asr.get('nrgr', 0),
            scr=tiered_asr.get('scr', 0),
            save_path=str(charts_dir / "tiered_asr.png")
        )
        plt.close(fig)
        saved_files.append("tiered_asr.png")
    
    # 2. Feature comparison heatmap
    if baseline_clean and (attack_failed or attack_success):
        fig = plot_attack_comparison_heatmap(
            baseline_features=baseline_clean or {},
            failed_attack_features=attack_failed or {},
            success_attack_features=attack_success or {},
            save_path=str(charts_dir / "feature_comparison.png")
        )
        plt.close(fig)
        saved_files.append("feature_comparison.png")
    
    return saved_files
