"""
Attention pattern visualization.

Provides tools for visualizing:
- Attention heatmaps
- Attention pattern comparisons
- Safety head highlighting
"""

from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def plot_attention_heatmap(
    attention_pattern: torch.Tensor,
    tokens: Optional[List[str]] = None,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create heatmap visualization of attention pattern.
    
    Args:
        attention_pattern: Attention weights (seq_len, seq_len)
        tokens: Token labels for axes
        layer_idx: Layer index for title
        head_idx: Head index for title
        title: Custom title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    pattern = attention_pattern.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'attention',
        ['#ffffff', '#3498db', '#2c3e50']
    )
    
    im = ax.imshow(pattern, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Token labels
    if tokens:
        # Truncate tokens for display
        display_tokens = [t[:10] + '..' if len(t) > 10 else t for t in tokens]
        
        if len(display_tokens) <= 50:  # Only show labels if not too many
            ax.set_xticks(range(len(display_tokens)))
            ax.set_yticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(display_tokens, fontsize=8)
    
    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif layer_idx is not None and head_idx is not None:
        ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_comparison(
    pattern_before: torch.Tensor,
    pattern_after: torch.Tensor,
    tokens_before: Optional[List[str]] = None,
    tokens_after: Optional[List[str]] = None,
    titles: Tuple[str, str] = ("Before Attack", "After Attack"),
    figsize: Tuple[int, int] = (20, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare attention patterns before and after an attack.
    
    Args:
        pattern_before: Attention before attack
        pattern_after: Attention after attack
        tokens_before: Token labels for before
        tokens_after: Token labels for after
        titles: Subplot titles
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    p_before = pattern_before.detach().cpu().numpy()
    p_after = pattern_after.detach().cpu().numpy()
    
    cmap = LinearSegmentedColormap.from_list(
        'attention',
        ['#ffffff', '#3498db', '#2c3e50']
    )
    
    # Before
    im1 = axes[0].imshow(p_before, cmap=cmap, aspect='auto')
    axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # After
    im2 = axes[1].imshow(p_after, cmap=cmap, aspect='auto')
    axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Difference
    # Pad if different sizes
    min_size = min(p_before.shape[0], p_after.shape[0])
    diff = p_after[:min_size, :min_size] - p_before[:min_size, :min_size]
    
    cmap_diff = plt.cm.RdBu_r
    im3 = axes[2].imshow(diff, cmap=cmap_diff, aspect='auto', vmin=-1, vmax=1)
    axes[2].set_title('Difference (After - Before)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_head_importance(
    head_scores: Dict[Tuple[int, int], float],
    n_layers: int,
    n_heads: int,
    title: str = "Attention Head Importance",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize importance scores across all attention heads.
    
    Args:
        head_scores: Dictionary mapping (layer, head) to importance score
        n_layers: Number of layers
        n_heads: Number of heads per layer
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    # Build matrix
    scores = np.zeros((n_layers, n_heads))
    for (layer, head), score in head_scores.items():
        if layer < n_layers and head < n_heads:
            scores[layer, head] = score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(scores, cmap='viridis', aspect='auto')
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    
    plt.colorbar(im, ax=ax, label='Importance Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_entropy(
    entropy_by_layer: Dict[int, float],
    title: str = "Attention Entropy by Layer",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot attention entropy across layers.
    
    Args:
        entropy_by_layer: Dictionary mapping layer index to entropy
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    layers = sorted(entropy_by_layer.keys())
    entropies = [entropy_by_layer[l] for l in layers]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(layers, entropies, color='#3498db', alpha=0.8, edgecolor='white')
    ax.plot(layers, entropies, 'ro-', markersize=8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Attention Entropy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
