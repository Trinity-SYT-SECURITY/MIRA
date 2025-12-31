"""
Enhanced Attention Heatmap Visualization

Generates publication-quality attention heatmaps comparing clean vs attack prompts.
Implements requirements from results.md for attention pattern analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_attention_comparison_heatmap(
    clean_attention: torch.Tensor,
    attack_attention: torch.Tensor,
    layer_idx: int,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """
    Plot side-by-side attention heatmaps for clean vs attack.
    
    Args:
        clean_attention: Attention weights from clean prompt [heads, seq, seq]
        attack_attention: Attention weights from attack prompt [heads, seq, seq]
        layer_idx: Layer index for title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Average over heads
    if clean_attention.dim() == 3:
        clean_attn_avg = clean_attention.mean(dim=0).cpu().numpy()
    else:
        clean_attn_avg = clean_attention.cpu().numpy()
    
    if attack_attention.dim() == 3:
        attack_attn_avg = attack_attention.mean(dim=0).cpu().numpy()
    else:
        attack_attn_avg = attack_attention.cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Find common scale
    vmax = max(clean_attn_avg.max(), attack_attn_avg.max())
    
    # Clean attention
    sns.heatmap(clean_attn_avg, ax=ax1, cmap='Blues', 
               vmin=0, vmax=vmax, square=True,
               cbar_kws={'label': 'Attention Weight'})
    ax1.set_title(f'Clean Prompt\nLayer {layer_idx}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    
    # Attack attention
    sns.heatmap(attack_attn_avg, ax=ax2, cmap='Reds',
               vmin=0, vmax=vmax, square=True,
               cbar_kws={'label': 'Attention Weight'})
    ax2.set_title(f'Attack Prompt\nLayer {layer_idx}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('')
    
    # Difference
    diff = attack_attn_avg - clean_attn_avg
    diff_max = max(abs(diff.min()), abs(diff.max()))
    
    sns.heatmap(diff, ax=ax3, cmap='RdBu_r', center=0,
               vmin=-diff_max, vmax=diff_max, square=True,
               cbar_kws={'label': 'Δ Attention'})
    ax3.set_title(f'Attack - Clean\nLayer {layer_idx}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_head_wise_attention(
    attention: torch.Tensor,
    layer_idx: int,
    prompt_type: str = "clean",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot attention patterns for each head separately.
    
    Args:
        attention: Attention weights [heads, seq, seq]
        layer_idx: Layer index
        prompt_type: "clean" or "attack"
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if attention.dim() != 3:
        raise ValueError(f"Expected 3D attention tensor, got {attention.dim()}D")
    
    num_heads = attention.shape[0]
    attn_np = attention.cpu().numpy()
    
    # Create grid of subplots
    n_cols = 4
    n_rows = (num_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    vmax = attn_np.max()
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(attn_np[head_idx], ax=ax, cmap='viridis',
                   vmin=0, vmax=vmax, square=True, cbar=False)
        ax.set_title(f'Head {head_idx}', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'{prompt_type.capitalize()} Prompt - Layer {layer_idx} - All Heads',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_entropy_comparison(
    clean_attention: torch.Tensor,
    attack_attention: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot attention entropy comparison across heads.
    
    Args:
        clean_attention: Clean attention [heads, seq, seq]
        attack_attention: Attack attention [heads, seq, seq]
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    def compute_entropy(attn):
        """Compute entropy for each head."""
        # attn: [heads, seq, seq]
        entropies = []
        for head_idx in range(attn.shape[0]):
            head_attn = attn[head_idx]  # [seq, seq]
            # Average entropy across query positions
            head_entropy = -(head_attn * torch.log(head_attn + 1e-10)).sum(dim=-1).mean()
            entropies.append(head_entropy.item())
        return entropies
    
    clean_entropies = compute_entropy(clean_attention)
    attack_entropies = compute_entropy(attack_attention)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(clean_entropies))
    width = 0.35
    
    ax.bar(x - width/2, clean_entropies, width, label='Clean', color='#3498db')
    ax.bar(x + width/2, attack_entropies, width, label='Attack', color='#e74c3c')
    
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Head', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_attention_heatmap_report(
    clean_attention_dict: Dict[int, torch.Tensor],
    attack_attention_dict: Dict[int, torch.Tensor],
    output_dir: str,
    target_layer: Optional[int] = None,
) -> List[str]:
    """
    Generate comprehensive attention heatmap report.
    
    Args:
        clean_attention_dict: Dict mapping layer_idx -> attention tensor
        attack_attention_dict: Dict mapping layer_idx -> attention tensor
        output_dir: Directory to save charts
        target_layer: Specific layer to focus on (None = middle layer)
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Determine target layer
    if target_layer is None:
        layers = sorted(clean_attention_dict.keys())
        target_layer = layers[len(layers) // 2] if layers else 0
    
    # 1. Comparison heatmap for target layer
    if target_layer in clean_attention_dict and target_layer in attack_attention_dict:
        fig = plot_attention_comparison_heatmap(
            clean_attention_dict[target_layer],
            attack_attention_dict[target_layer],
            target_layer,
            save_path=str(output_path / f"attention_comparison_layer{target_layer}.png")
        )
        plt.close(fig)
        saved_files.append(f"attention_comparison_layer{target_layer}.png")
    
    # 2. Head-wise attention for clean
    if target_layer in clean_attention_dict:
        fig = plot_head_wise_attention(
            clean_attention_dict[target_layer],
            target_layer,
            "clean",
            save_path=str(output_path / f"attention_heads_clean_layer{target_layer}.png")
        )
        plt.close(fig)
        saved_files.append(f"attention_heads_clean_layer{target_layer}.png")
    
    # 3. Head-wise attention for attack
    if target_layer in attack_attention_dict:
        fig = plot_head_wise_attention(
            attack_attention_dict[target_layer],
            target_layer,
            "attack",
            save_path=str(output_path / f"attention_heads_attack_layer{target_layer}.png")
        )
        plt.close(fig)
        saved_files.append(f"attention_heads_attack_layer{target_layer}.png")
    
    # 4. Entropy comparison
    if target_layer in clean_attention_dict and target_layer in attack_attention_dict:
        fig = plot_attention_entropy_comparison(
            clean_attention_dict[target_layer],
            attack_attention_dict[target_layer],
            save_path=str(output_path / f"attention_entropy_layer{target_layer}.png")
        )
        plt.close(fig)
        saved_files.append(f"attention_entropy_layer{target_layer}.png")
    
    return saved_files
