"""
Attention Visualizer for MIRA.

Captures and visualizes attention patterns to understand how attacks
affect the model's attention mechanism. Essential for identifying
which attention heads are vulnerable to adversarial manipulation.

Key analyses:
- Attention pattern heatmaps (clean vs attack)
- Attention divergence metrics by layer/head
- Head attribution for attack success
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from mira.core.model_wrapper import ModelWrapper


@dataclass
class AttentionResult:
    """Result from attention capture."""
    prompt: str
    tokens: List[str]
    attention_patterns: Dict[int, torch.Tensor]  # {layer_idx: [n_heads, seq_len, seq_len]}
    layer_head_info: Dict[int, Dict[str, float]]  # {layer: {entropy, max_attention, etc}}


class AttentionVisualizer:
    """
    Attention Pattern Visualization Tool.
    
    Captures attention weights from all layers/heads and provides
    visualization and analysis tools to compare clean vs attack patterns.
    
    Key insight: Attacks often manipulate attention to divert focus
    from safety-relevant tokens or inject malicious context.
    """
    
    def __init__(self, model: ModelWrapper):
        """
        Initialize Attention Visualizer.
        
        Args:
            model: ModelWrapper instance with loaded model
        """
        self.model = model
        self.device = model.device
        self.n_layers = model.n_layers
        self.n_heads = model.n_heads
        
    def capture(self, prompt: str) -> AttentionResult:
        """
        Capture attention patterns for a prompt.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            AttentionResult with attention patterns from all layers
        """
        inputs = self.model.tokenize(prompt)
        
        with torch.no_grad():
            outputs = self.model.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )
        
        # Get tokens for labeling
        tokens = self.model.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0].tolist()
        )
        
        # Extract attention patterns
        attention_patterns = {}
        layer_head_info = {}
        
        for layer_idx, attn in enumerate(outputs.attentions):
            # attn shape: [batch, n_heads, seq_len, seq_len]
            attn_np = attn[0].detach().cpu()
            attention_patterns[layer_idx] = attn_np
            
            # Compute statistics for each layer
            layer_head_info[layer_idx] = self._compute_attention_stats(attn_np)
        
        return AttentionResult(
            prompt=prompt,
            tokens=tokens,
            attention_patterns=attention_patterns,
            layer_head_info=layer_head_info,
        )
    
    def _compute_attention_stats(self, attn: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for attention patterns."""
        # attn shape: [n_heads, seq_len, seq_len]
        
        # Entropy (higher = more dispersed attention)
        # Compute per-head entropy then average
        eps = 1e-10
        entropy_per_head = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=-1)
        avg_entropy = entropy_per_head.mean().item()
        
        # Max attention weight (how focused)
        max_attn = attn.max().item()
        
        # Attention to first token (often indicates "attention sinks")
        first_token_attn = attn[:, :, 0].mean().item()
        
        # Diagonal attention (self-attention strength)
        diag_mask = torch.eye(attn.shape[-1], dtype=torch.bool)
        diag_attn = attn[:, diag_mask].mean().item() if attn.shape[-1] > 0 else 0
        
        return {
            'entropy': avg_entropy,
            'max_attention': max_attn,
            'first_token_attention': first_token_attn,
            'diagonal_attention': diag_attn,
        }
    
    def compare(
        self,
        clean_prompt: str,
        attack_prompt: str,
    ) -> Tuple[AttentionResult, AttentionResult, Dict]:
        """
        Compare attention patterns between clean and attack prompts.
        
        Args:
            clean_prompt: Normal/safe prompt
            attack_prompt: Attack/adversarial prompt
            
        Returns:
            Tuple of (clean_result, attack_result, comparison_stats)
        """
        clean_result = self.capture(clean_prompt)
        attack_result = self.capture(attack_prompt)
        
        comparison = {
            'kl_divergence': {},      # Per-layer KL divergence
            'entropy_change': {},      # Per-layer entropy change
            'head_divergence': {},     # Per-head divergence scores
            'most_affected_heads': [],  # Top affected heads
        }
        
        head_scores = []
        
        for layer_idx in clean_result.attention_patterns.keys():
            clean_attn = clean_result.attention_patterns[layer_idx]
            attack_attn = attack_result.attention_patterns[layer_idx]
            
            # Handle different sequence lengths (truncate to min)
            min_len = min(clean_attn.shape[-1], attack_attn.shape[-1])
            clean_attn = clean_attn[:, :min_len, :min_len]
            attack_attn = attack_attn[:, :min_len, :min_len]
            
            # Per-layer KL divergence (average across heads)
            eps = 1e-10
            kl_per_head = []
            for head_idx in range(clean_attn.shape[0]):
                c = clean_attn[head_idx].flatten() + eps
                a = attack_attn[head_idx].flatten() + eps
                kl = F.kl_div(a.log(), c, reduction='sum').item()
                kl_per_head.append(kl)
                head_scores.append((layer_idx, head_idx, kl))
            
            comparison['kl_divergence'][layer_idx] = np.mean(kl_per_head)
            comparison['head_divergence'][layer_idx] = kl_per_head
            
            # Entropy change
            clean_entropy = clean_result.layer_head_info[layer_idx]['entropy']
            attack_entropy = attack_result.layer_head_info[layer_idx]['entropy']
            comparison['entropy_change'][layer_idx] = attack_entropy - clean_entropy
        
        # Find most affected heads
        head_scores.sort(key=lambda x: x[2], reverse=True)
        comparison['most_affected_heads'] = head_scores[:10]
        
        return clean_result, attack_result, comparison
    
    def plot_attention_heatmap(
        self,
        result: AttentionResult,
        layer: int,
        head: Optional[int] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Plot attention heatmap for a specific layer/head.
        
        Args:
            result: AttentionResult from capture()
            layer: Layer index
            head: Head index (None = average across heads)
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        attn = result.attention_patterns[layer]
        
        if head is not None:
            attn_matrix = attn[head].numpy()
            title = f'Attention Pattern - Layer {layer}, Head {head}'
        else:
            attn_matrix = attn.mean(dim=0).numpy()
            title = f'Attention Pattern - Layer {layer} (Avg across heads)'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Truncate tokens for display
        tokens = [t[:10] for t in result.tokens]
        
        sns.heatmap(
            attn_matrix,
            ax=ax,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            vmin=0,
            vmax=min(1.0, attn_matrix.max()),
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_comparison(
        self,
        clean_result: AttentionResult,
        attack_result: AttentionResult,
        layer: int,
        head: Optional[int] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of clean vs attack attention.
        
        Args:
            clean_result: AttentionResult from clean prompt
            attack_result: AttentionResult from attack prompt
            layer: Layer index
            head: Head index (None = average)
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        clean_attn = clean_result.attention_patterns[layer]
        attack_attn = attack_result.attention_patterns[layer]
        
        if head is not None:
            clean_matrix = clean_attn[head].numpy()
            attack_matrix = attack_attn[head].numpy()
            head_str = f'Head {head}'
        else:
            clean_matrix = clean_attn.mean(dim=0).numpy()
            attack_matrix = attack_attn.mean(dim=0).numpy()
            head_str = 'Avg'
        
        # Truncate to same size for difference
        min_len = min(clean_matrix.shape[0], attack_matrix.shape[0])
        clean_matrix = clean_matrix[:min_len, :min_len]
        attack_matrix = attack_matrix[:min_len, :min_len]
        
        # Clean attention
        sns.heatmap(clean_matrix, ax=axes[0], cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title(f'Clean - Layer {layer}, {head_str}')
        axes[0].set_xlabel('Key')
        axes[0].set_ylabel('Query')
        
        # Attack attention
        sns.heatmap(attack_matrix, ax=axes[1], cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title(f'Attack - Layer {layer}, {head_str}')
        axes[1].set_xlabel('Key')
        axes[1].set_ylabel('Query')
        
        # Difference
        diff_matrix = attack_matrix - clean_matrix
        max_abs = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
        sns.heatmap(diff_matrix, ax=axes[2], cmap='RdBu_r', 
                    vmin=-max_abs, vmax=max_abs, center=0)
        axes[2].set_title('Difference (Attack - Clean)')
        axes[2].set_xlabel('Key')
        axes[2].set_ylabel('Query')
        
        plt.suptitle(f'Attention Comparison: Layer {layer}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_layer_divergence(
        self,
        comparison: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot attention divergence across layers.
        
        Args:
            comparison: Comparison dict from compare()
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        layers = sorted(comparison['kl_divergence'].keys())
        
        # KL divergence by layer
        kl_values = [comparison['kl_divergence'][l] for l in layers]
        axes[0].bar(layers, kl_values, color='steelblue', alpha=0.7)
        axes[0].plot(layers, kl_values, 'ro-', markersize=5)
        axes[0].set_xlabel('Layer', fontsize=12)
        axes[0].set_ylabel('KL Divergence', fontsize=12)
        axes[0].set_title('Attention KL Divergence by Layer', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Entropy change by layer
        entropy_values = [comparison['entropy_change'][l] for l in layers]
        colors = ['red' if v > 0 else 'blue' for v in entropy_values]
        axes[1].bar(layers, entropy_values, color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Layer', fontsize=12)
        axes[1].set_ylabel('Entropy Change', fontsize=12)
        axes[1].set_title('Attention Entropy Change by Layer\n(+: more dispersed, -: more focused)', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_head_attribution(
        self,
        comparison: Dict,
        top_k: int = 20,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot head attribution showing most affected attention heads.
        
        Args:
            comparison: Comparison dict from compare()
            top_k: Number of top heads to show
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        heads = comparison['most_affected_heads'][:top_k]
        labels = [f'L{l}H{h}' for l, h, _ in heads]
        values = [v for _, _, v in heads]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(values)))
        
        bars = ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        
        ax.set_xlabel('KL Divergence', fontsize=12)
        ax.set_ylabel('Layer.Head', fontsize=12)
        ax.set_title('Most Affected Attention Heads', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


def run_attention_analysis(
    model: ModelWrapper,
    clean_prompt: str,
    attack_prompt: str,
    output_dir: str = None,
    layers_to_plot: List[int] = None,
) -> Dict:
    """
    Convenience function to run full attention analysis.
    
    Args:
        model: ModelWrapper instance
        clean_prompt: Clean/safe prompt
        attack_prompt: Attack prompt
        output_dir: Directory to save plots
        layers_to_plot: Specific layers to generate heatmaps for
        
    Returns:
        Dict with results and comparison statistics
    """
    viz = AttentionVisualizer(model)
    
    clean_result, attack_result, comparison = viz.compare(
        clean_prompt, attack_prompt
    )
    
    result = {
        'clean': clean_result,
        'attack': attack_result,
        'comparison': comparison,
        'plots': {},
    }
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate layer divergence plot
        result['plots']['divergence'] = str(output_path / 'attention_divergence.png')
        viz.plot_layer_divergence(comparison, save_path=result['plots']['divergence'])
        
        # Generate head attribution plot
        result['plots']['head_attribution'] = str(output_path / 'attention_heads.png')
        viz.plot_head_attribution(comparison, save_path=result['plots']['head_attribution'])
        
        # Plot specific layers if requested
        if layers_to_plot is None:
            # Default: plot layers with highest divergence
            sorted_layers = sorted(comparison['kl_divergence'].items(), 
                                   key=lambda x: x[1], reverse=True)
            layers_to_plot = [l for l, _ in sorted_layers[:3]]
        
        for layer in layers_to_plot:
            path = str(output_path / f'attention_layer_{layer}.png')
            result['plots'][f'layer_{layer}'] = path
            viz.plot_comparison(clean_result, attack_result, layer, save_path=path)
    
    return result
