"""
Probe Visualizer for MIRA.

Visualizes probe accuracy and distinguishability across model layers.
This shows at which layer the model can best distinguish between 
clean (safe) and attack (harmful) prompts.

Key visualization:
- Probe Accuracy vs Layer curve (results.md requirement C)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def plot_probe_accuracy_vs_layer(
    accuracies: Dict[int, float],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot Probe Accuracy vs Layer curve.
    
    This visualization shows at which layer the model can best
    distinguish between clean and attack prompts.
    
    Args:
        accuracies: Dict mapping layer index to probe accuracy
        model_name: Name of model for title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = sorted(accuracies.keys())
    accs = [accuracies[l] * 100 for l in layers]  # Convert to percentage
    
    # Main line plot
    ax.plot(layers, accs, 'o-', linewidth=2, markersize=8, color='steelblue', label='Probe Accuracy')
    
    # Fill area under curve
    ax.fill_between(layers, accs, alpha=0.3, color='steelblue')
    
    # Add value labels on points
    for layer, acc in zip(layers, accs):
        ax.annotate(f'{acc:.1f}%', 
                    xy=(layer, acc), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9)
    
    # Reference lines
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect (100%)')
    
    # Find best layer
    best_layer = max(accuracies.keys(), key=lambda x: accuracies[x])
    best_acc = accuracies[best_layer] * 100
    ax.axvline(x=best_layer, color='orange', linestyle=':', alpha=0.7, 
               label=f'Best: Layer {best_layer} ({best_acc:.1f}%)')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Probe Accuracy (%)', fontsize=12)
    ax.set_title(f'Probe Distinguishability by Layer - {model_name}', fontsize=14)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multi_run_probe_comparison(
    run_accuracies: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot probe accuracy comparison across multiple runs/models.
    
    Args:
        run_accuracies: Dict mapping run_id to {layer: accuracy}
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_accuracies)))
    
    for idx, (run_id, accuracies) in enumerate(run_accuracies.items()):
        layers = sorted(accuracies.keys())
        accs = [accuracies[l] * 100 for l in layers]
        
        ax.plot(layers, accs, 'o-', linewidth=2, markersize=6, 
                color=colors[idx], label=run_id[:20])
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Probe Accuracy (%)', fontsize=12)
    ax.set_title('Probe Accuracy Comparison Across Runs/Models', fontsize=14)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def train_probes_across_layers(
    model,
    safe_prompts: List[str],
    harmful_prompts: List[str],
    layers: Optional[List[int]] = None,
    epochs: int = 30,
) -> Dict[int, float]:
    """
    Train probes at each layer and return accuracy dict.
    
    This is a standalone function that trains simple linear probes
    at each layer to measure distinguishability.
    
    Args:
        model: ModelWrapper instance
        safe_prompts: List of safe prompts
        harmful_prompts: List of harmful prompts
        layers: Layers to probe (default: all)
        epochs: Training epochs per probe
        
    Returns:
        Dict mapping layer index to accuracy
    """
    from mira.analysis.subspace import SubspaceAnalyzer
    
    # Default to sampling across layers
    if layers is None:
        n_layers = model.n_layers
        layers = list(range(0, n_layers, max(1, n_layers // 8)))
    
    accuracies = {}
    
    for layer_idx in layers:
        try:
            analyzer = SubspaceAnalyzer(model, layer_idx=layer_idx)
            result = analyzer.train_probe(safe_prompts, harmful_prompts)
            # SubspaceResult is a dataclass with probe_accuracy attribute
            accuracies[layer_idx] = result.probe_accuracy if result.probe_accuracy is not None else None
            if accuracies[layer_idx] is not None:
                print(f"  Layer {layer_idx}: {accuracies[layer_idx]:.1%}")
            else:
                print(f"  Layer {layer_idx}: N/A (no valid data)")

        except Exception as e:
            print(f"  Layer {layer_idx}: Failed ({e})")
            # Mark as None instead of 0.5 for honest reporting
            accuracies[layer_idx] = None
    
    return accuracies


def run_probe_layer_analysis(
    model,
    safe_prompts: List[str],
    harmful_prompts: List[str],
    output_dir: Optional[str] = None,
    n_layers_to_sample: int = 8,
) -> Dict:
    """
    Run complete probe-vs-layer analysis.
    
    Args:
        model: ModelWrapper instance
        safe_prompts: Safe prompts
        harmful_prompts: Harmful prompts
        output_dir: Directory to save outputs
        n_layers_to_sample: Number of layers to sample
        
    Returns:
        Dict with accuracies and best layer info
    """
    print("Running Probe vs Layer Analysis...")
    
    # Sample layers evenly
    total_layers = model.n_layers
    step = max(1, total_layers // n_layers_to_sample)
    layers = list(range(0, total_layers, step))
    
    accuracies = train_probes_across_layers(
        model, safe_prompts, harmful_prompts, layers
    )
    
    result = {
        'accuracies': accuracies,
        'model_name': model.model_name,
        'n_layers': total_layers,
    }
    
    # Find best layer
    if accuracies:
        best_layer = max(accuracies.keys(), key=lambda x: accuracies[x])
        result['best_layer'] = best_layer
        result['best_accuracy'] = accuracies[best_layer]
    
    # Generate plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_path = str(output_path / 'probe_accuracy_vs_layer.png')
        plot_probe_accuracy_vs_layer(
            accuracies, 
            model_name=model.model_name,
            save_path=plot_path
        )
        result['plot_path'] = plot_path
    
    return result
