"""
Live visualization for model internals during processing.

Provides real-time display of:
- Attention patterns
- Activation flows
- Layer-by-layer predictions
- Token probability distributions
"""

import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LiveVisualizer:
    """
    Real-time visualization of model processing.
    
    Shows model internals as processing happens.
    """
    
    def __init__(self, interactive: bool = True, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize live visualizer.
        
        Args:
            interactive: Enable interactive mode
            figsize: Figure size
        """
        self.interactive = interactive
        self.figsize = figsize
        self.fig = None
        self.axes = None
        
        if HAS_MATPLOTLIB and interactive:
            plt.ion()
    
    def display_attention_live(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        layer: int,
        head: int,
        delay: float = 0.5,
    ) -> None:
        """
        Display attention pattern in real-time.
        
        Args:
            attention_weights: Attention matrix [seq, seq]
            tokens: Token strings
            layer: Layer index
            head: Head index
            delay: Display delay in seconds
        """
        if not HAS_MATPLOTLIB:
            self._display_attention_ascii(attention_weights, tokens, layer, head)
            return
        
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)
        
        ax = self.axes[0]
        ax.clear()
        
        # Display heatmap
        im = ax.imshow(attention_weights, cmap="Blues", aspect="auto")
        
        # Add token labels if not too many
        n_tokens = len(tokens)
        if n_tokens <= 20:
            ax.set_xticks(range(n_tokens))
            ax.set_yticks(range(n_tokens))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
        
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(f"Attention Pattern - Layer {layer}, Head {head}")
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        if self.interactive:
            plt.pause(delay)
            plt.draw()
    
    def display_activation_flow(
        self,
        activations: Dict[int, np.ndarray],
        tokens: List[str],
        delay: float = 0.3,
    ) -> None:
        """
        Display activation magnitudes flowing through layers.
        
        Args:
            activations: Dict mapping layer -> activation tensor
            tokens: Token strings
            delay: Display delay
        """
        if not HAS_MATPLOTLIB:
            self._display_activation_ascii(activations)
            return
        
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)
        
        ax = self.axes[1]
        ax.clear()
        
        # Compute magnitudes per layer
        layers = sorted(activations.keys())
        n_layers = len(layers)
        n_tokens = len(tokens)
        
        # Create heatmap of activation magnitudes
        magnitude_matrix = np.zeros((n_layers, n_tokens))
        for i, layer in enumerate(layers):
            acts = activations[layer]
            if len(acts.shape) > 1:
                mags = np.linalg.norm(acts, axis=-1)
            else:
                mags = np.abs(acts)
            magnitude_matrix[i, :len(mags)] = mags[:n_tokens]
        
        im = ax.imshow(magnitude_matrix, cmap="viridis", aspect="auto")
        
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        ax.set_title("Activation Flow Through Layers")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in layers])
        
        if n_tokens <= 20:
            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.8, label="Magnitude")
        plt.tight_layout()
        
        if self.interactive:
            plt.pause(delay)
            plt.draw()
    
    def display_token_probabilities(
        self,
        token_probs: Dict[str, float],
        title: str = "Next Token Probabilities",
        top_k: int = 10,
    ) -> None:
        """
        Display top-k token probabilities as bar chart.
        
        Args:
            token_probs: Dict mapping token -> probability
            title: Chart title
            top_k: Number of top tokens to show
        """
        if not HAS_MATPLOTLIB:
            self._display_probs_ascii(token_probs, top_k)
            return
        
        # Sort by probability
        sorted_probs = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        tokens = [t for t, _ in sorted_probs]
        probs = [p for _, p in sorted_probs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(tokens)))
        bars = ax.barh(range(len(tokens)), probs, color=colors)
        
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel("Probability")
        ax.set_title(title)
        ax.invert_yaxis()
        
        # Add probability labels
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{prob:.2%}", va="center", fontsize=9)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def display_layer_predictions(
        self,
        layer_predictions: Dict[int, str],
        final_prediction: str,
    ) -> None:
        """
        Display how predictions evolve through layers.
        
        Args:
            layer_predictions: Dict mapping layer -> predicted token
            final_prediction: Final model prediction
        """
        print("\n  Layer-by-Layer Prediction Evolution:")
        print("  " + "-" * 50)
        
        layers = sorted(layer_predictions.keys())
        
        for layer in layers:
            pred = layer_predictions[layer]
            marker = "*" if pred == final_prediction else " "
            print(f"  {marker} Layer {layer:2d}: {pred}")
        
        print("  " + "-" * 50)
        print(f"    Final: {final_prediction}")
    
    def _display_attention_ascii(
        self,
        attention: np.ndarray,
        tokens: List[str],
        layer: int,
        head: int,
    ) -> None:
        """ASCII fallback for attention display."""
        print(f"\n  Attention Pattern - Layer {layer}, Head {head}")
        print("  " + "-" * 50)
        
        n = min(len(tokens), 10)
        
        # Header
        header = "      "
        for t in tokens[:n]:
            header += f"{t[:4]:>5}"
        print(header)
        
        # Rows
        for i in range(n):
            row = f"{tokens[i][:4]:>5} "
            for j in range(n):
                val = attention[i, j]
                if val > 0.3:
                    char = "##"
                elif val > 0.1:
                    char = "**"
                elif val > 0.05:
                    char = ".."
                else:
                    char = "  "
                row += f"{char:>5}"
            print(row)
    
    def _display_activation_ascii(self, activations: Dict[int, np.ndarray]) -> None:
        """ASCII fallback for activation display."""
        print("\n  Activation Magnitudes by Layer:")
        print("  " + "-" * 50)
        
        layers = sorted(activations.keys())
        max_mag = max(np.linalg.norm(activations[l]).item() for l in layers)
        
        for layer in layers:
            mag = np.linalg.norm(activations[layer]).item()
            bar_len = int(40 * mag / max_mag)
            bar = "#" * bar_len + "-" * (40 - bar_len)
            print(f"  Layer {layer:2d}: [{bar}] {mag:.2f}")
    
    def _display_probs_ascii(self, token_probs: Dict[str, float], top_k: int) -> None:
        """ASCII fallback for probability display."""
        print("\n  Top Token Probabilities:")
        print("  " + "-" * 50)
        
        sorted_probs = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for token, prob in sorted_probs:
            bar_len = int(40 * prob)
            bar = "#" * bar_len + "-" * (40 - bar_len)
            print(f"  {token:>15}: [{bar}] {prob:.2%}")
    
    def close(self) -> None:
        """Close visualizer."""
        if HAS_MATPLOTLIB and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


def print_model_state(
    prompt: str,
    layer: int,
    activation_norm: float,
    top_tokens: List[Tuple[str, float]],
    step: int = 0,
) -> None:
    """
    Print current model processing state.
    
    Args:
        prompt: Input prompt
        layer: Current layer being processed
        activation_norm: Activation magnitude
        top_tokens: Top predicted tokens with probabilities
        step: Processing step number
    """
    print(f"\n  {'='*60}")
    print(f"  PROCESSING STATE - Step {step}")
    print(f"  {'='*60}")
    print(f"  Input: {prompt[:50]}...")
    print(f"  Layer: {layer}")
    print(f"  Activation Norm: {activation_norm:.4f}")
    print(f"  Top Predictions:")
    for i, (token, prob) in enumerate(top_tokens[:5]):
        print(f"    {i+1}. {token:>15} ({prob:.2%})")


def visualize_attack_progress(
    step: int,
    total_steps: int,
    loss: float,
    best_loss: float,
    suffix_preview: str,
    success: bool = False,
) -> None:
    """
    Display attack optimization progress.
    
    Args:
        step: Current step
        total_steps: Total steps
        loss: Current loss
        best_loss: Best loss so far
        suffix_preview: Current suffix preview
        success: Whether attack succeeded
    """
    progress = step / total_steps
    bar_width = 40
    filled = int(bar_width * progress)
    bar = "#" * filled + "-" * (bar_width - filled)
    
    status = "SUCCESS" if success else "RUNNING"
    
    print(f"\r  [{bar}] {progress*100:5.1f}% | Loss: {loss:.4f} | Best: {best_loss:.4f} | {status}", end="", flush=True)
    
    if step == total_steps or success:
        print()


def display_subspace_analysis(
    probe_accuracy: float,
    refusal_norm: float,
    acceptance_norm: float,
    separation: float,
) -> None:
    """
    Display subspace analysis results visually.
    
    Args:
        probe_accuracy: Linear probe accuracy
        refusal_norm: Refusal direction norm
        acceptance_norm: Acceptance direction norm
        separation: Distance between subspaces
    """
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │               SUBSPACE ANALYSIS RESULTS                 │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │   Safe                    ●────────●  Harmful           │""")
    
    # Visual representation of separation
    sep_bar = int(separation * 20)
    sep_visual = "─" * sep_bar
    print(f"  │   ●{sep_visual}●                               │")
    
    print(f"""  
  │                                                         │
  │   Probe Accuracy:     {probe_accuracy:>30.2%}           │
  │   Refusal Direction:  {refusal_norm:>30.4f}             │
  │   Acceptance Dir:     {acceptance_norm:>30.4f}          │
  │   Separation:         {separation:>30.4f}               │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
    """)
