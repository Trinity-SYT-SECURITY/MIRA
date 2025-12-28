"""
Real-time flow visualization during model processing.

Provides live display of:
- Layer processing progress
- Safety direction shifts
- Attack bypass detection
- Decision emergence
"""

import sys
import time
from typing import Dict, List, Optional, Callable, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RealTimeFlowViz:
    """
    Real-time visualization of model processing flow.
    
    Shows live updates as model processes each layer.
    """
    
    def __init__(self, n_layers: int, figsize: tuple = (14, 8)):
        """
        Initialize real-time visualizer.
        
        Args:
            n_layers: Number of layers in model
            figsize: Figure size
        """
        self.n_layers = n_layers
        self.figsize = figsize
        self.layer_data = []
        self.fig = None
        self.ax = None
        
        if HAS_MATPLOTLIB:
            plt.ion()
    
    def start(self) -> None:
        """Start visualization session."""
        if HAS_MATPLOTLIB:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.ax.set_xlim(-1, self.n_layers)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_xlabel("Layer")
            self.ax.set_ylabel("Safety Score (- refusal, + acceptance)")
            self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            self.ax.set_title("Real-Time Processing Flow")
            plt.show(block=False)
        
        self.layer_data = []
        print("\n  ╔══════════════════════════════════════════════════════════╗")
        print("  ║         REAL-TIME MODEL PROCESSING VISUALIZATION         ║")
        print("  ╚══════════════════════════════════════════════════════════╝\n")
    
    def update_layer(
        self,
        layer_idx: int,
        refusal_score: float,
        acceptance_score: float,
        direction: str,
        top_prediction: str = "",
    ) -> None:
        """
        Update visualization with new layer data.
        
        Args:
            layer_idx: Current layer index
            refusal_score: Refusal direction score
            acceptance_score: Acceptance direction score
            direction: Current direction ("neutral", "refusal", "acceptance")
            top_prediction: Top predicted token
        """
        # Store data
        self.layer_data.append({
            "layer": layer_idx,
            "refusal": refusal_score,
            "acceptance": acceptance_score,
            "direction": direction,
            "prediction": top_prediction,
        })
        
        # Update matplotlib plot
        if HAS_MATPLOTLIB and self.fig:
            self.ax.clear()
            
            layers = [d["layer"] for d in self.layer_data]
            refusals = [d["refusal"] for d in self.layer_data]
            acceptances = [d["acceptance"] for d in self.layer_data]
            
            # Plot lines
            self.ax.plot(layers, refusals, 'r-o', label="Refusal", linewidth=2, markersize=8)
            self.ax.plot(layers, acceptances, 'g-o', label="Acceptance", linewidth=2, markersize=8)
            
            # Color current point by direction
            colors = {'neutral': 'gray', 'refusal': 'red', 'acceptance': 'green'}
            for d in self.layer_data:
                self.ax.scatter([d["layer"]], [0], c=colors[d["direction"]], s=100, zorder=5)
            
            self.ax.set_xlim(-0.5, self.n_layers - 0.5)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_xlabel("Layer")
            self.ax.set_ylabel("Direction Score")
            self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            self.ax.legend()
            self.ax.set_title(f"Processing Layer {layer_idx}/{self.n_layers-1}")
            
            plt.draw()
            plt.pause(0.05)
        
        # Console output
        self._print_layer_progress(layer_idx, refusal_score, acceptance_score, direction, top_prediction)
    
    def _print_layer_progress(
        self,
        layer_idx: int,
        refusal_score: float,
        acceptance_score: float,
        direction: str,
        top_prediction: str,
    ) -> None:
        """Print ASCII progress for layer."""
        # Create visual bar
        bar_width = 40
        mid = bar_width // 2
        
        # Normalize scores to bar positions
        ref_pos = max(0, min(mid, int(mid - refusal_score * mid)))
        acc_pos = max(mid, min(bar_width, int(mid + acceptance_score * mid)))
        
        bar = [" "] * bar_width
        bar[mid] = "│"
        
        # Fill based on direction
        if direction == "refusal":
            for i in range(ref_pos, mid):
                bar[i] = "█"
            symbol = "❌"
        elif direction == "acceptance":
            for i in range(mid + 1, acc_pos):
                bar[i] = "█"
            symbol = "✅"
        else:
            symbol = "○"
        
        bar_str = "".join(bar)
        pred = top_prediction[:10] if top_prediction else "?"
        
        print(f"\r  L{layer_idx:2d} [{bar_str}] {symbol} → {pred:<10}", end="", flush=True)
        
        if layer_idx == self.n_layers - 1:
            print()
    
    def show_summary(self) -> None:
        """Show final summary of processing flow."""
        if not self.layer_data:
            return
        
        print("\n  ─────────────────────────────────────────────────────────")
        print("  PROCESSING SUMMARY:")
        print("  ─────────────────────────────────────────────────────────")
        
        # Find decision point
        decision_layer = None
        for d in self.layer_data:
            if d["direction"] == "refusal":
                decision_layer = d["layer"]
                break
        
        if decision_layer is not None:
            print(f"  ⚠️  REFUSAL DETECTED at Layer {decision_layer}")
        else:
            print("  ✅ NO REFUSAL DETECTED")
        
        # Show direction changes
        changes = []
        prev_dir = None
        for d in self.layer_data:
            if d["direction"] != prev_dir:
                changes.append((d["layer"], d["direction"]))
                prev_dir = d["direction"]
        
        if len(changes) > 1:
            print(f"  Direction changes: {' → '.join([f'L{l}:{d}' for l, d in changes])}")
    
    def close(self) -> None:
        """Close visualization."""
        if HAS_MATPLOTLIB and self.fig:
            plt.close(self.fig)
            self.fig = None


def print_flow_diagram(layers_data: List[Dict], title: str = "Processing Flow") -> None:
    """
    Print ASCII flow diagram of processing.
    
    Args:
        layers_data: List of layer states
        title: Diagram title
    """
    print(f"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  {title:^63} ║
  ╠═══════════════════════════════════════════════════════════════════╣""")
    
    for data in layers_data:
        layer = data.get("layer", 0)
        direction = data.get("direction", "neutral")
        score = data.get("score", 0.0)
        
        if direction == "refusal":
            block = "████████████████████"
            side = "← REFUSAL"
        elif direction == "acceptance":
            block = "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
            side = "→ ACCEPT"
        else:
            block = "░░░░░░░░░░░░░░░░░░░░"
            side = "  NEUTRAL"
        
        print(f"  ║  Layer {layer:2d}: [{block}] {side:>12}           ║")
    
    print("  ╚═══════════════════════════════════════════════════════════════════╝")


def animate_attack_comparison(
    clean_data: List[Dict],
    attack_data: List[Dict],
    output_path: Optional[str] = None,
) -> None:
    """
    Animate comparison between clean and attacked processing.
    
    Args:
        clean_data: Clean prompt layer data
        attack_data: Attacked prompt layer data
        output_path: Optional path to save animation
    """
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available for animation")
        return
    
    n_layers = len(clean_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Clean processing
        if frame < n_layers:
            layers = [d["layer"] for d in clean_data[:frame+1]]
            scores = [d.get("refusal", 0) for d in clean_data[:frame+1]]
            ax1.bar(layers, scores, color='red', alpha=0.7)
        
        ax1.set_xlim(-0.5, n_layers - 0.5)
        ax1.set_ylim(-1, 1)
        ax1.set_title("Clean Prompt (Blocked)")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Refusal Score")
        
        # Attacked processing
        if frame < n_layers:
            layers = [d["layer"] for d in attack_data[:frame+1]]
            scores = [d.get("refusal", 0) for d in attack_data[:frame+1]]
            colors = ['green' if s < 0.1 else 'orange' for s in scores]
            ax2.bar(layers, scores, color=colors, alpha=0.7)
        
        ax2.set_xlim(-0.5, n_layers - 0.5)
        ax2.set_ylim(-1, 1)
        ax2.set_title("Attacked Prompt (Bypassed)")
        ax2.set_xlabel("Layer")
        
        return ax1, ax2
    
    ani = animation.FuncAnimation(fig, update, frames=n_layers, interval=200, repeat=False)
    
    if output_path:
        ani.save(output_path, writer='pillow', fps=5)
        print(f"  Animation saved: {output_path}")
    else:
        plt.show()
    
    plt.close()
