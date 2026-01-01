"""
MIRA Analysis Package

Provides mechanistic interpretability analysis tools:
- Logit Lens: Layer-wise logit projection analysis
- Attention Visualizer: Attention pattern comparison
- Multi-Run Analyzer: Cross-experiment consistency analysis
- Subspace Analyzer: Subspace analysis for attack detection
- Transformer Tracer: Trace forward pass through layers
- Probe Visualizer: Probe accuracy vs layer visualization
"""

# Import existing analysis tools
from mira.analysis.subspace import SubspaceAnalyzer
from mira.analysis.transformer_tracer import TransformerTracer

# Import new mechanistic interpretability tools
from mira.analysis.logit_lens import LogitLens
from mira.analysis.attention_visualizer import AttentionVisualizer
from mira.analysis.multi_run_analyzer import MultiRunAnalyzer
from mira.analysis.probe_visualizer import (
    plot_probe_accuracy_vs_layer,
    run_probe_layer_analysis,
)

__all__ = [
    'SubspaceAnalyzer',
    'TransformerTracer', 
    'LogitLens', 
    'AttentionVisualizer', 
    'MultiRunAnalyzer',
    'plot_probe_accuracy_vs_layer',
    'run_probe_layer_analysis',
]
