"""
MIRA Analysis Package

Provides mechanistic interpretability analysis tools:
- Logit Lens: Layer-wise logit projection analysis
- Attention Visualizer: Attention pattern comparison
- Multi-Run Analyzer: Cross-experiment consistency analysis
"""

from mira.analysis.logit_lens import LogitLens
from mira.analysis.attention_visualizer import AttentionVisualizer
from mira.analysis.multi_run_analyzer import MultiRunAnalyzer

__all__ = ['LogitLens', 'AttentionVisualizer', 'MultiRunAnalyzer']
