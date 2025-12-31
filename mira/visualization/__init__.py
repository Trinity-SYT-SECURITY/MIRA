"""Visualization module initialization."""

from mira.visualization.subspace_plot import (
    plot_subspace_2d,
    plot_subspace_3d,
    plot_trajectory,
    plot_loss_curve,
)
from mira.visualization.attention_plot import (
    plot_attention_heatmap,
    plot_attention_comparison,
    plot_head_importance,
    plot_attention_entropy,
)
from mira.visualization.research_charts import (
    ResearchChartGenerator,
)
from mira.visualization.live_display import (
    LiveVisualizer,
    visualize_attack_progress,
    display_subspace_analysis,
)
from mira.visualization.flow_viz import (
    RealTimeFlowViz,
    print_flow_diagram,
)
from mira.visualization.transformer_internals import (
    get_transformer_internals_html,
)
from mira.visualization.transformer_attack_viz import (
    get_transformer_attack_html,
)
from mira.visualization.transformer_detailed_viz import (
    get_detailed_transformer_html,
)
from mira.visualization.flow_graph_viz import (
    get_flow_graph_html,
)

# Signature Matrix Visualization
try:
    from mira.visualization.signature_viz import SignatureVisualizer
    SIGNATURE_VIZ_AVAILABLE = True
except ImportError:
    SIGNATURE_VIZ_AVAILABLE = False
    SignatureVisualizer = None

# Comprehensive Visualization
try:
    from mira.visualization.comprehensive_viz import ComprehensiveVisualizer
    COMPREHENSIVE_VIZ_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_VIZ_AVAILABLE = False
    ComprehensiveVisualizer = None

# Cost Visualization
try:
    from mira.visualization.cost_viz import (
        plot_cost_efficiency,
        plot_cost_breakdown,
        generate_cost_report,
    )
    COST_VIZ_AVAILABLE = True
except ImportError:
    COST_VIZ_AVAILABLE = False

# Heatmap Visualization (Three-tier ASR)
try:
    from mira.visualization.heatmap_viz import (
        plot_layer_feature_heatmap,
        plot_attack_comparison_heatmap,
        plot_feature_consistency_map,
        plot_tiered_asr_chart,
        generate_comprehensive_heatmap_report,
    )
    HEATMAP_VIZ_AVAILABLE = True
except ImportError:
    HEATMAP_VIZ_AVAILABLE = False

__all__ = [
    "plot_subspace_2d",
    "plot_subspace_3d",
    "plot_trajectory",
    "plot_loss_curve",
    "plot_attention_heatmap",
    "plot_attention_comparison",
    "plot_head_importance",
    "plot_attention_entropy",
    "ResearchChartGenerator",
    "LiveVisualizer",
    "visualize_attack_progress",
    "display_subspace_analysis",
    "RealTimeFlowViz",
    "print_flow_diagram",
    "get_transformer_internals_html",
    "get_transformer_attack_html",
    "get_detailed_transformer_html",
    "get_flow_graph_html",
    "SignatureVisualizer",
    "ComprehensiveVisualizer",
    "plot_cost_efficiency",
    "plot_cost_breakdown",
    "generate_cost_report",
]

