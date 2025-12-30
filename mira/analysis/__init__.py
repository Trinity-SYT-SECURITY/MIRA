"""Analysis module initialization."""

from mira.analysis.subspace import SubspaceAnalyzer, SubspaceResult
from mira.analysis.activation import ActivationAnalyzer
from mira.analysis.attention import AttentionAnalyzer
from mira.analysis.flow_tracer import AttackFlowTracer, FlowTrace
from mira.analysis.transformer_tracer import (
    TransformerTracer,
    TransformerTrace,
    LayerTrace,
    analyze_attack_patterns,
)

# New mechanistic interpretability tools
from mira.analysis.logit_lens import (
    LogitProjector,
    PredictionTrajectory,
    LogitLensVisualizer,
    run_logit_lens_analysis,
)
from mira.analysis.uncertainty import (
    UncertaintyAnalyzer,
    GenerationTracker,
    RiskDetector,
    analyze_generation_uncertainty,
)
from mira.analysis.comparison import (
    MultiModelRunner,
    ModelConfig,
    ComparisonReport,
    get_recommended_models,
    download_comparison_models,
    COMPARISON_MODELS,
)
from mira.analysis.reverse_search import (
    ReverseActivationSearch,
    SSROptimizer,
    extract_refusal_direction,
)

__all__ = [
    # Existing
    "SubspaceAnalyzer",
    "SubspaceResult",
    "ActivationAnalyzer",
    "AttentionAnalyzer",
    "AttackFlowTracer",
    "FlowTrace",
    "TransformerTracer",
    "TransformerTrace",
    "LayerTrace",
    "analyze_attack_patterns",
    # Logit Lens
    "LogitProjector",
    "PredictionTrajectory",
    "LogitLensVisualizer",
    "run_logit_lens_analysis",
    # Uncertainty
    "UncertaintyAnalyzer",
    "GenerationTracker",
    "RiskDetector",
    "analyze_generation_uncertainty",
    # Multi-Model Comparison
    "MultiModelRunner",
    "ModelConfig",
    "ComparisonReport",
    "get_recommended_models",
    "download_comparison_models",
    "COMPARISON_MODELS",
    # Reverse Search
    "ReverseActivationSearch",
    "SSROptimizer",
    "extract_refusal_direction",
]
