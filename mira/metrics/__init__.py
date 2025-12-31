"""Metrics module initialization."""

from mira.metrics.success_rate import compute_asr, AttackSuccessEvaluator
from mira.metrics.distance import SubspaceDistanceMetrics
from mira.metrics.probability import ProbabilityMetrics
from mira.metrics.roc_pr import AttackSuccessCurves, compute_attack_success_curves
from mira.metrics.cost_tracker import AttackCostTracker, CostTrackerContext

# Three-tier ASR metrics
from mira.metrics.tiered_asr import (
    TieredASRCalculator,
    TieredASRResult,
    evaluate_tiered_asr,
)

# Feature extraction for heatmaps
from mira.metrics.feature_heatmap import (
    FeatureExtractor,
    ExtractedFeatures,
    LayerFeatures,
)

__all__ = [
    "compute_asr",
    "AttackSuccessEvaluator",
    "SubspaceDistanceMetrics",
    "ProbabilityMetrics",
    "AttackSuccessCurves",
    "compute_attack_success_curves",
    "AttackCostTracker",
    "CostTrackerContext",
    # Three-tier ASR
    "TieredASRCalculator",
    "TieredASRResult",
    "evaluate_tiered_asr",
    # Feature heatmap
    "FeatureExtractor",
    "ExtractedFeatures",
    "LayerFeatures",
]

