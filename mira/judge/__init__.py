"""
MIRA Attack Success Judge System.

Multi-model ensemble approach for judging attack success:
- ML-based classifiers (DistilBERT, Toxic classifiers)
- One-Class SVM for anomaly detection
- Committee voting for robust decisions

No API calls required - all local CPU inference.

Usage Examples:
    # Use preset configuration
    from mira.judge import create_judge_from_preset
    judge = create_judge_from_preset("balanced")
    judge.load_models()
    result = judge.judge("I cannot help with that.")
    
    # Custom configuration
    from mira.judge import create_custom_judge
    judge = create_custom_judge(
        use_ml=True,
        use_patterns=True,
        use_svm=False,
        use_heuristics=True,
        min_votes=2,
    )
    
    # Direct usage
    from mira.judge import EnsembleJudge
    judge = EnsembleJudge(
        use_ml=True,
        use_patterns=True,
        use_svm=False,
        use_heuristics=True,
    )
"""

from .ml_judge import MLJudge, JudgeResult, JudgeConfig
from .ensemble import EnsembleJudge, EnsembleResult
from .config import (
    create_judge_from_preset,
    create_custom_judge,
    list_presets,
    PRESETS,
)

__all__ = [
    "MLJudge",
    "JudgeResult", 
    "JudgeConfig",
    "EnsembleJudge",
    "EnsembleResult",
    "create_judge_from_preset",
    "create_custom_judge",
    "list_presets",
    "PRESETS",
]

