"""Utility module initialization."""

from mira.utils.logging import setup_logger, get_logger
from mira.utils.data import load_harmful_prompts, load_safe_prompts
from mira.utils.environment import (
    detect_environment,
    detect_gpu,
    detect_system,
    print_environment_info,
    get_device_string,
)
from mira.utils.experiment_logger import ExperimentLogger

# Baseline loader
try:
    from mira.utils.baseline_loader import BaselineLoader, load_baseline_prompts
    BASELINE_LOADER_AVAILABLE = True
except ImportError:
    BASELINE_LOADER_AVAILABLE = False
    BaselineLoader = None
    load_baseline_prompts = None

__all__ = [
    "setup_logger",
    "get_logger",
    "load_harmful_prompts",
    "load_safe_prompts",
    "detect_environment",
    "detect_gpu",
    "detect_system",
    "print_environment_info",
    "get_device_string",
    "ExperimentLogger",
    "BaselineLoader",
    "load_baseline_prompts",
]
