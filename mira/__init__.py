"""
MIRA: Mechanistic Interpretability Research and Attack Framework

A framework for analyzing LLM internals and developing mechanistic attacks.
"""

__version__ = "0.1.0"
__author__ = "Systematic Research Team"

from mira.core.model_wrapper import ModelWrapper
from mira.core.hook_manager import HookManager

__all__ = ["ModelWrapper", "HookManager", "__version__"]
