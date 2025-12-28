"""
Configuration loader for MIRA framework.

Loads configuration from YAML files and provides runtime access.
All parameters are externalized - no hardcoded values in code.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import yaml
import torch


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Look for config.yaml in framework directory
        framework_dir = Path(__file__).parent.parent
        config_path = framework_dir / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return defaults if no config file
        return get_default_config_dict()
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config_dict() -> Dict[str, Any]:
    """Get default configuration as dictionary."""
    return {
        "model": {
            "name": "EleutherAI/pythia-70m",
            "device": "auto",
            "dtype": "float32",
            "cache_dir": None,
            "trust_remote_code": False,
            "quantization": {
                "enabled": False,
                "method": None,
                "quantized_model_path": None,
            }
        },
        "analysis": {
            "n_components": 64,
            "reduction_method": "pca",
            "layers_to_analyze": None,
            "token_position": "last",
            "include_residual": True,
            "include_mlp": True,
            "include_attention": True,
            "batch_size": 8,
            "max_seq_length": 512,
        },
        "attack": {
            "num_steps": 100,
            "learning_rate": 0.01,
            "batch_size": 512,
            "top_k": 256,
            "suffix_length": 20,
            "suffix_init": "exclamation",
            "custom_suffix_init": None,
            "target_weight": 1.0,
            "distance_weight": 0.1,
            "fluency_weight": 0.01,
            "early_stop": True,
            "patience": 20,
            "min_improvement": 0.0001,
        },
        "evaluation": {
            "success_threshold": 0.5,
            "ambiguous_response_length": 50,
            "unclear_response_length": 30,
            "refusal_patterns": [],
            "acceptance_patterns": [],
            "affirmative_targets": ["Sure", "Here", "Certainly", "Of"],
        },
        "visualization": {
            "figure_size_2d": [10, 8],
            "figure_size_3d": [12, 10],
            "dpi": 150,
            "save_format": "png",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_file": None,
        }
    }


def detect_device() -> str:
    """Detect available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    
    name: str = "EleutherAI/pythia-70m"
    device: str = "auto"
    dtype: str = "float32"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    quantization_enabled: bool = False
    quantization_method: Optional[str] = None
    quantized_model_path: Optional[str] = None
    
    def get_device(self) -> str:
        """Get actual device string."""
        if self.device == "auto":
            return detect_device()
        return self.device
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        return get_dtype(self.dtype)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Create from config dictionary."""
        model_cfg = config.get("model", {})
        quantization = model_cfg.get("quantization", {})
        
        return cls(
            name=model_cfg.get("name", "EleutherAI/pythia-70m"),
            device=model_cfg.get("device", "auto"),
            dtype=model_cfg.get("dtype", "float32"),
            cache_dir=model_cfg.get("cache_dir"),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            quantization_enabled=quantization.get("enabled", False),
            quantization_method=quantization.get("method"),
            quantized_model_path=quantization.get("quantized_model_path"),
        )


@dataclass
class AnalysisConfig:
    """Configuration for mechanistic analysis."""
    
    n_components: int = 64
    reduction_method: str = "pca"
    layers_to_analyze: Optional[List[int]] = None
    token_position: str = "last"
    include_residual: bool = True
    include_mlp: bool = True
    include_attention: bool = True
    batch_size: int = 8
    max_seq_length: int = 512
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AnalysisConfig":
        """Create from config dictionary."""
        analysis_cfg = config.get("analysis", {})
        return cls(**{k: v for k, v in analysis_cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class AttackConfig:
    """Configuration for attack optimization."""
    
    num_steps: int = 100
    learning_rate: float = 0.01
    batch_size: int = 512
    top_k: int = 256
    suffix_length: int = 20
    suffix_init: str = "exclamation"
    custom_suffix_init: Optional[str] = None
    target_weight: float = 1.0
    distance_weight: float = 0.1
    fluency_weight: float = 0.01
    early_stop: bool = True
    patience: int = 20
    min_improvement: float = 0.0001
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AttackConfig":
        """Create from config dictionary."""
        attack_cfg = config.get("attack", {})
        return cls(**{k: v for k, v in attack_cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class EvaluationConfig:
    """Configuration for attack evaluation."""
    
    success_threshold: float = 0.5
    ambiguous_response_length: int = 50
    unclear_response_length: int = 30
    refusal_patterns: List[str] = field(default_factory=list)
    acceptance_patterns: List[str] = field(default_factory=list)
    affirmative_targets: List[str] = field(default_factory=lambda: ["Sure", "Here", "Certainly"])
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EvaluationConfig":
        """Create from config dictionary."""
        eval_cfg = config.get("evaluation", {})
        return cls(
            success_threshold=eval_cfg.get("success_threshold", 0.5),
            ambiguous_response_length=eval_cfg.get("ambiguous_response_length", 50),
            unclear_response_length=eval_cfg.get("unclear_response_length", 30),
            refusal_patterns=eval_cfg.get("refusal_patterns", []),
            acceptance_patterns=eval_cfg.get("acceptance_patterns", []),
            affirmative_targets=eval_cfg.get("affirmative_targets", ["Sure", "Here", "Certainly"]),
        )


@dataclass
class MiraConfig:
    """Complete MIRA framework configuration."""
    
    model: ModelConfig
    analysis: AnalysisConfig
    attack: AttackConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "MiraConfig":
        """Load configuration from file."""
        config_dict = load_config(config_path)
        
        return cls(
            model=ModelConfig.from_dict(config_dict),
            analysis=AnalysisConfig.from_dict(config_dict),
            attack=AttackConfig.from_dict(config_dict),
            evaluation=EvaluationConfig.from_dict(config_dict),
        )
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": {
                "name": self.model.name,
                "device": self.model.device,
                "dtype": self.model.dtype,
                "cache_dir": self.model.cache_dir,
                "trust_remote_code": self.model.trust_remote_code,
                "quantization": {
                    "enabled": self.model.quantization_enabled,
                    "method": self.model.quantization_method,
                    "quantized_model_path": self.model.quantized_model_path,
                }
            },
            "analysis": {
                "n_components": self.analysis.n_components,
                "reduction_method": self.analysis.reduction_method,
                "layers_to_analyze": self.analysis.layers_to_analyze,
                "token_position": self.analysis.token_position,
                "batch_size": self.analysis.batch_size,
                "max_seq_length": self.analysis.max_seq_length,
            },
            "attack": {
                "num_steps": self.attack.num_steps,
                "suffix_length": self.attack.suffix_length,
                "suffix_init": self.attack.suffix_init,
                "top_k": self.attack.top_k,
            },
            "evaluation": {
                "refusal_patterns": self.evaluation.refusal_patterns,
                "acceptance_patterns": self.evaluation.acceptance_patterns,
                "affirmative_targets": self.evaluation.affirmative_targets,
            },
        }
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
