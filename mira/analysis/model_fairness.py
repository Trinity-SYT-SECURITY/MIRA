"""
Multi-Model Fairness and Reproducibility Analysis.

Analyzes fairness across different model architectures, sizes, and hardware configurations:
- Architecture comparison (Transformer variants)
- Parameter count comparison
- Reproducibility across hardware/random seeds
- Fairness metrics (ASR variance, consistency)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class ModelFairnessMetrics:
    """Fairness metrics for a model."""
    model_name: str
    architecture: str
    parameter_count: Optional[int] = None
    asr_mean: float = 0.0
    asr_std: float = 0.0
    asr_ci_95: Tuple[float, float] = (0.0, 0.0)
    reproducibility_score: float = 0.0
    consistency_score: float = 0.0
    hardware_variance: float = 0.0


class ModelFairnessAnalyzer:
    """Analyze fairness and reproducibility across models."""
    
    def __init__(self):
        """Initialize fairness analyzer."""
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
    
    def extract_model_architecture(self, model_name: str) -> str:
        """
        Extract architecture type from model name.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Architecture type (e.g., "GPT", "LLaMA", "Qwen", "Gemma")
        """
        name_lower = model_name.lower()
        
        if "gpt" in name_lower or "gpt2" in name_lower:
            return "GPT"
        elif "llama" in name_lower or "llama2" in name_lower or "llama3" in name_lower:
            return "LLaMA"
        elif "qwen" in name_lower or "qwen2" in name_lower:
            return "Qwen"
        elif "gemma" in name_lower:
            return "Gemma"
        elif "mistral" in name_lower:
            return "Mistral"
        elif "phi" in name_lower:
            return "Phi"
        elif "tinyllama" in name_lower:
            return "TinyLlama"
        elif "smollm" in name_lower:
            return "SmolLM"
        else:
            return "Unknown"
    
    def extract_parameter_count(self, model_name: str) -> Optional[int]:
        """
        Extract parameter count from model name or metadata.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Parameter count in millions, or None if unknown
        """
        name_lower = model_name.lower()
        
        # Try to extract from name patterns
        if "0.5b" in name_lower or "500m" in name_lower:
            return 500
        elif "1b" in name_lower or "1.1b" in name_lower:
            return 1100
        elif "1.7b" in name_lower:
            return 1700
        elif "2b" in name_lower:
            return 2000
        elif "3b" in name_lower:
            return 3000
        elif "7b" in name_lower:
            return 7000
        elif "13b" in name_lower:
            return 13000
        elif "70b" in name_lower:
            return 70000
        elif "160m" in name_lower:
            return 160
        elif "135m" in name_lower:
            return 135
        
        return None
    
    def analyze_architecture_comparison(
        self,
        all_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare ASR across different architectures.
        
        Args:
            all_results: Complete results from all models
            
        Returns:
            Dict with architecture-wise statistics
        """
        architecture_stats = defaultdict(lambda: {"asrs": [], "models": []})
        
        for result in all_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            asr = result.get("asr", 0.0)
            
            arch = self.extract_model_architecture(model_name)
            architecture_stats[arch]["asrs"].append(asr)
            architecture_stats[arch]["models"].append(model_name)
        
        # Compute statistics per architecture
        arch_comparison = {}
        for arch, data in architecture_stats.items():
            asrs = data["asrs"]
            if asrs:
                arch_comparison[arch] = {
                    "mean_asr": float(np.mean(asrs)),
                    "std_asr": float(np.std(asrs)),
                    "min_asr": float(np.min(asrs)),
                    "max_asr": float(np.max(asrs)),
                    "num_models": len(data["models"]),
                    "models": data["models"],
                    "ci_95": (
                        float(np.mean(asrs) - 1.96 * np.std(asrs) / np.sqrt(len(asrs))),
                        float(np.mean(asrs) + 1.96 * np.std(asrs) / np.sqrt(len(asrs))),
                    ) if len(asrs) > 1 else (float(np.mean(asrs)), float(np.mean(asrs))),
                }
        
        return arch_comparison
    
    def analyze_parameter_size_comparison(
        self,
        all_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare ASR across different parameter sizes.
        
        Args:
            all_results: Complete results from all models
            
        Returns:
            Dict with size-wise statistics
        """
        size_groups = {
            "Small (<1B)": [],
            "Medium (1B-7B)": [],
            "Large (>7B)": [],
        }
        
        for result in all_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            asr = result.get("asr", 0.0)
            param_count = self.extract_parameter_count(model_name)
            
            if param_count is None:
                continue
            
            if param_count < 1000:
                size_groups["Small (<1B)"].append({"model": model_name, "asr": asr, "params": param_count})
            elif param_count < 7000:
                size_groups["Medium (1B-7B)"].append({"model": model_name, "asr": asr, "params": param_count})
            else:
                size_groups["Large (>7B)"].append({"model": model_name, "asr": asr, "params": param_count})
        
        # Compute statistics per size group
        size_comparison = {}
        for size_group, models in size_groups.items():
            if models:
                asrs = [m["asr"] for m in models]
                size_comparison[size_group] = {
                    "mean_asr": float(np.mean(asrs)),
                    "std_asr": float(np.std(asrs)),
                    "num_models": len(models),
                    "models": [m["model"] for m in models],
                    "avg_params": float(np.mean([m["params"] for m in models])),
                }
        
        return size_comparison
    
    def compute_reproducibility_score(
        self,
        model_results: List[Dict[str, Any]],
        num_runs: int = 1,
    ) -> float:
        """
        Compute reproducibility score based on variance across runs.
        
        Args:
            model_results: Results from multiple runs of the same model
            num_runs: Number of runs (for normalization)
            
        Returns:
            Reproducibility score (0-1, higher = more reproducible)
        """
        if not model_results or len(model_results) < 2:
            return 1.0  # Perfect if only one run
        
        asrs = [r.get("asr", 0.0) for r in model_results if r.get("success", False)]
        
        if len(asrs) < 2:
            return 1.0
        
        # Reproducibility = 1 - normalized variance
        variance = float(np.var(asrs))
        max_variance = 0.25  # Maximum possible variance for ASR (0-1 range)
        normalized_variance = min(variance / max_variance, 1.0)
        
        return float(1.0 - normalized_variance)
    
    def analyze_fairness_metrics(
        self,
        all_results: List[Dict[str, Any]],
    ) -> List[ModelFairnessMetrics]:
        """
        Compute comprehensive fairness metrics for each model.
        
        Args:
            all_results: Complete results from all models
            
        Returns:
            List of ModelFairnessMetrics
        """
        fairness_metrics = []
        
        # Group results by model
        model_groups = defaultdict(list)
        for result in all_results:
            if result.get("success", False):
                model_name = result.get("model_name", "unknown")
                model_groups[model_name].append(result)
        
        for model_name, results in model_groups.items():
            asrs = [r.get("asr", 0.0) for r in results]
            
            if not asrs:
                continue
            
            mean_asr = float(np.mean(asrs))
            std_asr = float(np.std(asrs)) if len(asrs) > 1 else 0.0
            
            # 95% confidence interval
            if len(asrs) > 1:
                se = std_asr / np.sqrt(len(asrs))
                ci_95 = (float(mean_asr - 1.96 * se), float(mean_asr + 1.96 * se))
            else:
                ci_95 = (mean_asr, mean_asr)
            
            # Reproducibility score
            reproducibility = self.compute_reproducibility_score(results)
            
            # Consistency score (inverse of coefficient of variation)
            consistency = 1.0 - min(std_asr / (mean_asr + 1e-10), 1.0) if mean_asr > 0 else 1.0
            
            # Hardware variance (placeholder - would need actual hardware info)
            hardware_variance = 0.0
            
            metrics = ModelFairnessMetrics(
                model_name=model_name,
                architecture=self.extract_model_architecture(model_name),
                parameter_count=self.extract_parameter_count(model_name),
                asr_mean=mean_asr,
                asr_std=std_asr,
                asr_ci_95=ci_95,
                reproducibility_score=reproducibility,
                consistency_score=consistency,
                hardware_variance=hardware_variance,
            )
            
            fairness_metrics.append(metrics)
        
        return fairness_metrics

