"""
Model Selector - Interactive model selection based on system capabilities

Now only shows target models (victim models) that are actually downloaded.
Judge models are automatically configured.
"""

import torch
import platform
import psutil
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    display_name: str
    params: str
    min_ram_gb: int
    min_vram_gb: int
    speed: str  # "fast", "medium", "slow"
    difficulty: str  # "easy", "medium", "hard"
    description: str
    category: str  # "tiny", "small", "medium", "large", "chat"
    local_name: Optional[str] = None  # Local directory name


def get_available_target_models() -> List[ModelInfo]:
    """
    Get list of target models that are actually downloaded in project/models/.
    Only returns models with role='target' from MODEL_REGISTRY.
    """
    from mira.utils.model_manager import get_model_manager, MODEL_REGISTRY, get_model_info
    
    manager = get_model_manager()
    downloaded = manager.list_downloaded_models()
    
    # Filter to only target models (not judge models)
    JUDGE_MODELS = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "unitary/toxic-bert",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
        "distilbert-sst2",
        "toxic-bert",
    ]
    
    # Convert judge model names to local names for filtering
    judge_local_names = set()
    for judge in JUDGE_MODELS:
        info = get_model_info(judge)
        if info:
            judge_local_names.add(info.get("local_name", judge))
        # Also add variations
        judge_local_names.add(judge.replace("/", "--"))
        judge_local_names.add(judge.replace("--", "/"))
        # Add direct local name matches
        if "/" in judge:
            judge_local_names.add(judge.split("/")[-1])
    
    # Filter out judge models and datasets
    target_models = []
    for model_dir in downloaded:
        # Skip judge models and datasets
        if model_dir in judge_local_names or model_dir == "alpaca":
            continue
        
        # Get model info from registry
        info = None
        hf_name_used = None
        for hf_name, reg_info in MODEL_REGISTRY.items():
            if reg_info.get("local_name") == model_dir:
                info = reg_info
                hf_name_used = hf_name
                break
        
        # If not found in registry, try to infer from local name
        if not info:
            # Try common patterns
            if "pythia" in model_dir.lower():
                if "160m" in model_dir:
                    hf_name_used = "EleutherAI/pythia-160m"
                elif "70m" in model_dir:
                    hf_name_used = "EleutherAI/pythia-70m"
            elif "qwen" in model_dir.lower():
                if "2.5-3B" in model_dir or "2.5-3b" in model_dir:
                    hf_name_used = "Qwen/Qwen2.5-3B"
                elif "0.5B" in model_dir or "0.5b" in model_dir:
                    hf_name_used = "Qwen/Qwen2-0.5B"
            elif "gpt2" in model_dir.lower():
                if "medium" in model_dir:
                    hf_name_used = "gpt2-medium"
                elif "distil" in model_dir:
                    hf_name_used = "distilgpt2"
                else:
                    hf_name_used = "gpt2"
            elif "smollm2" in model_dir.lower():
                if "1.7b" in model_dir or "1.7B" in model_dir:
                    hf_name_used = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
                elif "135m" in model_dir or "135M" in model_dir:
                    hf_name_used = "HuggingFaceTB/SmolLM2-135M-Instruct"
            elif "tinyllama" in model_dir.lower():
                hf_name_used = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif "deepseek" in model_dir.lower():
                hf_name_used = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            
            if hf_name_used:
                info = get_model_info(hf_name_used)
        
        if info:
            # Check if this is a target model (support both single role and list of roles)
            model_role = info.get("role")
            is_target = False
            if isinstance(model_role, list):
                is_target = "target" in model_role
            else:
                is_target = model_role == "target"
            
            if is_target:
                # Create ModelInfo from registry
                size = info.get("size", "?")
                recommended = info.get("recommended", False)
                
                # Map size to params
                params = size
                
                # Determine speed and difficulty based on size
                if "M" in size:
                    size_num = float(size.replace("M", ""))
                    if size_num < 200:
                        speed = "fast"
                        difficulty = "easy"
                        min_ram = 2
                    elif size_num < 500:
                        speed = "fast"
                        difficulty = "easy"
                        min_ram = 4
                    else:
                        speed = "medium"
                        difficulty = "easy"
                        min_ram = 4
                elif "B" in size or "b" in size:
                    size_num = float(size.replace("B", "").replace("b", ""))
                    if size_num < 2:
                        speed = "medium"
                        difficulty = "easy"
                        min_ram = 4
                    elif size_num < 5:
                        speed = "medium"
                        difficulty = "medium"
                        min_ram = 8
                    else:
                        speed = "slow"
                        difficulty = "medium"
                        min_ram = 8
                else:
                    speed = "medium"
                    difficulty = "easy"
                    min_ram = 4
                
                # Determine category
                if "M" in size and float(size.replace("M", "")) < 200:
                    category = "tiny"
                elif "M" in size:
                    category = "small"
                elif "B" in size or "b" in size:
                    size_num = float(size.replace("B", "").replace("b", ""))
                    if size_num < 2:
                        category = "small"
                    elif size_num < 5:
                        category = "medium"
                    else:
                        category = "large"
                else:
                    category = "small"
                
                model_info = ModelInfo(
                    name=hf_name_used,
                    display_name=info.get("description", hf_name_used).split(",")[0] if info.get("description") else hf_name_used.split("/")[-1],
                    params=params,
                    min_ram_gb=min_ram,
                    min_vram_gb=0,  # CPU-friendly
                    speed=speed,
                    difficulty=difficulty,
                    description=info.get("description", ""),
                    category=category,
                    local_name=model_dir,
                )
                target_models.append(model_info)
    
    # Sort by size (smallest first for CPU)
    def sort_key(m):
        if "M" in m.params:
            return float(m.params.replace("M", ""))
        elif "B" in m.params or "b" in m.params:
            return float(m.params.replace("B", "").replace("b", "")) * 1000
        else:
            return 999
    
    target_models.sort(key=sort_key)
    
    return target_models


class ModelSelector:
    """Interactive model selector based on system capabilities."""
    
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.has_gpu else None
        self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if self.has_gpu else 0
        self.ram_gb = psutil.virtual_memory().total / 1e9
        self.os_name = platform.system()
    
    def print_system_info(self):
        """Print system capabilities."""
        print("\n" + "="*60)
        print("  SYSTEM DETECTION")
        print("="*60)
        print(f"\n  OS: {self.os_name}")
        print(f"  RAM: {self.ram_gb:.1f} GB")
        
        if self.has_gpu:
            print(f"  GPU: {self.gpu_name}")
            print(f"  VRAM: {self.vram_gb:.1f} GB")
            print(f"  ✓ GPU available - can run larger models")
        else:
            print(f"  GPU: Not available")
            print(f"  ⚠ CPU only - recommend smaller models")
        
        print("="*60 + "\n")
    
    def get_compatible_models(self, available_models: List[ModelInfo]) -> List[ModelInfo]:
        """Get models compatible with current system from available models."""
        compatible = []
        
        for model in available_models:
            # Check RAM
            if model.min_ram_gb > self.ram_gb:
                continue
            
            # Check VRAM (if GPU required)
            if model.min_vram_gb > 0:
                if not self.has_gpu or model.min_vram_gb > self.vram_gb:
                    continue
            
            compatible.append(model)
        
        return compatible
    
    def get_recommended_models(self, available_models: List[ModelInfo]) -> Tuple[List[ModelInfo], List[ModelInfo]]:
        """Get recommended and advanced models from available models."""
        compatible = self.get_compatible_models(available_models)
        
        # Recommended: easy/medium difficulty, smaller models
        recommended = [m for m in compatible if m.difficulty in ["easy", "medium"] and m.category in ["tiny", "small"]]
        
        # Advanced: larger models or hard difficulty
        advanced = [m for m in compatible if m.difficulty == "hard" or m.category in ["medium", "large"]]
        
        return recommended, advanced
    
    def print_model_list(self, models: List[ModelInfo], title: str):
        """Print formatted model list."""
        if not models:
            return
        
        print(f"\n{title}")
        print("-" * 60)
        
        for i, model in enumerate(models, 1):
            # Color coding
            if model.speed == "fast":
                speed_icon = "🟢"
            elif model.speed == "medium":
                speed_icon = "🟡"
            else:
                speed_icon = "🔴"
            
            if model.difficulty == "easy":
                diff_icon = "⭐"
            elif model.difficulty == "medium":
                diff_icon = "⭐⭐"
            else:
                diff_icon = "⭐⭐⭐"
            
            # Show local name if different
            local_info = f" ({model.local_name})" if model.local_name and model.local_name != model.name.split("/")[-1] else ""
            
            print(f"\n  [{i}] {model.display_name} ({model.params}){local_info}")
            print(f"      {speed_icon} Speed: {model.speed.capitalize()}  |  {diff_icon} Difficulty: {model.difficulty.capitalize()}")
            print(f"      💾 RAM: {model.min_ram_gb}GB", end="")
            if model.min_vram_gb > 0:
                print(f"  |  🎮 VRAM: {model.min_vram_gb}GB", end="")
            print()
            if model.description:
                print(f"      📝 {model.description}")
    
    def select_model(self) -> str:
        """Interactive model selection - only shows downloaded target models."""
        self.print_system_info()
        
        # Get available target models from project/models/
        available_models = get_available_target_models()
        
        if not available_models:
            print("❌ No target models found in project/models/!")
            print("   Please download models first (Mode 5)")
            print("   Recommended: smollm2-135m, tinyllama-1.1b, gpt2-medium")
            return "EleutherAI/pythia-70m"  # Fallback
        
        recommended, advanced = self.get_recommended_models(available_models)
        
        # If no recommended, use all compatible
        if not recommended and not advanced:
            compatible = self.get_compatible_models(available_models)
            if compatible:
                recommended = compatible[:3]
                advanced = compatible[3:]
            else:
                print("❌ No compatible models found for your system!")
                print("   Minimum requirement: 2GB RAM")
                return available_models[0].name  # Fallback to first available
        
        # Print recommended models
        if recommended:
            self.print_model_list(recommended, "🎯 RECOMMENDED MODELS (Downloaded)")
        
        # Print advanced models
        if advanced:
            self.print_model_list(advanced, "🚀 ADVANCED MODELS (Downloaded)")
        
        # Get user choice
        all_models = recommended + advanced
        
        print("\n" + "="*60)
        print(f"  Enter number (1-{len(all_models)}) or press Enter for default")
        print("="*60)
        print("\n  💡 Judge models are automatically configured")
        print("     (distilbert, toxic-bert, sentence-transformers)")
        
        while True:
            try:
                choice = input("\n  Your choice: ").strip()
                
                if not choice:
                    # Default: first recommended model
                    default_model = recommended[0] if recommended else all_models[0]
                    print(f"\n  ✓ Using default: {default_model.display_name} ({default_model.name})")
                    return default_model.name
                
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    selected = all_models[idx]
                    print(f"\n  ✓ Selected: {selected.display_name} ({selected.name})")
                    
                    # Warn if challenging
                    if selected.difficulty == "hard":
                        print(f"  ⚠ Warning: This is a safety-aligned model - attacks will be harder!")
                    
                    return selected.name
                else:
                    print(f"  ❌ Invalid choice. Enter 1-{len(all_models)}")
            
            except ValueError:
                print(f"  ❌ Invalid input. Enter a number 1-{len(all_models)}")
            except KeyboardInterrupt:
                print("\n\n  Cancelled. Using default.")
                default_model = recommended[0] if recommended else all_models[0]
                return default_model.name


def select_model_interactive() -> str:
    """
    Main function for interactive model selection.
    Only shows target models that are actually downloaded.
    Judge models are automatically configured.
    
    Returns:
        Model name (HuggingFace identifier)
    """
    selector = ModelSelector()
    return selector.select_model()


if __name__ == "__main__":
    # Test the selector
    model_name = select_model_interactive()
    print(f"\nSelected model: {model_name}")
