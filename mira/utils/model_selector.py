"""
Model Selector - Interactive model selection based on system capabilities

Detects GPU availability and recommends appropriate models.
"""

import torch
import platform
import psutil
from typing import List, Dict, Tuple
from dataclasses import dataclass


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


# Model catalog
MODEL_CATALOG = [
    # Tiny models (CPU friendly)
    ModelInfo(
        name="EleutherAI/pythia-70m",
        display_name="Pythia 70M",
        params="70M",
        min_ram_gb=2,
        min_vram_gb=0,
        speed="fast",
        difficulty="easy",
        description="Fastest, best for quick testing",
        category="tiny"
    ),
    ModelInfo(
        name="EleutherAI/pythia-160m",
        display_name="Pythia 160M",
        params="160M",
        min_ram_gb=2,
        min_vram_gb=0,
        speed="fast",
        difficulty="easy",
        description="Small but capable",
        category="tiny"
    ),
    
    # Small models
    ModelInfo(
        name="gpt2",
        display_name="GPT-2",
        params="124M",
        min_ram_gb=2,
        min_vram_gb=0,
        speed="fast",
        difficulty="easy",
        description="Classic baseline model",
        category="small"
    ),
    ModelInfo(
        name="EleutherAI/pythia-410m",
        display_name="Pythia 410M",
        params="410M",
        min_ram_gb=4,
        min_vram_gb=0,
        speed="medium",
        difficulty="easy",
        description="Good balance for CPU",
        category="small"
    ),
    ModelInfo(
        name="gpt2-medium",
        display_name="GPT-2 Medium",
        params="355M",
        min_ram_gb=4,
        min_vram_gb=0,
        speed="medium",
        difficulty="easy",
        description="Larger GPT-2 variant",
        category="small"
    ),
    
    # Medium models (GPU recommended)
    ModelInfo(
        name="EleutherAI/pythia-1b",
        display_name="Pythia 1B",
        params="1B",
        min_ram_gb=8,
        min_vram_gb=4,
        speed="medium",
        difficulty="medium",
        description="Research standard",
        category="medium"
    ),
    ModelInfo(
        name="EleutherAI/pythia-1.4b",
        display_name="Pythia 1.4B",
        params="1.4B",
        min_ram_gb=8,
        min_vram_gb=6,
        speed="medium",
        difficulty="medium",
        description="Best balance for GPU",
        category="medium"
    ),
    ModelInfo(
        name="EleutherAI/gpt-neo-1.3B",
        display_name="GPT-Neo 1.3B",
        params="1.3B",
        min_ram_gb=8,
        min_vram_gb=6,
        speed="medium",
        difficulty="medium",
        description="Alternative architecture",
        category="medium"
    ),
    
    # Large models (Strong GPU required)
    ModelInfo(
        name="EleutherAI/pythia-2.8b",
        display_name="Pythia 2.8B",
        params="2.8B",
        min_ram_gb=16,
        min_vram_gb=8,
        speed="slow",
        difficulty="medium",
        description="Larger research model",
        category="large"
    ),
    ModelInfo(
        name="EleutherAI/gpt-j-6b",
        display_name="GPT-J 6B",
        params="6B",
        min_ram_gb=16,
        min_vram_gb=12,
        speed="slow",
        difficulty="medium",
        description="Popular research model",
        category="large"
    ),
    ModelInfo(
        name="EleutherAI/pythia-6.9b",
        display_name="Pythia 6.9B",
        params="6.9B",
        min_ram_gb=16,
        min_vram_gb=14,
        speed="slow",
        difficulty="hard",
        description="Large research model",
        category="large"
    ),
    
    # Chat models (Safety aligned)
    ModelInfo(
        name="meta-llama/Llama-2-7b-chat-hf",
        display_name="Llama 2 Chat 7B",
        params="7B",
        min_ram_gb=16,
        min_vram_gb=16,
        speed="slow",
        difficulty="hard",
        description="Safety-aligned, hardest to attack",
        category="chat"
    ),
    ModelInfo(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        display_name="Mistral 7B Instruct",
        params="7B",
        min_ram_gb=16,
        min_vram_gb=16,
        speed="slow",
        difficulty="hard",
        description="Modern instruction-tuned model",
        category="chat"
    ),
]


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
    
    def get_compatible_models(self) -> List[ModelInfo]:
        """Get models compatible with current system."""
        compatible = []
        
        for model in MODEL_CATALOG:
            # Check RAM
            if model.min_ram_gb > self.ram_gb:
                continue
            
            # Check VRAM (if GPU required)
            if model.min_vram_gb > 0:
                if not self.has_gpu or model.min_vram_gb > self.vram_gb:
                    continue
            
            compatible.append(model)
        
        return compatible
    
    def get_recommended_models(self) -> Tuple[List[ModelInfo], List[ModelInfo]]:
        """Get recommended and advanced models."""
        compatible = self.get_compatible_models()
        
        # Recommended: easy/medium difficulty
        recommended = [m for m in compatible if m.difficulty in ["easy", "medium"]]
        
        # Advanced: hard difficulty or large models
        advanced = [m for m in compatible if m.difficulty == "hard" or m.category == "large"]
        
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
            
            print(f"\n  [{i}] {model.display_name} ({model.params})")
            print(f"      {speed_icon} Speed: {model.speed.capitalize()}  |  {diff_icon} Difficulty: {model.difficulty.capitalize()}")
            print(f"      💾 RAM: {model.min_ram_gb}GB", end="")
            if model.min_vram_gb > 0:
                print(f"  |  🎮 VRAM: {model.min_vram_gb}GB", end="")
            print()
            print(f"      📝 {model.description}")
    
    def select_model(self) -> str:
        """Interactive model selection."""
        self.print_system_info()
        
        recommended, advanced = self.get_recommended_models()
        
        if not recommended and not advanced:
            print("❌ No compatible models found for your system!")
            print("   Minimum requirement: 2GB RAM")
            return "EleutherAI/pythia-70m"  # Fallback
        
        # Print recommended models
        self.print_model_list(recommended, "🎯 RECOMMENDED MODELS")
        
        # Print advanced models
        if advanced:
            self.print_model_list(advanced, "🚀 ADVANCED MODELS (Harder to attack)")
        
        # Get user choice
        all_models = recommended + advanced
        
        print("\n" + "="*60)
        print(f"  Enter number (1-{len(all_models)}) or press Enter for default")
        print("="*60)
        
        while True:
            try:
                choice = input("\n  Your choice: ").strip()
                
                if not choice:
                    # Default: first recommended model
                    default_model = recommended[0] if recommended else all_models[0]
                    print(f"\n  ✓ Using default: {default_model.display_name}")
                    return default_model.name
                
                idx = int(choice) - 1
                if 0 <= idx < len(all_models):
                    selected = all_models[idx]
                    print(f"\n  ✓ Selected: {selected.display_name}")
                    
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
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get info for a specific model."""
        for model in MODEL_CATALOG:
            if model.name == model_name:
                return model
        
        # Return default if not found
        return MODEL_CATALOG[0]


def select_model_interactive() -> str:
    """
    Main function for interactive model selection.
    
    Returns:
        Model name (HuggingFace identifier)
    """
    selector = ModelSelector()
    return selector.select_model()


if __name__ == "__main__":
    # Test the selector
    model_name = select_model_interactive()
    print(f"\nSelected model: {model_name}")
