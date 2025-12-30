"""
Centralized model management for MIRA.

Handles model storage, downloading, and loading with unified directory structure.
All models are stored in project/models/ directory.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys


# ============================================================
# MODEL REGISTRY - Role assignments and recommendations
# ============================================================
# Roles: target (被攻擊), judge (評估), attacker (攻擊者)
# recommended: True = CPU-friendly and well-tested

MODEL_REGISTRY = {
    # ========== TARGET MODELS (被攻擊的模型) ==========
    "HuggingFaceTB/SmolLM2-135M-Instruct": {
        "local_name": "smollm2-135m",
        "role": "target",
        "recommended": True,
        "size": "135M",
        "description": "Ultra-lightweight, good for baseline testing",
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "local_name": "smollm2-1.7b",
        "role": "target",
        "recommended": True,
        "size": "1.7B",
        "description": "Mid-size SmolLM, good CPU performance",
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "local_name": "tinyllama-1.1b",
        "role": "target",
        "recommended": True,
        "size": "1.1B",
        "description": "LLaMA-based, great for mechanistic analysis",
    },
    "gpt2": {
        "local_name": "gpt2",
        "role": "target",
        "recommended": False,
        "size": "117M",
        "description": "Classic baseline model",
    },
    "gpt2-medium": {
        "local_name": "gpt2-medium",
        "role": "target",
        "recommended": False,
        "size": "345M",
        "description": "Medium GPT-2 variant",
    },
    "distilgpt2": {
        "local_name": "distilgpt2",
        "role": "target",
        "recommended": False,
        "size": "82M",
        "description": "Distilled GPT-2, fast but limited",
    },
    "EleutherAI/pythia-160m": {
        "local_name": "EleutherAI--pythia-160m",
        "role": "target",
        "recommended": False,
        "size": "160M",
        "description": "Pythia series, good for interpretability",
    },
    "EleutherAI/pythia-70m": {
        "local_name": "EleutherAI--pythia-70m",
        "role": "target",
        "recommended": False,
        "size": "70M",
        "description": "Smallest Pythia, ultra-fast",
    },
    "Qwen/Qwen2-0.5B": {
        "local_name": "Qwen--Qwen2-0.5B",
        "role": "target",
        "recommended": False,
        "size": "0.5B",
        "description": "Alibaba Qwen2, multilingual",
    },
    
    # ========== JUDGE MODELS (評估模型) ==========
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "local_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "judge",
        "recommended": True,
        "size": "66M",
        "description": "Attack success classifier",
    },
    "unitary/toxic-bert": {
        "local_name": "unitary--toxic-bert",
        "role": "judge",
        "recommended": True,
        "size": "110M",
        "description": "Toxic content detector",
    },
    
    # ========== ATTACKER MODELS (攻擊者模型) ==========
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "local_name": "smollm2-1.7b",
        "role": "attacker",  # Can also be attacker
        "recommended": True,
        "size": "1.7B",
        "description": "Can generate attack prompts",
    },
}


def get_recommended_models(role: str = None) -> List[Dict[str, Any]]:
    """
    Get recommended models, optionally filtered by role.
    
    Args:
        role: Filter by role ('target', 'judge', 'attacker', or None for all)
        
    Returns:
        List of model info dicts
    """
    models = []
    for hf_name, info in MODEL_REGISTRY.items():
        if role is None or info.get("role") == role:
            models.append({
                "hf_name": hf_name,
                **info
            })
    return models


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get info for a specific model from registry."""
    # Try exact match
    if model_name in MODEL_REGISTRY:
        return {"hf_name": model_name, **MODEL_REGISTRY[model_name]}
    
    # Try local name match
    for hf_name, info in MODEL_REGISTRY.items():
        if info.get("local_name") == model_name:
            return {"hf_name": hf_name, **info}
    
    return None


def is_model_recommended(model_name: str) -> bool:
    """Check if model is recommended."""
    info = get_model_info(model_name)
    return info.get("recommended", False) if info else False


class ModelManager:
    """
    Centralized model manager for MIRA.
    
    Manages model storage location, downloading, and loading.
    First-run setup asks user where to store models.
    """
    
    def __init__(self):
        # Go up 3 levels: model_manager.py -> utils -> mira -> project_root
        self.project_root = Path(__file__).parent.parent.parent
        self.config_file = self.project_root / ".mira_config.json"
        self.default_models_dir = self.project_root / "project" / "models"
        
        # Load or create config
        self.config = self._load_config()
        self.models_dir = Path(self.config.get("models_directory", str(self.default_models_dir)))
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MIRA configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_config(self):
        """Save MIRA configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def setup_models_directory(self, interactive: bool = True) -> Path:
        """
        Setup models directory on first run.
        
        Args:
            interactive: If True, ask user for directory location
            
        Returns:
            Path to models directory
        """
        # Check if already configured
        if "models_directory" in self.config and Path(self.config["models_directory"]).exists():
            return Path(self.config["models_directory"])
        
        if not interactive:
            # Non-interactive: use default
            self.models_dir = self.default_models_dir
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.config["models_directory"] = str(self.models_dir)
            self._save_config()
            return self.models_dir
        
        # Interactive setup
        print("\n" + "="*70)
        print("  MIRA FIRST-RUN SETUP")
        print("="*70)
        print("""
  MIRA needs to download and store language models for testing.
  
  Default location: project/models/
  
  You can:
    [1] Use default location (recommended)
    [2] Specify custom directory
    [3] Use HuggingFace cache (~/.cache/huggingface)
""")
        
        try:
            choice = input("  Select option (1-3, default: 1): ").strip()
            if not choice:
                choice = "1"
        except:
            choice = "1"
        
        if choice == "1":
            # Use default
            models_dir = self.default_models_dir
        elif choice == "2":
            # Custom directory
            custom_path = input("  Enter directory path: ").strip()
            if custom_path:
                models_dir = Path(custom_path).expanduser().absolute()
            else:
                models_dir = self.default_models_dir
        elif choice == "3":
            # Use HuggingFace cache
            models_dir = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            models_dir = self.default_models_dir
        
        # Create directory
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config["models_directory"] = str(models_dir)
        self._save_config()
        self.models_dir = models_dir
        
        print(f"\n  ✓ Models will be stored in: {models_dir}")
        print()
        
        return models_dir
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Get path for a specific model.
        
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'EleutherAI/pythia-160m')
            
        Returns:
            Path where model should be stored
        """
        # Convert model name to directory name
        # e.g., "EleutherAI/pythia-160m" -> "EleutherAI--pythia-160m"
        dir_name = model_name.replace("/", "--")
        return self.models_dir / dir_name
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if model is already downloaded."""
        model_path = self.get_model_path(model_name)
        
        # Check for key files
        if not model_path.exists():
            return False
        
        # Check for config.json (all models should have this)
        config_file = model_path / "config.json"
        return config_file.exists()
    
    def download_model(
        self, 
        model_name: str,
        force: bool = False,
        verbose: bool = True
    ) -> bool:
        """
        Download a model from HuggingFace.
        
        Args:
            model_name: HuggingFace model name
            force: Force re-download even if exists
            verbose: Print progress
            
        Returns:
            True if successful
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Check if already downloaded
        if not force and self.is_model_downloaded(model_name):
            if verbose:
                print(f"  ✓ {model_name} already downloaded")
            return True
        
        model_path = self.get_model_path(model_name)
        
        if verbose:
            print(f"  Downloading {model_name}...")
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(model_path),
            )
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(model_path),
            )
            
            # Save to local directory
            tokenizer.save_pretrained(str(model_path))
            model.save_pretrained(str(model_path))
            
            if verbose:
                print(f"    ✓ {model_name} downloaded to {model_path}")
            
            # Clean up
            del model, tokenizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed to download {model_name}: {e}")
            return False
    
    def download_batch(
        self,
        model_names: List[str],
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Download multiple models.
        
        Args:
            model_names: List of HuggingFace model names
            verbose: Print progress
            
        Returns:
            Dict mapping model name to success status
        """
        results = {}
        
        if verbose:
            print(f"\n  Downloading {len(model_names)} models to: {self.models_dir}\n")
        
        for i, model_name in enumerate(model_names):
            if verbose:
                print(f"  [{i+1}/{len(model_names)}] {model_name}")
            
            success = self.download_model(model_name, verbose=verbose)
            results[model_name] = success
        
        if verbose:
            successful = sum(1 for v in results.values() if v)
            print(f"\n  ✓ Downloaded {successful}/{len(model_names)} models")
        
        return results
    
    def load_model(self, model_name: str, device: str = "auto"):
        """
        Load a model from local storage or HuggingFace cache.
        
        Priority:
        1. Check project/models/ directory
        2. Fallback to HuggingFace cache
        3. Download if not found anywhere
        
        Args:
            model_name: HuggingFace model name
            device: Device to load on ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Check if model exists in project/models
        model_path = self.get_model_path(model_name)
        
        # Try loading from project/models first
        if self.is_model_downloaded(model_name):
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    dtype=torch.float32,
                    attn_implementation="eager",  # Enable output_attentions
                )
                print(f"  ✓ Loaded {model_name} from project/models")
            except Exception as e:
                print(f"  ⚠️  Failed to load from project/models: {e}")
                print(f"  → Trying HuggingFace cache...")
                model, tokenizer = self._load_from_hf_cache(model_name)
        else:
            # Try HuggingFace cache
            print(f"  Model {model_name} not in project/models")
            print(f"  → Checking HuggingFace cache...")
            
            try:
                model, tokenizer = self._load_from_hf_cache(model_name)
                print(f"  ✓ Loaded from HuggingFace cache")
                
                # Ask if user wants to copy to project/models
                print(f"\n  Would you like to copy this model to project/models for faster access?")
                print(f"  (This will not affect the HuggingFace cache)")
                try:
                    copy_choice = input("  Copy to project/models? (y/n, default: n): ").strip().lower()
                    if copy_choice == 'y':
                        print(f"  Copying {model_name} to project/models...")
                        self.download_model(model_name, force=False, verbose=True)
                except:
                    pass
                
            except Exception as e:
                print(f"  ⚠️  Not found in HuggingFace cache")
                print(f"  → Downloading to project/models...")
                self.download_model(model_name)
                
                # Load from newly downloaded location
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    dtype=torch.float32,
                    attn_implementation="eager",  # Enable output_attentions
                )
        
        model = model.to(device)
        model.eval()
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _load_from_hf_cache(self, model_name: str):
        """
        Load model directly from HuggingFace cache.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Load from HF cache (will use cache if available)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            attn_implementation="eager",  # Enable output_attentions
        )
        
        return model, tokenizer
    
    def list_downloaded_models(self) -> List[str]:
        """List all downloaded models."""
        if not self.models_dir.exists():
            return []
        
        models = []
        for item in self.models_dir.iterdir():
            if item.is_dir():
                # Check if it's a valid model directory
                if (item / "config.json").exists():
                    # Convert directory name back to model name
                    model_name = item.name.replace("--", "/")
                    models.append(model_name)
        
        return sorted(models)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about model storage."""
        info = {
            "models_directory": str(self.models_dir),
            "downloaded_models": self.list_downloaded_models(),
            "num_models": len(self.list_downloaded_models()),
        }
        
        # Calculate total size
        if self.models_dir.exists():
            total_size = sum(
                f.stat().st_size 
                for f in self.models_dir.rglob('*') 
                if f.is_file()
            )
            info["total_size_gb"] = total_size / (1024**3)
        else:
            info["total_size_gb"] = 0.0
        
        return info


# Global instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def setup_models(interactive: bool = True) -> Path:
    """
    Setup models directory (first-run).
    
    Args:
        interactive: Ask user for directory location
        
    Returns:
        Path to models directory
    """
    manager = get_model_manager()
    return manager.setup_models_directory(interactive=interactive)


def download_required_models(
    model_names: List[str] = None,
    interactive: bool = True
) -> Dict[str, bool]:
    """
    Download required models for MIRA.
    
    Args:
        model_names: List of models to download (default: recommended set)
        interactive: Ask user for confirmation
        
    Returns:
        Dict mapping model name to success status
    """
    manager = get_model_manager()
    
    # Default recommended models
    if model_names is None:
        model_names = [
            "gpt2",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
        ]
    
    if interactive:
        print("\n" + "="*70)
        print("  DOWNLOAD REQUIRED MODELS")
        print("="*70)
        print(f"\n  The following models will be downloaded:\n")
        for name in model_names:
            print(f"    • {name}")
        print(f"\n  Storage location: {manager.models_dir}\n")
        
        try:
            confirm = input("  Continue? (y/n, default: y): ").strip().lower()
            if confirm and confirm != 'y':
                print("  Cancelled.")
                return {}
        except:
            pass
    
    return manager.download_batch(model_names)


def load_model(model_name: str, device: str = "auto"):
    """
    Load a model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    manager = get_model_manager()
    return manager.load_model(model_name, device)
