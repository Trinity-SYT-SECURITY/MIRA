#!/usr/bin/env python3
"""
Migrate HuggingFace cache models to project/models directory.

This script safely copies models from ~/.cache/huggingface to project/models
without affecting the original cache or other projects.
"""

import os
import shutil
from pathlib import Path
import json


def find_hf_cache_models():
    """Find all models in HuggingFace cache."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not hf_cache.exists():
        print("  HuggingFace cache not found")
        return []
    
    models = []
    for item in hf_cache.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            # Extract model name from directory name
            # e.g., "models--gpt2" -> "gpt2"
            # e.g., "models--EleutherAI--pythia-160m" -> "EleutherAI/pythia-160m"
            model_name = item.name.replace("models--", "").replace("--", "/")
            models.append({
                "name": model_name,
                "cache_path": item,
                "size_mb": sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024**2)
            })
    
    return models


def copy_model_to_project(model_info, project_models_dir):
    """
    Copy a model from HF cache to project/models.
    
    Args:
        model_info: Dict with name, cache_path, size_mb
        project_models_dir: Path to project/models directory
    """
    model_name = model_info["name"]
    cache_path = model_info["cache_path"]
    
    # Create target directory name (replace / with --)
    target_name = model_name.replace("/", "--")
    target_path = project_models_dir / target_name
    
    # Check if already exists
    if target_path.exists():
        print(f"  ⚠️  {model_name} already exists in project/models, skipping")
        return False
    
    print(f"  Copying {model_name} ({model_info['size_mb']:.1f} MB)...")
    
    try:
        # Find the actual model files in snapshots
        snapshots_dir = cache_path / "snapshots"
        if not snapshots_dir.exists():
            print(f"    ✗ No snapshots found for {model_name}")
            return False
        
        # Get the latest snapshot (usually only one)
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            print(f"    ✗ No snapshot directories found")
            return False
        
        latest_snapshot = snapshots[0]  # Use first (usually only) snapshot
        
        # Copy the snapshot contents to target
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from snapshot
        for item in latest_snapshot.iterdir():
            if item.is_file():
                shutil.copy2(item, target_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
        
        print(f"    ✓ Copied to {target_path}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error copying {model_name}: {e}")
        # Clean up partial copy
        if target_path.exists():
            shutil.rmtree(target_path)
        return False


def main():
    print("\n" + "="*70)
    print("  MIGRATE HUGGINGFACE MODELS TO project/models")
    print("="*70 + "\n")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir
    project_models_dir = project_root / "project" / "models"
    
    # Create project/models if not exists
    project_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Find models in HF cache
    print("  Scanning HuggingFace cache...")
    models = find_hf_cache_models()
    
    if not models:
        print("  No models found in HuggingFace cache")
        return
    
    print(f"\n  Found {len(models)} models in HuggingFace cache:\n")
    
    total_size = 0
    for i, model in enumerate(models, 1):
        print(f"    [{i}] {model['name']:<40} {model['size_mb']:>8.1f} MB")
        total_size += model['size_mb']
    
    print(f"\n  Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"  Target directory: {project_models_dir}")
    
    # Ask for confirmation
    print("\n" + "="*70)
    try:
        confirm = input("  Copy all models to project/models? (y/n): ").strip().lower()
        if confirm != 'y':
            print("  Cancelled.")
            return
    except:
        print("  Cancelled.")
        return
    
    print("\n  Starting migration...\n")
    
    # Copy each model
    copied = 0
    skipped = 0
    failed = 0
    
    for model in models:
        success = copy_model_to_project(model, project_models_dir)
        if success:
            copied += 1
        elif (project_models_dir / model['name'].replace("/", "--")).exists():
            skipped += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("  MIGRATION COMPLETE")
    print("="*70)
    print(f"  ✓ Copied: {copied}")
    print(f"  ⚠️  Skipped (already exists): {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(models)}")
    print("="*70)
    
    # Update .mira_config.json to use project/models
    config_file = project_root / ".mira_config.json"
    config = {}
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    config["models_directory"] = str(project_models_dir)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  ✓ Updated .mira_config.json to use project/models")
    print(f"\n  Models are now centralized in: {project_models_dir}")
    print("\n  Note: Original HuggingFace cache is unchanged and safe.")
    print()


if __name__ == "__main__":
    main()
