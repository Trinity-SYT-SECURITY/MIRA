"""
Model Downloader Mode

Download comparison models from HuggingFace.
"""

from mira.analysis.comparison import ModelDownloader, COMPARISON_MODELS


def run_model_downloader():
    """Run model downloader mode."""
    print("\n" + "="*70)
    print("  MODEL DOWNLOADER")
    print("="*70 + "\n")
    
    # Show available models
    print("  Available models:\n")
    for m in COMPARISON_MODELS:
        print(f"    • {m.name:<25} {m.size_gb:.1f} GB  ({m.hf_name})")
    
    print()
    
    # Get max size
    try:
        max_size = input("  Max model size in GB (default: 2.0): ").strip()
        max_size = float(max_size) if max_size else 2.0
    except:
        max_size = 2.0
    
    # Filter models
    models = [m for m in COMPARISON_MODELS if m.size_gb <= max_size]
    
    print(f"\n  Will download {len(models)} models under {max_size} GB\n")
    
    confirm = input("  Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Cancelled.")
        return
    
    # Download
    downloader = ModelDownloader()
    downloaded = downloader.download_all(configs=models, max_size_gb=max_size)
    
    print(f"\n  ✓ Downloaded {len(downloaded)} models")
