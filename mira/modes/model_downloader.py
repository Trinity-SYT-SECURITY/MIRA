"""
Model Downloader Mode

Download comparison models and judge models from HuggingFace.
"""

from mira.analysis.comparison import ModelDownloader, COMPARISON_MODELS


# Judge/Embedding models for attack evaluation
JUDGE_MODELS = [
    {"name": "DistilBERT Judge", "hf_name": "distilbert-base-uncased-finetuned-sst-2-english", "size_gb": 0.3},
    {"name": "Toxic-BERT", "hf_name": "unitary/toxic-bert", "size_gb": 0.4},
    {"name": "MiniLM Embeddings", "hf_name": "sentence-transformers/all-MiniLM-L6-v2", "size_gb": 0.1},
    {"name": "BGE Embeddings", "hf_name": "BAAI/bge-base-en-v1.5", "size_gb": 0.4},
]


def run_model_downloader():
    """Run model downloader mode."""
    from mira.utils.model_manager import get_model_manager
    
    print("\n" + "="*70)
    print("  MODEL DOWNLOADER")
    print("="*70)
    
    # Get already downloaded models
    manager = get_model_manager()
    downloaded = manager.list_downloaded_models()
    
    # Choose category
    print("""
  What would you like to download?
  
    [1] Attack Models - Language models for security testing
    [2] Judge Models - Evaluation models for attack success
    [3] Both - Download all recommended models
    """)
    
    try:
        category = input("  Select category (1-3, default=1): ").strip()
        if category == "":
            category = "1"
    except:
        category = "1"
    
    print()
    
    if category == "1":
        download_attack_models(downloaded, manager)
    elif category == "2":
        download_judge_models(downloaded, manager)
    elif category == "3":
        download_attack_models(downloaded, manager)
        download_judge_models(downloaded, manager)


def download_attack_models(downloaded, manager):
    """Download attack models with selection interface."""
    print("  " + "="*60)
    print("  ATTACK MODELS (for security testing)")
    print("  " + "="*60 + "\n")
    
    # Filter already downloaded
    available = []
    for m in COMPARISON_MODELS:
        status = "✓" if m.hf_name in downloaded or m.name in downloaded else " "
        available.append((m, status))
    
    # Show models
    for i, (m, status) in enumerate(available):
        print(f"    [{i+1}] [{status}] {m.name:<20} {m.size_gb:.1f} GB  ({m.hf_name})")
    
    print(f"\n    [a] Download ALL attack models")
    print(f"    [s] Skip")
    print()
    
    print("  Select models to download:")
    print("    - Enter numbers separated by commas (e.g., 1,3,5)")
    print("    - Models marked [✓] are already downloaded")
    
    try:
        selection = input("\n  Your selection: ").strip().lower()
        
        if selection == 's':
            print("  Skipped attack models.")
            return
        elif selection == 'a':
            models_to_download = [m for m, status in available if status == " "]
        elif selection == '':
            print("  No models selected.")
            return
        else:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            models_to_download = [available[i][0] for i in indices if 0 <= i < len(available) and available[i][1] == " "]
    except:
        print("  Invalid input.")
        return
    
    if not models_to_download:
        print("\n  All selected models are already downloaded!")
        return
    
    print(f"\n  Will download {len(models_to_download)} models:")
    for m in models_to_download:
        print(f"    • {m.name} ({m.size_gb:.1f} GB)")
    
    confirm = input("\n  Continue? (y/n, default=y): ").strip().lower()
    if confirm == 'n':
        print("  Cancelled.")
        return
    
    # Download
    downloader = ModelDownloader()
    for m in models_to_download:
        print(f"\n  Downloading {m.name}...")
        try:
            downloader.download_model(m)
            print(f"  ✓ {m.name} downloaded")
        except Exception as e:
            print(f"  ✗ {m.name} failed: {e}")
    
    print(f"\n  ✓ Attack models download complete")


def download_judge_models(downloaded, manager):
    """Download judge/evaluation models."""
    print("\n  " + "="*60)
    print("  JUDGE MODELS (for attack evaluation)")
    print("  " + "="*60 + "\n")
    
    # Filter already downloaded
    available = []
    for m in JUDGE_MODELS:
        status = "✓" if m["hf_name"] in downloaded else " "
        available.append((m, status))
    
    # Show models
    for i, (m, status) in enumerate(available):
        print(f"    [{i+1}] [{status}] {m['name']:<20} {m['size_gb']:.1f} GB  ({m['hf_name']})")
    
    print(f"\n    [a] Download ALL judge models")
    print(f"    [s] Skip")
    print()
    
    print("  Select models to download:")
    
    try:
        selection = input("\n  Your selection: ").strip().lower()
        
        if selection == 's':
            print("  Skipped judge models.")
            return
        elif selection == 'a':
            models_to_download = [m for m, status in available if status == " "]
        elif selection == '':
            print("  No models selected.")
            return
        else:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            models_to_download = [available[i][0] for i in indices if 0 <= i < len(available) and available[i][1] == " "]
    except:
        print("  Invalid input.")
        return
    
    if not models_to_download:
        print("\n  All selected models are already downloaded!")
        return
    
    print(f"\n  Will download {len(models_to_download)} judge models:")
    for m in models_to_download:
        print(f"    • {m['name']} ({m['size_gb']:.1f} GB)")
    
    confirm = input("\n  Continue? (y/n, default=y): ").strip().lower()
    if confirm == 'n':
        print("  Cancelled.")
        return
    
    # Download using transformers
    from transformers import AutoModel, AutoTokenizer
    import os
    
    for m in models_to_download:
        print(f"\n  Downloading {m['name']}...")
        try:
            # Download to project/models
            model_dir = manager.models_dir / m['hf_name'].replace('/', '_')
            
            if 'sentence-transformers' in m['hf_name']:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(m['hf_name'])
                model.save(str(model_dir))
            else:
                tokenizer = AutoTokenizer.from_pretrained(m['hf_name'])
                model = AutoModel.from_pretrained(m['hf_name'])
                tokenizer.save_pretrained(str(model_dir))
                model.save_pretrained(str(model_dir))
            
            print(f"  ✓ {m['name']} downloaded")
        except Exception as e:
            print(f"  ✗ {m['name']} failed: {e}")
    
    print(f"\n  ✓ Judge models download complete")
