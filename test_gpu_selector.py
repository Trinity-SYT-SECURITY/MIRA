#!/usr/bin/env python3
"""
Test script to diagnose GPU model selector issue.
"""

import sys
from pathlib import Path

# Add MIRA to path
sys.path.insert(0, str(Path(__file__).parent))

from mira.utils.model_selector import get_available_target_models, ModelSelector

def test_gpu_detection():
    """Test GPU detection."""
    print("=" * 60)
    print("TEST 1: GPU Detection")
    print("=" * 60)
    
    selector = ModelSelector()
    print(f"Has GPU: {selector.has_gpu}")
    print(f"GPU Name: {selector.gpu_name}")
    print(f"VRAM: {selector.vram_gb:.1f} GB")
    print(f"RAM: {selector.ram_gb:.1f} GB")
    print()

def test_available_models():
    """Test available models detection."""
    print("=" * 60)
    print("TEST 2: Available Models")
    print("=" * 60)
    
    models = get_available_target_models()
    print(f"Total models found: {len(models)}")
    
    for model in models:
        print(f"\n  Model: {model.name}")
        print(f"    Category: {model.category}")
        print(f"    Size: {model.params}")
        print(f"    Min RAM: {model.min_ram_gb} GB")
        print(f"    Min VRAM: {model.min_vram_gb} GB")
    print()

def test_model_filtering():
    """Test model filtering logic."""
    print("=" * 60)
    print("TEST 3: Model Filtering")
    print("=" * 60)
    
    selector = ModelSelector()
    models = get_available_target_models()
    
    # Test compatible models
    compatible = selector.get_compatible_models(models)
    print(f"\nCompatible models: {len(compatible)}")
    for m in compatible:
        print(f"  - {m.name} ({m.category})")
    
    # Test recommended/advanced split
    recommended, advanced = selector.get_recommended_models(models)
    
    print(f"\n🎯 RECOMMENDED ({len(recommended)}):")
    for m in recommended:
        print(f"  - {m.name} ({m.category}, {m.params})")
    
    print(f"\n🚀 ADVANCED ({len(advanced)}):")
    for m in advanced:
        print(f"  - {m.name} ({m.category}, {m.params})")
    
    print()

def test_gpu_logic():
    """Test GPU-specific logic."""
    print("=" * 60)
    print("TEST 4: GPU Logic Check")
    print("=" * 60)
    
    selector = ModelSelector()
    models = get_available_target_models()
    compatible = selector.get_compatible_models(models)
    
    print(f"\nGPU Check: has_gpu={selector.has_gpu}, vram_gb={selector.vram_gb:.1f}")
    print(f"Condition (has_gpu and vram_gb >= 16): {selector.has_gpu and selector.vram_gb >= 16}")
    
    # Manually check what should be recommended
    if selector.has_gpu and selector.vram_gb >= 16:
        print("\n✅ Should use GPU mode (medium/large models)")
        medium_large = [m for m in compatible if m.category in ["medium", "large"]]
        print(f"   Medium/Large models available: {len(medium_large)}")
        for m in medium_large:
            print(f"     - {m.name} ({m.category})")
    else:
        print("\n❌ Should use CPU mode (tiny/small models)")
        tiny_small = [m for m in compatible if m.category in ["tiny", "small"]]
        print(f"   Tiny/Small models available: {len(tiny_small)}")
        for m in tiny_small:
            print(f"     - {m.name} ({m.category})")
    
    print()

if __name__ == "__main__":
    print("\n🔍 MIRA GPU Model Selector Diagnostic\n")
    
    try:
        test_gpu_detection()
        test_available_models()
        test_model_filtering()
        test_gpu_logic()
        
        print("=" * 60)
        print("✅ All tests completed")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
